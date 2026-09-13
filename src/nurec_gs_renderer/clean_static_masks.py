"""Build conservative per-frame dynamic masks for clean InstantNuRec static PLYs.

InstantNuRec normally relies on its predicted ``MOVABLE`` semantic class plus
NCore cuboids.  That is not sufficient for this clip: a moving work vehicle
and some pedestrians still enter the exported static layer.  This module
creates *input-domain* exclusion masks.  They are deliberately conservative:
high-confidence SegFormer road-agent pixels are masked even without a cuboid,
and every NCore cuboid classified as an agent/equipment or moving over the
chunk is masked with a padded FTheta projection.

The output is an immutable manifest plus PNG masks.  The companion local
InstantNuRec patch consumes this manifest and applies the masks after the same
camera resize/crop as RGB, before static Gaussian packaging.  The NCore source
clip and existing PLYs are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask
from .chunking import sample_chunk_frame_indices, sequence_sampling_interval
from .pose_sources import _create_ncore_loader, _ncore_exposure_timestamps
from .sky_build import dilate_boolean_mask
from .street_dataset import SegformerVehicleSegmenter, _interpolate_track_pose


_ALWAYS_DYNAMIC_LABELS = frozenset(
    {
        "pedestrian", "person", "person_group",
        "stroller", "rider", "bicycle", "bicycle_with_rider", "cyclist", "motorcycle", "motorcycle_with_rider",
        "cycle", "protruding_object", "construction_vehicle", "work_equipment",
    }
)
_SEMANTIC_DYNAMIC_LABELS = frozenset({"person", "car", "bus", "truck", "van", "bicycle", "motorcycle"})


@dataclass(frozen=True)
class CleanStaticMaskOptions:
    ncore_path: Path
    segformer_model_dir: Path
    output: Path
    chunk_index: int = 0
    model: str = "pa-front"
    device: str = "cuda"
    cuboid_margin_pixels: int = 12
    semantic_probability_threshold: float = 0.70
    semantic_margin_threshold: float = 0.10
    semantic_dilation_pixels: int = 4
    movement_threshold_m: float = 1.50
    instance_track_ids: tuple[int, ...] = ()
    instance_component_min_overlap_pixels: int = 16
    instance_component_dilation_pixels: int = 4
    resume: bool = False

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.cuboid_margin_pixels < 0 or self.semantic_dilation_pixels < 0:
            raise ValueError("chunk and mask pixel settings must be non-negative")
        if not 0.0 <= self.semantic_probability_threshold <= 1.0:
            raise ValueError("semantic_probability_threshold must be in [0,1]")
        if not -1.0 <= self.semantic_margin_threshold <= 1.0 or self.movement_threshold_m < 0.0:
            raise ValueError("semantic margin or movement threshold is invalid")
        if any(value < 0 for value in self.instance_track_ids) or len(set(self.instance_track_ids)) != len(self.instance_track_ids):
            raise ValueError("instance_track_ids must be unique non-negative integers")
        if self.instance_component_min_overlap_pixels <= 0 or self.instance_component_dilation_pixels < 0:
            raise ValueError("instance component parameters are invalid")
        if self.output.exists() and any(self.output.iterdir()) and not self.resume:
            raise FileExistsError(f"refusing to overwrite non-empty mask output: {self.output}")


class SegformerDynamicSegmenter(SegformerVehicleSegmenter):
    """Local-only high-confidence road-agent segmentation.

    It reuses the already validated model loader but resolves all dynamic ADE
    labels from checkpoint metadata rather than assuming class IDs.
    """

    def __init__(self, model_dir: Path, device: str) -> None:
        super().__init__(model_dir, device)
        self.dynamic_label_ids = tuple(
            sorted(
                int(index)
                for index, label in self.model.config.id2label.items()
                if str(label).strip().casefold() in _SEMANTIC_DYNAMIC_LABELS
            )
        )
        if not self.dynamic_label_ids:
            raise ValueError("SegFormer model has no supported dynamic labels")
        self.dynamic_labels = tuple(str(self.model.config.id2label[index]) for index in self.dynamic_label_ids)
        self.person_label_ids = tuple(
            sorted(
                int(index)
                for index, label in self.model.config.id2label.items()
                if str(label).strip().casefold() == "person"
            )
        )

    def dynamic_scores(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            probabilities = logits.softmax(dim=1)[0]
            vehicle = probabilities[list(self.vehicle_label_ids)].sum(dim=0)
            non_vehicle = probabilities.clone()
            non_vehicle[list(self.vehicle_label_ids)] = -1.0
            vehicle_margin = vehicle - non_vehicle.max(dim=0).values
            if self.person_label_ids:
                person = probabilities[list(self.person_label_ids)].sum(dim=0)
                non_person = probabilities.clone()
                non_person[list(self.person_label_ids)] = -1.0
                person_margin = person - non_person.max(dim=0).values
            else:
                person = torch.zeros_like(vehicle)
                person_margin = torch.zeros_like(vehicle)
        return tuple(value.float().cpu().numpy() for value in (vehicle, vehicle_margin, person, person_margin))


def _collect_dynamic_tracks(loader: object, start_us: int, end_us: int, *, movement_threshold_m: float) -> list[dict[str, Any]]:
    """Collect cuboids that can plausibly be a dynamic actor/equipment."""
    try:
        from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
        from instant_nurec.utils.types import HalfClosedInterval
    except ImportError as exc:  # pragma: no cover - installation specific
        raise RuntimeError("requires the local InstantNuRec checkout on PYTHONPATH") from exc
    raw_tracks = consolidate_cuboid_tracks(
        compute_cuboid_df(loader, HalfClosedInterval(start_us - 1_000_000, end_us + 1_000_000)),
        loader,
        ["AUTOLABEL"],
        0.0,
        np.eye(4),
    )
    selected: list[dict[str, Any]] = []
    for raw_id, track in raw_tracks.items():
        timestamps = np.asarray(track.get("timestamps_us"), dtype=np.int64)
        poses = np.asarray(track.get("poses"), dtype=np.float64)
        dimensions = np.asarray(track.get("dimension", []), dtype=np.float64).reshape(-1)
        if len(timestamps) < 2 or poses.shape != (len(timestamps), 4, 4) or dimensions.shape != (3,) or np.any(dimensions <= 0):
            continue
        label = str(track.get("label_class", "")).strip().casefold()
        movement = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
        if label not in _ALWAYS_DYNAMIC_LABELS and movement < movement_threshold_m:
            continue
        try:
            track_id: int | str = int(raw_id)
        except (TypeError, ValueError):
            track_id = str(raw_id)
        selected.append(
            {
                "track_id": track_id,
                "label": label,
                "movement_m": movement,
                "mask_reason": "agent_or_equipment" if label in _ALWAYS_DYNAMIC_LABELS else "moved_over_threshold",
                "timestamps_us": timestamps.tolist(),
                "actor_to_world": poses.tolist(),
                "length_width_height": dimensions.tolist(),
            }
        )
    return sorted(selected, key=lambda item: str(item["track_id"]))


def _track_mask(
    tracks: list[dict[str, Any]], timestamp_us: int, camera_to_world: np.ndarray, model: object, *,
    source_height: int, source_width: int, margin_pixels: int, device: str,
) -> np.ndarray:
    """Project all dynamic cuboids through the native FTheta calibration."""
    usable = [track for track in tracks if _interpolate_track_pose(track, timestamp_us) is not None]
    return _ftheta_cuboid_mask(
        usable, timestamp_us, camera_to_world, model,
        height=source_height, width=source_width, source_height=source_height, source_width=source_width,
        margin_pixels=margin_pixels, device=device,
    ).astype(bool)


def _tracks_for_labels(tracks: list[dict[str, Any]], labels: frozenset[str]) -> list[dict[str, Any]]:
    return [track for track in tracks if str(track["label"]).casefold() in labels]


def semantic_components_touching_seed(
    semantic: np.ndarray,
    seed: np.ndarray,
    *,
    min_overlap_pixels: int,
) -> np.ndarray:
    """Select 8-connected vehicle semantic components attached to one cuboid."""
    semantic = np.asarray(semantic, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    if semantic.shape != seed.shape:
        raise ValueError("semantic and seed masks must have matching shapes")
    if min_overlap_pixels <= 0:
        raise ValueError("min_overlap_pixels must be positive")
    if not semantic.any() or not seed.any():
        return np.zeros_like(semantic)
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "instance-track masking requires scipy; install the project with the '[instance]' extra"
        ) from exc
    labels, count = ndimage.label(semantic, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(semantic)
    touched = labels[seed]
    counts = np.bincount(touched, minlength=count + 1)
    selected = np.flatnonzero(counts >= min_overlap_pixels)
    selected = selected[selected != 0]
    return np.isin(labels, selected)


def _instance_track_masks(
    tracks: list[dict[str, Any]], track_ids: tuple[int, ...], timestamp_us: int,
    camera_to_world: np.ndarray, model: object, *, source_height: int, source_width: int,
    margin_pixels: int, vehicle_semantic: np.ndarray, component_min_overlap_pixels: int,
    component_dilation_pixels: int, device: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Return target-cuboid plus attached vehicle-silhouette exclusion masks."""
    output = np.zeros((source_height, source_width), dtype=bool)
    details: dict[str, dict[str, float]] = {}
    by_id = {int(track["track_id"]): track for track in tracks}
    missing = sorted(set(track_ids).difference(by_id))
    if missing:
        raise ValueError(f"selected instance track IDs are not dynamic tracks in this chunk: {missing}")
    for track_id in track_ids:
        track = by_id[int(track_id)]
        cuboid = _track_mask(
            [track], timestamp_us, camera_to_world, model, source_height=source_height, source_width=source_width,
            margin_pixels=margin_pixels, device=device,
        )
        component = semantic_components_touching_seed(
            vehicle_semantic, cuboid, min_overlap_pixels=component_min_overlap_pixels,
        )
        if component_dilation_pixels:
            component = dilate_boolean_mask(component, component_dilation_pixels)
        instance = cuboid | component
        output |= instance
        details[str(track_id)] = {
            "cuboid_fraction": float(cuboid.mean()),
            "vehicle_component_fraction": float(component.mean()),
            "instance_fraction": float(instance.mean()),
        }
    return output, details


def build_clean_static_masks(options: CleanStaticMaskOptions) -> Path:
    """Write masks matching one InstantNuRec profile/chunk and return manifest path."""
    try:
        from instant_nurec.pretrained import get_model_profile
    except ImportError as exc:  # pragma: no cover - installation specific
        raise RuntimeError("requires the local InstantNuRec checkout on PYTHONPATH") from exc
    profile = get_model_profile(options.model)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    loader = _create_ncore_loader(options.ncore_path)
    camera_ids = tuple(profile.context_camera_ids)
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in camera_ids}
    exposures = {camera_id: _ncore_exposure_timestamps(sensor) for camera_id, sensor in sensors.items()}
    timestamps = {camera_id: values[:, 1].astype(np.int64) for camera_id, values in exposures.items()}
    sequence_start, sequence_end = sequence_sampling_interval(loader, sensors, timestamps)
    selected, references, chunk_count = sample_chunk_frame_indices(
        timestamps, sequence_start, sequence_end, options.chunk_index,
        n_frames_per_chunk=profile.n_frames_per_sample,
        max_frame_gap_timestamp_us=750_000,
    )
    tracks = _collect_dynamic_tracks(
        loader, min(references), max(references), movement_threshold_m=options.movement_threshold_m
    )
    if options.instance_track_ids:
        available_ids = {int(track["track_id"]) for track in tracks}
        missing = sorted(set(options.instance_track_ids).difference(available_ids))
        if missing:
            raise ValueError(f"selected instance track IDs are unavailable in chunk {options.chunk_index}: {missing}")
    parameters = {
        "semantic_probability_threshold": options.semantic_probability_threshold,
        "semantic_margin_threshold": options.semantic_margin_threshold,
        "semantic_dilation_pixels": options.semantic_dilation_pixels,
        "cuboid_margin_pixels": options.cuboid_margin_pixels,
        "movement_threshold_m": options.movement_threshold_m,
        "instance_track_ids": list(options.instance_track_ids),
        "instance_component_min_overlap_pixels": options.instance_component_min_overlap_pixels,
        "instance_component_dilation_pixels": options.instance_component_dilation_pixels,
    }
    partial_path = output / "manifest.partial.json"
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if options.resume:
        if not partial_path.is_file():
            raise FileNotFoundError(f"cannot resume without partial manifest: {partial_path}")
        with partial_path.open("r", encoding="utf-8") as handle:
            partial = json.load(handle)
        expected = {
            "schema": "nurec-clean-static-mask", "version": 1, "profile": options.model,
            "chunk_index": options.chunk_index, "ncore_path": str(options.ncore_path.resolve()),
            "parameters": parameters, "camera_ids": list(camera_ids),
        }
        if any(partial.get(key) != value for key, value in expected.items()):
            raise ValueError(f"partial mask manifest does not match this run: {partial_path}")
        for entry in partial.get("entries", []):
            if not isinstance(entry, dict):
                raise ValueError(f"malformed partial mask entry: {partial_path}")
            key = (str(entry.get("camera_id")), int(entry.get("frame_index", -1)))
            path = output / str(entry.get("path", ""))
            if key[1] >= 0 and path.is_file():
                existing[key] = entry
    segmenter = SegformerDynamicSegmenter(options.segformer_model_dir, options.device)
    entries: list[dict[str, Any]] = []

    def write_manifest(path: Path, *, status: str) -> None:
        ordered = sorted(entries, key=lambda item: (camera_ids.index(str(item["camera_id"])), int(item["slot"])))
        manifest = {
            "schema": "nurec-clean-static-mask", "version": 1, "status": status,
            "ncore_path": str(options.ncore_path.resolve()), "profile": options.model,
            "chunk_index": options.chunk_index, "chunk_count": chunk_count, "camera_ids": list(camera_ids),
            "native_mask_value": "255 means exclude source pixel from static Gaussian export",
            "semantic_labels": list(segmenter.dynamic_labels), "tracks": tracks, "parameters": parameters,
            "entries": ordered,
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output, delete=False, suffix=".json") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            temporary = Path(handle.name)
        temporary.replace(path)
    print(
        f"Building clean-static masks: profile={options.model} chunk={options.chunk_index}/{chunk_count - 1} "
        f"cameras={len(camera_ids)} frames={profile.n_frames_per_sample} tracks={len(tracks)}",
        flush=True,
    )
    for camera_id in camera_ids:
        sensor = sensors[camera_id]
        native_width, native_height = map(int, sensor.model_parameters.resolution)
        from ncore.impl.sensors.camera import FThetaCameraModel

        model = FThetaCameraModel(sensor.model_parameters, device=options.device)
        for slot, frame_index in enumerate(selected[camera_id]):
            frame_index = int(frame_index)
            existing_entry = existing.get((camera_id, frame_index))
            if existing_entry is not None:
                entries.append(existing_entry)
                print(f"[{camera_id} {slot + 1:02d}/{profile.n_frames_per_sample}] frame={frame_index} resumed", flush=True)
                continue
            started = time.perf_counter()
            image = Image.fromarray(sensor.get_frame_image_array(frame_index), mode="RGB")
            vehicle_probability, vehicle_margin, person_probability, person_margin = segmenter.dynamic_scores(image)
            vehicle_semantic = (
                (vehicle_probability >= options.semantic_probability_threshold)
                & (vehicle_margin >= options.semantic_margin_threshold)
            )
            person_semantic = (
                (person_probability >= options.semantic_probability_threshold)
                & (person_margin >= options.semantic_margin_threshold)
            )
            if options.semantic_dilation_pixels:
                vehicle_semantic = dilate_boolean_mask(vehicle_semantic, options.semantic_dilation_pixels)
                person_semantic = dilate_boolean_mask(person_semantic, options.semantic_dilation_pixels)
            midpoint_us = int((int(exposures[camera_id][frame_index, 0]) + int(exposures[camera_id][frame_index, 1])) // 2)
            rig_to_world = np.asarray(
                loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint_us], dtype=np.uint64))
            )[0]
            from .camera import compose_ncore_camera_pose

            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            all_cuboid = _track_mask(
                tracks, midpoint_us, camera_to_world, model, source_height=native_height, source_width=native_width,
                margin_pixels=options.cuboid_margin_pixels, device=options.device,
            )
            agent_cuboid = _track_mask(
                _tracks_for_labels(tracks, _ALWAYS_DYNAMIC_LABELS), midpoint_us, camera_to_world, model,
                source_height=native_height, source_width=native_width, margin_pixels=options.cuboid_margin_pixels,
                device=options.device,
            )
            instance_mask, instance_details = _instance_track_masks(
                tracks, options.instance_track_ids, midpoint_us, camera_to_world, model,
                source_height=native_height, source_width=native_width, margin_pixels=options.cuboid_margin_pixels,
                vehicle_semantic=vehicle_semantic,
                component_min_overlap_pixels=options.instance_component_min_overlap_pixels,
                component_dilation_pixels=options.instance_component_dilation_pixels, device=options.device,
            ) if options.instance_track_ids else (np.zeros_like(agent_cuboid, dtype=bool), {})
            # Moving vehicles require both independent sources of evidence:
            # semantic silhouette plus its NCore cuboid.  The person/equipment
            # classes remain box-backed (or semantic-only for unlabelled small
            # people), because ADE's vehicle labels do not describe them.
            mask = (vehicle_semantic & all_cuboid) | person_semantic | agent_cuboid | instance_mask
            relative = Path("masks") / camera_id / f"{int(frame_index):06d}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(target)
            entries.append({
                    "camera_id": camera_id,
                    "frame_index": int(frame_index),
                    "slot": slot,
                    "reference_timestamp_us": int(references[slot]),
                    "timestamp_midpoint_us": midpoint_us,
                    "path": relative.as_posix(),
                    "mask_fraction": float(mask.mean()),
                    "vehicle_semantic_fraction": float(vehicle_semantic.mean()),
                    "person_semantic_fraction": float(person_semantic.mean()),
                    "moving_cuboid_fraction": float(all_cuboid.mean()),
                    "agent_equipment_cuboid_fraction": float(agent_cuboid.mean()),
                    "instance_track_mask_fraction": float(instance_mask.mean()),
                    "instance_track_details": instance_details,
                    "seconds": time.perf_counter() - started,
            })
            write_manifest(partial_path, status="partial")
            print(
                f"[{camera_id} {slot + 1:02d}/{profile.n_frames_per_sample}] frame={frame_index} "
                f"mask={mask.mean():.2%} vehicle∩cuboid={(vehicle_semantic & all_cuboid).mean():.2%} "
                f"agent={agent_cuboid.mean():.2%}",
                flush=True,
            )
    write_manifest(output / "manifest.json", status="complete")
    print(f"Wrote clean-static mask manifest: {(output / 'manifest.json').resolve()}", flush=True)
    return output / "manifest.json"
