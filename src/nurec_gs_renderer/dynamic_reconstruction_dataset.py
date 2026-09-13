"""Build exact-NCore supervision for a separately reconstructed dynamic layer.

This is intentionally not another filter for InstantNuRec's three-keyframe
dynamic NPZ.  It writes observations that a new object-centric dynamic model
can train from directly: original FTheta images remain in NCore, while this
dataset records their rolling-shutter poses, track-stable instance masks and
actor-region LiDAR depths.  Rigid road vehicles and deformable road users are
kept in different actor families so that a later trainer cannot accidentally
apply an SE(3)-only model to pedestrians.

The writer never changes NCore data, PLYs or an existing renderer output.
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
from .camera import compose_ncore_camera_pose
from .chunking import chunk_time_range, sequence_sampling_interval
from .clean_static_masks import SegformerDynamicSegmenter, _collect_dynamic_tracks
from .pose_sources import _create_ncore_loader, _ncore_exposure_timestamps
from .sky_build import DEFAULT_CAMERA_IDS
from .static_leakage import _project_lidar_depth_ftheta
from .street_dataset import _nearest_index


SCHEMA = "ncore-dynamic-reconstruction-dataset"
VERSION = 1
RIGID_LABELS = frozenset({"automobile", "heavy_truck", "construction_vehicle", "work_equipment"})
DEFORMABLE_LABELS = frozenset({
    "pedestrian", "person", "person_group", "rider", "bicycle", "bicycle_with_rider",
    "cyclist", "motorcycle", "motorcycle_with_rider", "cycle", "stroller",
})


def actor_family(label: str) -> str | None:
    """Classify only labels for which this dataset has an explicit model family."""
    normalized = str(label).strip().casefold()
    if normalized in RIGID_LABELS:
        return "rigid"
    if normalized in DEFORMABLE_LABELS:
        return "deformable"
    return None


def semantic_track_mask(cuboid: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Require cuboid and semantic evidence; never train a whole rectangle."""
    cuboid = np.asarray(cuboid, dtype=bool)
    semantic = np.asarray(semantic, dtype=bool)
    if cuboid.shape != semantic.shape:
        raise ValueError("cuboid and semantic masks must have matching shapes")
    return cuboid & semantic


def sparse_depth_inside_mask(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact pixel centres and positive sparse depths for one instance."""
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("depth and mask must have matching shapes")
    yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 0.0))
    return np.stack((xx, yy), axis=1).astype(np.uint16), depth[yy, xx].astype(np.float32)


def select_chunk_reference_indices(
    end_timestamps_us: np.ndarray,
    *,
    sequence_start_us: int,
    sequence_end_us: int,
    chunk_index: int,
    frame_start: int | None,
    frame_end: int | None,
    stride: int,
) -> np.ndarray:
    """Select all reference-camera frames in an InstantNuRec-compatible chunk."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    layout = chunk_time_range(
        sequence_start_us, sequence_end_us, chunk_index,
        n_frames_per_chunk=18, max_frame_gap_timestamp_us=750_000,
    )
    timestamps = np.asarray(end_timestamps_us, dtype=np.int64)
    inclusive = layout.chunk_index + 1 == layout.chunk_count
    selected = np.flatnonzero(
        (timestamps >= layout.start_us) & (timestamps <= layout.end_us if inclusive else timestamps < layout.end_us)
    )
    selected = selected[slice(frame_start, frame_end, stride)]
    if not len(selected):
        raise ValueError("requested dynamic reconstruction frame window is empty")
    return selected.astype(np.int64)


@dataclass(frozen=True)
class DynamicReconstructionDatasetOptions:
    ncore_path: Path
    segformer_model_dir: Path
    output: Path
    chunk_index: int = 0
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    reference_camera_id: str = "camera_front_wide_120fov"
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    validation_stride: int = 10
    device: str = "cuda"
    cuboid_margin_pixels: int = 8
    vehicle_probability_threshold: float = 0.60
    vehicle_margin_threshold: float = 0.05
    person_probability_threshold: float = 0.70
    person_margin_threshold: float = 0.10
    minimum_mask_pixels: int = 24
    movement_threshold_m: float = 0.50
    write_lidar_depth: bool = True
    max_lidar_timestamp_delta_us: int = 100_000
    resume: bool = False

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or not self.camera_ids or self.reference_camera_id not in self.camera_ids:
            raise ValueError("chunk/camera selection is invalid")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.stride <= 0 or self.validation_stride <= 1 or self.minimum_mask_pixels <= 0:
            raise ValueError("stride, validation_stride and minimum_mask_pixels must be positive")
        if self.cuboid_margin_pixels < 0 or self.movement_threshold_m < 0 or self.max_lidar_timestamp_delta_us <= 0:
            raise ValueError("margin/movement/LiDAR parameters are invalid")
        for value in (self.vehicle_probability_threshold, self.person_probability_threshold):
            if not 0.0 <= value <= 1.0:
                raise ValueError("semantic probability thresholds must be in [0,1]")
        for value in (self.vehicle_margin_threshold, self.person_margin_threshold):
            if not -1.0 <= value <= 1.0:
                raise ValueError("semantic margin thresholds must be in [-1,1]")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def _integer_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep stable, modelled tracks and never silently use a generated ID."""
    selected: list[dict[str, Any]] = []
    for track in tracks:
        try:
            track_id = int(track["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        family = actor_family(str(track.get("label", "")))
        if family is None:
            continue
        copied = dict(track)
        copied["track_id"] = track_id
        copied["actor_family"] = family
        selected.append(copied)
    return sorted(selected, key=lambda item: int(item["track_id"]))


def _mask_path(camera_id: str, reference_index: int, track_id: int) -> Path:
    return Path("instance_masks") / camera_id / f"{reference_index:06d}" / f"track_{track_id}.png"


def build_dynamic_reconstruction_dataset(options: DynamicReconstructionDatasetOptions) -> Path:
    """Write a resumable cross-camera dynamic-supervision manifest."""
    from ncore.impl.sensors.camera import FThetaCameraModel

    output = options.output.resolve()
    partial_path = output / "manifest.partial.json"
    final_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()) and not options.resume:
        raise FileExistsError(f"refusing to overwrite non-empty dynamic dataset directory: {output}")
    if options.resume and not partial_path.is_file():
        raise FileNotFoundError(f"cannot resume without {partial_path}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "instance_masks").mkdir(exist_ok=True)
    if options.write_lidar_depth:
        (output / "lidar_depth").mkdir(exist_ok=True)

    loader = _create_ncore_loader(options.ncore_path)
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in options.camera_ids}
    reference_sensor = sensors[options.reference_camera_id]
    reference_exposures = _ncore_exposure_timestamps(reference_sensor)
    all_end_times = {
        camera_id: _ncore_exposure_timestamps(sensor)[:, 1].astype(np.int64)
        for camera_id, sensor in sensors.items()
    }
    sequence_start, sequence_end = sequence_sampling_interval(loader, sensors, all_end_times)
    selected = select_chunk_reference_indices(
        reference_exposures[:, 1], sequence_start_us=sequence_start, sequence_end_us=sequence_end,
        chunk_index=options.chunk_index, frame_start=options.frame_start, frame_end=options.frame_end,
        stride=options.stride,
    )
    reference_midpoints = (
        reference_exposures[selected, 0].astype(np.int64) + reference_exposures[selected, 1].astype(np.int64)
    ) // 2
    tracks = _integer_tracks(_collect_dynamic_tracks(
        loader, int(reference_midpoints.min()), int(reference_midpoints.max()),
        movement_threshold_m=options.movement_threshold_m,
    ))
    if not tracks:
        raise RuntimeError("no rigid or deformable dynamic tracks were found in this frame window")
    models = {camera_id: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera_id, sensor in sensors.items()}
    segmenter = SegformerDynamicSegmenter(options.segformer_model_dir, options.device)
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov") if options.write_lidar_depth else None

    completed: dict[int, dict[str, Any]] = {}
    if options.resume:
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        expected = {"schema": SCHEMA, "version": VERSION, "ncore_path": str(options.ncore_path.resolve()),
                    "chunk_index": options.chunk_index, "camera_ids": list(options.camera_ids),
                    "reference_camera_id": options.reference_camera_id}
        if any(partial.get(key) != value for key, value in expected.items()):
            raise ValueError("partial dynamic reconstruction manifest does not match this run")
        completed = {int(item["reference_frame_index"]): item for item in partial.get("frames", [])}

    def manifest(status: str) -> dict[str, Any]:
        frames = [completed[int(index)] for index in selected if int(index) in completed]
        return {
            "schema": SCHEMA, "version": VERSION, "status": status,
            "ncore_path": str(options.ncore_path.resolve()), "chunk_index": options.chunk_index,
            "camera_ids": list(options.camera_ids), "reference_camera_id": options.reference_camera_id,
            "reference_frame_indices": selected.tolist(), "tracks": tracks,
            "actor_families": {"rigid": sorted(RIGID_LABELS), "deformable": sorted(DEFORMABLE_LABELS)},
            "camera_model": "native_ncore_ftheta_with_exposure_start_end_poses",
            "parameters": {
                "cuboid_margin_pixels": options.cuboid_margin_pixels,
                "vehicle_probability_threshold": options.vehicle_probability_threshold,
                "vehicle_margin_threshold": options.vehicle_margin_threshold,
                "person_probability_threshold": options.person_probability_threshold,
                "person_margin_threshold": options.person_margin_threshold,
                "minimum_mask_pixels": options.minimum_mask_pixels,
                "movement_threshold_m": options.movement_threshold_m,
                "max_lidar_timestamp_delta_us": options.max_lidar_timestamp_delta_us,
            },
            "frames": frames,
            "limitations": [
                "Masks are semantic-and-cuboid supervision, not an automatic instance segmentation ground truth.",
                "Rigid actors may use one canonical SE(3) representation; deformable actors require a temporal deformation model.",
                "LiDAR is sparse and only stored inside accepted instance masks.",
                "This dataset is training input and is not itself a rendered dynamic asset.",
            ],
        }

    print(
        f"Building dynamic reconstruction dataset: frames={len(selected)} cameras={len(sensors)} tracks={len(tracks)} "
        f"(rigid={sum(track['actor_family'] == 'rigid' for track in tracks)}, "
        f"deformable={sum(track['actor_family'] == 'deformable' for track in tracks)})",
        flush=True,
    )
    for ordinal, reference_index in enumerate(selected, start=1):
        reference_index = int(reference_index)
        if reference_index in completed:
            print(f"[{ordinal}/{len(selected)}] reference={reference_index} resumed", flush=True)
            continue
        reference_time = int(reference_exposures[reference_index, 1])
        lidar_world: np.ndarray | None = None
        lidar_index: int | None = None
        lidar_time: int | None = None
        if lidar_sensor is not None:
            lidar_index = int(lidar_sensor.get_closest_frame_index(reference_time))
            lidar_time = int(lidar_sensor.get_frame_timestamp_us(lidar_index))
            if abs(lidar_time - reference_time) <= options.max_lidar_timestamp_delta_us:
                cloud = lidar_sensor.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
                lidar_world = np.asarray(cloud, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        camera_records: list[dict[str, Any]] = []
        for camera_id, sensor in sensors.items():
            started = time.perf_counter()
            exposures = _ncore_exposure_timestamps(sensor)
            source_index = _nearest_index(exposures[:, 1].astype(np.int64), reference_time)
            start_us, end_us = (int(value) for value in exposures[source_index])
            midpoint_us = (start_us + end_us) // 2
            rig_poses = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([start_us, end_us], dtype=np.uint64)))
            camera_to_world_start = compose_ncore_camera_pose(rig_poses[0], np.asarray(sensor.T_sensor_rig))
            camera_to_world_end = compose_ncore_camera_pose(rig_poses[1], np.asarray(sensor.T_sensor_rig))
            midpoint_pose = compose_ncore_camera_pose(
                np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint_us], dtype=np.uint64)))[0],
                np.asarray(sensor.T_sensor_rig),
            )
            source = np.asarray(sensor.get_frame_image_array(source_index), dtype=np.uint8)
            image = Image.fromarray(source, mode="RGB")
            vehicle_probability, vehicle_margin, person_probability, person_margin = segmenter.dynamic_scores(image)
            vehicle_semantic = (vehicle_probability >= options.vehicle_probability_threshold) & (vehicle_margin >= options.vehicle_margin_threshold)
            person_semantic = (person_probability >= options.person_probability_threshold) & (person_margin >= options.person_margin_threshold)
            source_h, source_w = source.shape[:2]
            lidar_depth = None if lidar_world is None else _project_lidar_depth_ftheta(
                lidar_world, np.linalg.inv(midpoint_pose), models[camera_id],
                source_width=source_w, source_height=source_h, width=source_w, height=source_h, device=options.device,
            )
            instances: list[dict[str, Any]] = []
            union = np.zeros((source_h, source_w), dtype=bool)
            for track in tracks:
                cuboid = _ftheta_cuboid_mask(
                    [track], midpoint_us, midpoint_pose, models[camera_id], height=source_h, width=source_w,
                    source_height=source_h, source_width=source_w, margin_pixels=options.cuboid_margin_pixels,
                    device=options.device,
                ).astype(bool)
                semantic = vehicle_semantic if track["actor_family"] == "rigid" else person_semantic
                mask = semantic_track_mask(cuboid, semantic)
                pixels = int(mask.sum())
                if pixels < options.minimum_mask_pixels:
                    continue
                track_id = int(track["track_id"])
                relative = _mask_path(camera_id, reference_index, track_id)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(target)
                record: dict[str, Any] = {
                    "track_id": track_id, "actor_family": track["actor_family"], "label": track["label"],
                    "mask": relative.as_posix(), "mask_pixels": pixels,
                    "mask_fraction": float(mask.mean()), "cuboid_fraction": float(cuboid.mean()),
                }
                if lidar_depth is not None:
                    xy, depth = sparse_depth_inside_mask(lidar_depth, mask)
                    if len(depth):
                        depth_relative = Path("lidar_depth") / camera_id / f"{reference_index:06d}" / f"track_{track_id}.npz"
                        depth_target = output / depth_relative
                        depth_target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(depth_target, xy=xy, depth_m=depth)
                        record["lidar_depth"] = depth_relative.as_posix()
                        record["lidar_pixels"] = int(len(depth))
                    else:
                        record["lidar_pixels"] = 0
                instances.append(record)
                union |= mask
            union_relative = Path("union_masks") / camera_id / f"{reference_index:06d}.png"
            union_target = output / union_relative
            union_target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(union_target)
            camera_records.append({
                "camera_id": camera_id, "source_frame_index": int(source_index),
                "timestamp_start_us": start_us, "timestamp_end_us": end_us, "timestamp_midpoint_us": midpoint_us,
                "source_resolution": [source_w, source_h], "camera_to_world_start": camera_to_world_start.tolist(),
                "camera_to_world_end": camera_to_world_end.tolist(), "union_mask": union_relative.as_posix(),
                "union_mask_fraction": float(union.mean()), "instances": instances,
                "lidar_frame_index": lidar_index, "lidar_timestamp_us": lidar_time,
                "seconds": time.perf_counter() - started,
            })
        completed[reference_index] = {
            "reference_frame_index": reference_index, "reference_timestamp_end_us": reference_time,
            "split": "validation" if ordinal % options.validation_stride == 0 else "train", "cameras": camera_records,
        }
        _atomic_json(partial_path, manifest("partial"))
        accepted = sum(len(camera["instances"]) for camera in camera_records)
        print(f"[{ordinal}/{len(selected)}] reference={reference_index} actor_observations={accepted}", flush=True)
    _atomic_json(final_path, manifest("complete"))
    partial_path.unlink(missing_ok=True)
    print(f"Wrote dynamic reconstruction dataset: {final_path}", flush=True)
    return final_path
