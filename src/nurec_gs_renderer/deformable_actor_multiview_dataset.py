"""Materialise every accepted deformable multi-camera plan without manual picks.

Images are still rectified source supervision, but each record retains the
physical camera and midpoint pose.  This makes the output suitable for a later
visibility-aware canonical/deformable geometry trainer, unlike a directory of
unrelated actor crops.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .deformable_actor_multiview_plan import SCHEMA as PLAN_SCHEMA
from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose, pinhole_intrinsics
from .two_dgs_actor_dataset import _midpoint_pose, actor_centre_tile_angles
from .two_dgs_dataset import _rectify_binary_mask, _rectify_frame


SCHEMA = "ncore-deformable-actor-multiview-dataset"
VERSION = 1


@dataclass(frozen=True)
class DeformableActorMultiviewDatasetOptions:
    ncore_path: Path
    dynamic_dataset_manifest: Path
    multiview_plan: Path
    output: Path
    width: int = 480
    height: int = 270
    horizontal_fov_deg: float = 40.0
    minimum_mask_pixels: int = 800
    rectify_device: str = "cuda"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or not 20.0 <= self.horizontal_fov_deg < 120.0 or self.minimum_mask_pixels <= 0:
            raise ValueError("invalid multiview dataset options")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def select_multiview_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "complete":
        raise ValueError("multiview plan must be complete")
    candidates = [value for value in plan.get("candidates", []) if isinstance(value, dict)]
    if not candidates:
        raise RuntimeError("multiview plan has no accepted candidates")
    for candidate in candidates:
        if len(candidate.get("camera_ids", [])) != 2 or not candidate.get("observations") or not candidate.get("train_references") or not candidate.get("holdout_references"):
            raise ValueError("multiview candidate is incomplete")
    return candidates


def _write_candidate(
    *, candidate: dict[str, Any], dynamic: dict[str, Any], dynamic_root: Path, output: Path,
    loader: Any, tracks: dict[int, dict[str, Any]], intrinsic: np.ndarray, options: DeformableActorMultiviewDatasetOptions,
) -> dict[str, Any]:
    track_id = int(candidate["track_id"]); cameras = [str(value) for value in candidate["camera_ids"]]
    track = tracks[track_id]; start, end = int(candidate["reference_start"]), int(candidate["reference_end"])
    key = f"track{track_id:03d}-{'-'.join(value.replace('camera_', '').replace('_', '-') for value in cameras)}-{start:03d}-{end:03d}"
    root = output / key
    temporary_root = output / f".{key}.partial"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    images = temporary_root / "images"; images.mkdir(parents=True)
    frame_map = {int(value["reference_frame_index"]): value for value in dynamic.get("frames", []) if isinstance(value, dict)}
    train_refs = {int(value) for value in candidate["train_references"]}; holdout_refs = {int(value) for value in candidate["holdout_references"]}
    records: list[dict[str, Any]] = []
    try:
        for observation in candidate["observations"]:
            reference = int(observation["reference_frame"]); frame = frame_map.get(reference)
            if frame is None:
                raise ValueError(f"plan reference {reference} absent from dynamic dataset")
            camera_map = {str(value["camera_id"]): value for value in frame.get("cameras", []) if isinstance(value, dict)}
            for camera_id in cameras:
                camera = camera_map.get(camera_id)
                if camera is None:
                    raise ValueError(f"plan camera {camera_id} absent at reference {reference}")
                instance = next((value for value in camera.get("instances", []) if int(value.get("track_id", -1)) == track_id and int(value.get("mask_pixels", 0)) >= options.minimum_mask_pixels), None)
                if instance is None:
                    raise ValueError(f"plan mask unavailable for track {track_id}/{camera_id}/{reference}")
                timestamp = int(camera["timestamp_midpoint_us"]); root_to_world = _interpolate_track_pose(track, timestamp)
                if root_to_world is None:
                    raise RuntimeError(f"track {track_id} has no pose at {timestamp}")
                source_c2w = _midpoint_pose(camera["camera_to_world_start"], camera["camera_to_world_end"])
                yaw, pitch = actor_centre_tile_angles(root_to_world, source_c2w)
                if abs(yaw) >= 80.0 or abs(pitch) >= 55.0:
                    raise RuntimeError(f"accepted plan projects outside safe tile range: {track_id}/{camera_id}/{reference}")
                sensor = loader.get_camera_sensor(camera_id)
                raw = np.asarray(sensor.get_frame_image_array(int(camera["source_frame_index"])), dtype=np.uint8)
                with Image.open(dynamic_root / str(instance["mask"])) as opened:
                    source_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
                rectified, valid = _rectify_frame(sensor, raw, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch)
                mask = ((_rectify_binary_mask(sensor, source_mask, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch) != 0) & (valid != 0)).astype(np.uint8)
                if int(mask.sum()) < options.minimum_mask_pixels:
                    raise RuntimeError(f"rectified mask too small for accepted plan: {track_id}/{camera_id}/{reference}")
                filename = f"{reference:06d}_{camera_id}_track{track_id}.png"
                Image.fromarray(np.concatenate((rectified, mask[..., None] * 255), axis=-1), mode="RGBA").save(images / filename)
                records.append({"image": f"images/{filename}", "split": "train" if reference in train_refs else "holdout", "reference_frame": reference, "physical_camera_id": camera_id, "source_frame": int(camera["source_frame_index"]), "timestamp_us": timestamp, "mask_pixels": int(mask.sum()), "tile_yaw_deg": yaw, "tile_pitch_deg": pitch, "root_to_world": np.asarray(root_to_world, np.float64).tolist(), "source_camera_to_world_midpoint": source_c2w.tolist(), "source_camera_to_world_start": camera["camera_to_world_start"], "source_camera_to_world_end": camera["camera_to_world_end"], "lidar_pixels": int(instance.get("lidar_pixels", 0)), "lidar_depth": instance.get("lidar_depth")})
                print(f"Multiview deformable [{key}] reference={reference} camera={camera_id} mask={int(mask.sum())}", flush=True)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    records.sort(key=lambda value: (int(value["timestamp_us"]), str(value["physical_camera_id"])))
    timestamps = np.asarray([value["timestamp_us"] for value in records], dtype=np.int64); span = max(int(timestamps.max() - timestamps.min()), 1)
    for record in records:
        record["time_normalized"] = float((int(record["timestamp_us"]) - int(timestamps.min())) / span)
    if len({value["reference_frame"] for value in records if value["split"] == "train"}) < 2 or not any(value["split"] == "holdout" for value in records):
        raise RuntimeError("multiview dataset requires two train references and one holdout reference")
    dimensions = np.asarray(track.get("length_width_height"), dtype=np.float32)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
        raise ValueError(f"track {track_id} has invalid length_width_height")
    manifest = {"schema": SCHEMA, "version": VERSION, "status": "complete", "dynamic_dataset_manifest": str(dynamic_root / "manifest.json"), "track_id": track_id, "label": track.get("label"), "actor_family": "deformable", "actor_dimensions_lwh_m": dimensions.tolist(), "camera_ids": cameras, "reference_window": [start, end], "width": options.width, "height": options.height, "horizontal_fov_deg": options.horizontal_fov_deg, "plan_geometry": {"mean_baseline_m": candidate["mean_baseline_m"], "mean_triangulation_angle_deg": candidate["mean_triangulation_angle_deg"]}, "records": records, "limitations": ["Records retain original source-camera midpoint/start/end poses, but rectified images remain source supervision.", "This dataset is an input to a future visibility-aware deformable geometry trainer, not an L4 render or complete body mesh."]}
    (temporary_root / "ncore_deformable_actor_multiview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_root.replace(root)
    return {"key": key, "track_id": track_id, "camera_ids": cameras, "records": len(records), "train_records": sum(value["split"] == "train" for value in records), "holdout_records": sum(value["split"] == "holdout" for value in records), "manifest": str((root / "ncore_deformable_actor_multiview_manifest.json").resolve())}


def build_deformable_actor_multiview_dataset(options: DeformableActorMultiviewDatasetOptions) -> Path:
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty multiview dataset output: {output}")
    dynamic_path = options.dynamic_dataset_manifest.resolve(); dynamic = _read(dynamic_path); plan = _read(options.multiview_plan.resolve())
    if dynamic.get("schema") != "ncore-dynamic-reconstruction-dataset" or dynamic.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be complete")
    if Path(str(plan.get("dynamic_dataset_manifest"))).resolve() != dynamic_path:
        raise ValueError("multiview plan does not refer to --dynamic-dataset-manifest")
    if Path(str(dynamic.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic dataset NCore path does not match --ncore-path")
    candidates = select_multiview_candidates(plan); tracks = {int(value["track_id"]): value for value in dynamic.get("tracks", []) if isinstance(value, dict)}
    output.mkdir(parents=True); loader = _create_ncore_loader(options.ncore_path); intrinsic = pinhole_intrinsics(options.width, options.height, options.horizontal_fov_deg)
    result: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            result.append(_write_candidate(candidate=candidate, dynamic=dynamic, dynamic_root=dynamic_path.parent, output=output, loader=loader, tracks=tracks, intrinsic=intrinsic, options=options))
        except RuntimeError as error:
            rejected.append({"track_id": int(candidate["track_id"]), "camera_ids": candidate["camera_ids"], "reference_window": [candidate["reference_start"], candidate["reference_end"]], "reason": str(error)})
            print(f"Rejected multiview deformable candidate track={candidate['track_id']}: {error}", flush=True)
    if not result:
        raise RuntimeError("no multiview candidate survived final rectified-mask checks")
    summary = {"schema": SCHEMA, "version": VERSION, "status": "complete", "dynamic_dataset_manifest": str(dynamic_path), "multiview_plan": str(options.multiview_plan.resolve()), "datasets": result, "rejected": rejected, "limitations": ["All candidates are selected automatically from the plan; no manual camera/window selection is used.", "A candidate failing final rectified-mask checks is excluded rather than weakened by a lower threshold.", "This is source multi-view supervision only and must pass later geometry/visibility holdout before any target-camera composition."]}
    path = output / "manifest.json"; path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote deformable multiview datasets: {path} ({len(result)} windows)", flush=True)
    return path
