"""Build short-window, time-conditioned supervision for deformable actors.

This produces rectified source images, masks and coarse root-frame poses for
an eventual deformation-aware model.  It deliberately does *not* emit a
COLMAP static scene, a Gaussian PLY or a target-view renderer: treating a
person's articulated body as one rigid canonical 2DGS asset was ruled out by
the project's earlier actor experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose, pinhole_intrinsics
from .two_dgs_actor_dataset import _midpoint_pose, actor_centre_tile_angles
from .two_dgs_dataset import _rectify_binary_mask, _rectify_frame


SCHEMA = "ncore-deformable-actor-time-conditioned-proxy"
VERSION = 1


@dataclass(frozen=True)
class DeformableActorDatasetOptions:
    ncore_path: Path
    dynamic_dataset_manifest: Path
    deformable_plan: Path
    output: Path
    track_id: int
    source_camera: str
    reference_start: int | None = None
    reference_end: int | None = None
    width: int = 480
    height: int = 270
    horizontal_fov_deg: float = 40.0
    minimum_mask_pixels: int = 800
    holdout_every: int = 5
    max_observations: int | None = None
    rectify_device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or not self.source_camera or self.width <= 0 or self.height <= 0:
            raise ValueError("track/source-camera/dimensions are invalid")
        if not 20.0 <= self.horizontal_fov_deg < 120.0 or self.minimum_mask_pixels <= 0 or self.holdout_every < 2:
            raise ValueError("invalid deformable proxy thresholds")
        if self.max_observations is not None and self.max_observations < 3:
            raise ValueError("max_observations must be at least three when set")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def select_deformable_plan_window(
    plan: dict[str, Any], *, track_id: int, source_camera: str,
    reference_start: int | None, reference_end: int | None,
) -> dict[str, Any]:
    """Require the requested short source window to exist in the trusted plan."""
    if plan.get("schema") != "ncore-deformable-actor-window-plan" or plan.get("status") != "complete":
        raise ValueError("deformable plan must be complete")
    candidates = [value for value in plan.get("candidates", []) if isinstance(value, dict) and int(value.get("track_id", -1)) == track_id and str(value.get("camera_id")) == source_camera]
    if reference_start is not None:
        candidates = [value for value in candidates if int(value.get("reference_start", -1)) == reference_start]
    if reference_end is not None:
        candidates = [value for value in candidates if int(value.get("reference_end", -1)) == reference_end]
    if len(candidates) != 1:
        raise ValueError("requested track/camera/window must match exactly one deformable-plan candidate")
    return candidates[0]


def build_deformable_actor_dataset(options: DeformableActorDatasetOptions) -> Path:
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty deformable actor dataset: {output}")
    dynamic_path = options.dynamic_dataset_manifest.resolve(); dynamic = _read(dynamic_path)
    if dynamic.get("schema") != "ncore-dynamic-reconstruction-dataset" or dynamic.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be complete")
    if Path(str(dynamic.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic dataset NCore path does not match --ncore-path")
    plan = _read(options.deformable_plan.resolve())
    if Path(str(plan.get("dynamic_dataset_manifest"))).resolve() != dynamic_path:
        raise ValueError("deformable plan does not refer to --dynamic-dataset-manifest")
    window = select_deformable_plan_window(plan, track_id=options.track_id, source_camera=options.source_camera, reference_start=options.reference_start, reference_end=options.reference_end)
    tracks = {int(value["track_id"]): value for value in dynamic.get("tracks", []) if isinstance(value, dict) and "track_id" in value}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "deformable":
        raise ValueError("requested track is not a deformable actor in the dynamic dataset")
    start, end = int(window["reference_start"]), int(window["reference_end"])
    intrinsic = pinhole_intrinsics(options.width, options.height, options.horizontal_fov_deg)
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for frame in dynamic.get("frames", []):
        reference = int(frame.get("reference_frame_index", -1))
        if not start <= reference < end:
            continue
        for camera in frame.get("cameras", []):
            if str(camera.get("camera_id")) != options.source_camera:
                continue
            for instance in camera.get("instances", []):
                if int(instance.get("track_id", -1)) == options.track_id and int(instance.get("mask_pixels", 0)) >= options.minimum_mask_pixels:
                    candidates.append((reference, camera, instance))
    candidates.sort(key=lambda value: value[0])
    if options.max_observations is not None and len(candidates) > options.max_observations:
        indices = np.linspace(0, len(candidates) - 1, options.max_observations, dtype=np.int64)
        candidates = [candidates[int(index)] for index in indices]
    if len(candidates) < 3:
        raise RuntimeError("fewer than three planned deformable observations remain after mask filtering")
    output.mkdir(parents=True); images = output / "images"; images.mkdir()
    loader = _create_ncore_loader(options.ncore_path)
    records: list[dict[str, Any]] = []
    root_poses: list[np.ndarray] = []
    for reference, record, instance in candidates:
        timestamp = int(record["timestamp_midpoint_us"])
        root_to_world = _interpolate_track_pose(track, timestamp)
        if root_to_world is None:
            continue
        sensor = loader.get_camera_sensor(options.source_camera)
        source_c2w = _midpoint_pose(record["camera_to_world_start"], record["camera_to_world_end"])
        try:
            yaw, pitch = actor_centre_tile_angles(root_to_world, source_c2w)
        except ValueError:
            continue
        if abs(yaw) >= 80.0 or abs(pitch) >= 55.0:
            continue
        raw = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), dtype=np.uint8)
        mask_path = dynamic_path.parent / str(instance["mask"])
        with Image.open(mask_path) as opened:
            source_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
        rectified, valid = _rectify_frame(sensor, raw, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch)
        mask = _rectify_binary_mask(sensor, source_mask, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch)
        mask = ((mask != 0) & (valid != 0)).astype(np.uint8)
        if int(mask.sum()) < options.minimum_mask_pixels:
            continue
        filename = f"{reference:06d}_{options.source_camera}_track{options.track_id}.png"
        Image.fromarray(np.concatenate((rectified, mask[..., None] * 255), axis=-1), mode="RGBA").save(images / filename)
        split = "holdout" if reference % options.holdout_every == 0 else "train"
        records.append({
            "image": f"images/{filename}", "split": split, "reference_frame": reference,
            "source_camera": options.source_camera, "source_frame": int(record["source_frame_index"]),
            "timestamp_us": timestamp, "time_normalized": 0.0, "tile_yaw_deg": yaw, "tile_pitch_deg": pitch,
            "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
            "root_to_world": np.asarray(root_to_world, np.float64).tolist(),
            "source_camera_to_world_midpoint": source_c2w.tolist(),
        })
        root_poses.append(np.asarray(root_to_world, np.float32))
        print(f"Deformable actor [{len(records)}/{len(candidates)}] reference={reference} mask={int(mask.sum())}", flush=True)
    if sum(value["split"] == "train" for value in records) < 2 or sum(value["split"] == "holdout" for value in records) < 1:
        raise RuntimeError("deformable dataset needs at least two train and one temporal holdout observation")
    timestamps = np.asarray([value["timestamp_us"] for value in records], np.int64)
    span = max(int(timestamps[-1] - timestamps[0]), 1)
    for value in records:
        value["time_normalized"] = float((int(value["timestamp_us"]) - int(timestamps[0])) / span)
    np.savez_compressed(output / "temporal_metadata.npz", timestamps_us=timestamps, time_normalized=np.asarray([value["time_normalized"] for value in records], np.float32), root_to_world=np.stack(root_poses).astype(np.float32), intrinsics=intrinsic.astype(np.float32))
    manifest = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "ncore_path": str(options.ncore_path.resolve()),
        "dynamic_dataset_manifest": str(dynamic_path), "deformable_plan": str(options.deformable_plan.resolve()),
        "track_id": options.track_id, "label": track.get("label"), "actor_family": "deformable",
        "source_camera": options.source_camera, "reference_window": [start, end], "width": options.width, "height": options.height,
        "horizontal_fov_deg": options.horizontal_fov_deg, "temporal_metadata": "temporal_metadata.npz", "records": records,
        "limitations": [
            "Root cuboid poses are coarse body anchors only; they do not make a person rigid or canonical.",
            "Images are FTheta-rectified midpoint-pinhole proxies. A trainer must model time-dependent deformation and camera-conditioned appearance.",
            "This is source-view supervision only: it contains no target-L4 render, Gaussian asset or novel-view person claim.",
        ],
    }
    path = output / "ncore_deformable_actor_manifest.json"; path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote deformable time-conditioned actor dataset: {path} (train={sum(value['split'] == 'train' for value in records)} holdout={sum(value['split'] == 'holdout' for value in records)})", flush=True)
    return path
