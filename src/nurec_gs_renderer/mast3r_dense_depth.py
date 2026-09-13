"""Audit MASt3R metric dense depth before using it for a dynamic object layer.

Sparse descriptor triangulation is insufficient for a complete vehicle.  This
module instead treats MASt3R metric depth as a proposal along *native NCore
FTheta rolling-shutter rays*, calibrates its range against only raw actor-mask
LiDAR, and refuses output unless the cuboid/LiDAR checks are credible.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .mast3r_correspondence import _load_mast3r, _mast3r_view
from .pose_sources import _create_ncore_loader
from .roma_multiview_audit import _native_world_rays, _rectify_tangent_patch, pixels_in_mask
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-mast3r-metric-dense-depth-audit"
DATASET_SCHEMA = "ncore-dynamic-reconstruction-dataset"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((item for item in record["instances"] if int(item["track_id"]) == track_id), None)


def metric_range_scale(predicted_range: np.ndarray, lidar_range: np.ndarray) -> tuple[float, float]:
    """Robust single-scale calibration and median relative residual."""
    predicted, lidar = np.asarray(predicted_range, np.float32), np.asarray(lidar_range, np.float32)
    valid = np.isfinite(predicted) & np.isfinite(lidar) & (predicted > .05) & (lidar > .05)
    if int(valid.sum()) < 3:
        raise ValueError("at least three finite metric/LiDAR pairs are required")
    scale = float(np.median(lidar[valid] / predicted[valid]))
    error = np.abs(predicted[valid] * scale - lidar[valid]) / lidar[valid]
    return scale, float(np.median(error))


def _dense_prediction(model: Any, patch_a: np.ndarray, patch_b: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return MASt3R's dense first-view metric point/range proposal."""
    import torch
    with torch.inference_mode():
        first, second = _mast3r_view(patch_a, "a"), _mast3r_view(patch_b, "b")
        shape_a, shape_b = torch.from_numpy(first["true_shape"]).to(device), torch.from_numpy(second["true_shape"]).to(device)
        feature_a, feature_b, position_a, position_b = model._encode_image_pairs(first["img"].to(device), second["img"].to(device), shape_a, shape_b)
        decoder_a, _ = model._decoder(feature_a, position_a, feature_b, position_b)
        with torch.amp.autocast("cuda", enabled=False):
            result = model._downstream_head(1, [token.float() for token in decoder_a], shape_a)
    points = result["pts3d"][0].detach().float().cpu().numpy().astype(np.float32)
    confidence = result.get("conf")
    confidence_np = np.ones(points.shape[:2], np.float32) if confidence is None else confidence[0].detach().float().cpu().numpy().astype(np.float32)
    return points, confidence_np


def _nearest_patch_indices(raw_map: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map sparse raw LiDAR pixels to their nearest valid tangent-patch pixel."""
    mapping = np.asarray(raw_map, np.float32).reshape(-1, 2)
    finite = np.isfinite(mapping).all(axis=1)
    source_indices = np.flatnonzero(finite)
    source = mapping[finite]
    target = np.asarray(xy, np.float32)
    chosen = np.full(len(target), -1, np.int64)
    distance = np.full(len(target), np.inf, np.float32)
    for index, point in enumerate(target):
        squared = ((source - point[None]) ** 2).sum(axis=1)
        best = int(np.argmin(squared))
        chosen[index], distance[index] = source_indices[best], float(np.sqrt(squared[best]))
    return chosen, distance


@dataclass(frozen=True)
class Mast3rDenseDepthAuditOptions:
    dataset_manifest: Path
    mast3r_root: Path
    mast3r_weights: Path
    output: Path
    track_id: int
    camera_a: str = "camera_cross_right_120fov"
    camera_b: str = "camera_front_wide_120fov"
    reference_index: int = 140
    patch_size: int = 512
    patch_fov_deg: float = 35.0
    patch_stride: int = 4
    min_confidence: float = 1.0
    max_lidar_pixel_distance: float = 3.0
    max_median_lidar_relative_error: float = .25
    min_cuboid_fraction: float = .75
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.reference_index < 0 or self.patch_size < 128 or self.patch_stride <= 0:
            raise ValueError("track/reference/patch settings are invalid")
        if self.output.suffix != ".json" or not 1 < self.patch_fov_deg < 170:
            raise ValueError("output and FOV are invalid")
        if self.min_confidence < 0 or self.max_lidar_pixel_distance <= 0 or self.max_median_lidar_relative_error <= 0 or not 0 < self.min_cuboid_fraction <= 1:
            raise ValueError("acceptance thresholds are invalid")


def audit_mast3r_dense_depth(options: Mast3rDenseDepthAuditOptions) -> dict[str, Any]:
    if options.output.exists() or options.output.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite dense-depth audit output")
    manifest_path = options.dataset_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset manifest must be complete")
    tracks = {int(item["track_id"]): item for item in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("dense metric audit currently requires a listed rigid actor")
    frame = next((item for item in manifest["frames"] if int(item["reference_frame_index"]) == options.reference_index), None)
    if frame is None:
        raise ValueError("requested reference is absent from dynamic dataset")
    records = {str(item["camera_id"]): item for item in frame["cameras"]}
    record_a, record_b = records.get(options.camera_a), records.get(options.camera_b)
    if record_a is None or record_b is None:
        raise ValueError("requested camera pair absent from reference")
    instance_a, instance_b = _instance(record_a, options.track_id), _instance(record_b, options.track_id)
    if instance_a is None or instance_b is None:
        raise ValueError("requested track is not masked in both source cameras")
    root = manifest_path.parent
    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    from ncore.impl.sensors.camera import FThetaCameraModel
    sensor_a, sensor_b = loader.get_camera_sensor(options.camera_a), loader.get_camera_sensor(options.camera_b)
    native_a = FThetaCameraModel(sensor_a.model_parameters, device=options.device)
    native_b = FThetaCameraModel(sensor_b.model_parameters, device=options.device)
    image_a = np.asarray(sensor_a.get_frame_image_array(int(record_a["source_frame_index"])), np.uint8)
    image_b = np.asarray(sensor_b.get_frame_image_array(int(record_b["source_frame_index"])), np.uint8)
    mask_a = np.asarray(Image.open(root / str(instance_a["mask"])).convert("L"), np.uint8) > 0
    mask_b = np.asarray(Image.open(root / str(instance_b["mask"])).convert("L"), np.uint8) > 0
    centre_a = np.asarray(np.nonzero(mask_a)[::-1], np.float32).mean(axis=1)
    centre_b = np.asarray(np.nonzero(mask_b)[::-1], np.float32).mean(axis=1)
    patch_a, raw_map_a, valid_a = _rectify_tangent_patch(native_a, image_a, centre_a, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
    patch_b, _, _ = _rectify_tangent_patch(native_b, image_b, centre_b, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
    model = _load_mast3r(options.mast3r_root.resolve(), options.mast3r_weights.resolve(), options.device)
    points_patch, confidence = _dense_prediction(model, patch_a, patch_b, options.device)
    if points_patch.shape[:2] != raw_map_a.shape[:2]:
        raise RuntimeError(f"MASt3R/map shape mismatch: {points_patch.shape[:2]} != {raw_map_a.shape[:2]}")
    flat_map, flat_points, flat_conf = raw_map_a.reshape(-1, 2), points_patch.reshape(-1, 3), confidence.reshape(-1)
    inside = pixels_in_mask(mask_a, np.nan_to_num(flat_map, nan=-1.0))
    keep = np.isfinite(flat_map).all(axis=1) & inside & (flat_conf >= options.min_confidence)
    grid = np.arange(len(keep))
    keep &= (grid % options.patch_stride) == 0
    raw_pixels, predicted_range = flat_map[keep], np.linalg.norm(flat_points[keep], axis=1)
    origins, rays = _native_world_rays(native_a, raw_pixels, record_a)
    lidar_xy = np.empty((0, 2), np.float32); lidar_z = np.empty(0, np.float32)
    if isinstance(instance_a.get("lidar_depth"), str):
        with np.load(root / str(instance_a["lidar_depth"]), allow_pickle=False) as archive:
            lidar_xy, lidar_z = np.asarray(archive["xy"], np.float32), np.asarray(archive["depth_m"], np.float32)
    match, distance = _nearest_patch_indices(raw_pixels, lidar_xy) if len(lidar_xy) else (np.empty(0, np.int64), np.empty(0, np.float32))
    calib = (match >= 0) & (distance <= options.max_lidar_pixel_distance)
    c2w_start = np.asarray(record_a["camera_to_world_start"], np.float32)
    ray_forward = np.maximum(rays[match[calib]] @ c2w_start[:3, 2], 1e-4) if calib.any() else np.empty(0, np.float32)
    lidar_range = lidar_z[calib] / ray_forward if calib.any() else np.empty(0, np.float32)
    scale, lidar_error = metric_range_scale(predicted_range[match[calib]], lidar_range)
    world = origins + rays * (predicted_range * scale)[:, None]
    timestamp = int(record_a["timestamp_midpoint_us"])
    pose = _interpolate_track_pose(track, timestamp)
    if pose is None:
        raise RuntimeError("track has no pose at requested timestamp")
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    half = np.asarray(track["length_width_height"], np.float32) * .5 + .15
    in_cuboid = np.all(np.abs(local) <= half[None], axis=1)
    accepted = bool(lidar_error <= options.max_median_lidar_relative_error and float(in_cuboid.mean()) >= options.min_cuboid_fraction)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.output.with_suffix(".npz"), points_actor_local=local[in_cuboid], colors=image_a[np.clip(np.rint(raw_pixels[in_cuboid, 1]).astype(np.int64), 0, image_a.shape[0]-1), np.clip(np.rint(raw_pixels[in_cuboid, 0]).astype(np.int64), 0, image_a.shape[1]-1)], confidence=flat_conf[keep][in_cuboid], reference_index=np.asarray([options.reference_index], np.int32))
    report = {"schema": SCHEMA, "status": "complete", "accepted": accepted, "asset": str(options.output.with_suffix(".npz").resolve()), "track_id": options.track_id, "reference_index": options.reference_index, "camera_ids": [options.camera_a, options.camera_b], "dense": {"masked_candidates": int(len(raw_pixels)), "inside_cuboid": int(in_cuboid.sum()), "cuboid_fraction": float(in_cuboid.mean()), "patch_valid_fraction": float(valid_a.mean())}, "lidar_calibration": {"samples": int(calib.sum()), "scale": scale, "median_relative_range_error": lidar_error}, "thresholds": {"max_median_lidar_relative_error": options.max_median_lidar_relative_error, "min_cuboid_fraction": options.min_cuboid_fraction}, "limitations": ["Dense metric depth is a frozen proposal and is not rendered by this audit.", "Only a passing LiDAR/cuboid audit may be considered for a per-frame actor layer; unobserved surfaces remain unknown."]}
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MASt3R dense metric audit ref={options.reference_index} candidates={len(raw_pixels)} cuboid={in_cuboid.mean():.3%} lidar_rel={lidar_error:.3f} accepted={accepted}", flush=True)
    return report
