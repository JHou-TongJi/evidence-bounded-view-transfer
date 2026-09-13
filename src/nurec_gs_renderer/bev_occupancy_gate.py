"""Audit a conservative LiDAR-BEV occupancy gate for a dynamic actor.

This is intentionally a *diagnostic*, not another image-generation model.  It
uses only a canonical cloud fused from raw actor-mask LiDAR on train frames.
On temporally held-out frames it ray-marches the exact NCore FTheta/rolling
rays through actor-local occupied BEV pillars with vertical bins.  The output
answers a falsifiable question: can observed occupancy reject background
coverage without making the dynamic instance disappear?

It never reads a static PLY/depth, never consumes DAV2, and never fills cells
that lack raw LiDAR support.  Therefore an insufficient result is a useful
failure: it prevents a sparse BEV grid from being mistaken for a complete car.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .canonical_actor import CanonicalActorAsset
from .pose_sources import _create_ncore_loader
from .roma_multiview_audit import _native_world_rays
from .street_dataset import _interpolate_track_pose


TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"
SCHEMA = "ncore-lidar-bev-occupancy-gate-audit"
VERSION = 1


@dataclass(frozen=True)
class BevOccupancyGateOptions:
    trusted_manifest: Path
    actor_asset: Path
    output: Path
    track_id: int
    camera_ids: tuple[str, ...]
    reference_start: int
    reference_end: int
    reference_parity: int = 1
    device: str = "cuda"
    voxel_size_m: float = .10
    dilation_cells_xy: int = 1
    dilation_cells_z: int = 1
    roi_margin_pixels: int = 16

    def __post_init__(self) -> None:
        if self.output.suffix != ".json" or self.track_id < 0:
            raise ValueError("output must be a .json path and track_id non-negative")
        if not self.camera_ids or len(self.camera_ids) != len(set(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique")
        if self.reference_start < 0 or self.reference_end <= self.reference_start or self.reference_parity not in {0, 1}:
            raise ValueError("invalid reference range/parity")
        if self.voxel_size_m <= 0 or self.dilation_cells_xy < 0 or self.dilation_cells_z < 0 or self.roi_margin_pixels < 0:
            raise ValueError("invalid BEV voxel/dilation/ROI parameters")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _voxel_hash(keys: np.ndarray) -> np.ndarray:
    """Hash small actor-local voxel coordinates without Python per-ray loops."""
    key = np.asarray(keys, np.int64)
    if key.ndim != 2 or key.shape[1] != 3 or np.any(np.abs(key) >= 10_000):
        raise ValueError("actor-local voxel keys must be [N,3] inside the supported range")
    stride = np.int64(20_001)
    return ((key[:, 0] + 10_000) * stride + (key[:, 1] + 10_000)) * stride + (key[:, 2] + 10_000)


def dilated_bev_voxels(positions: np.ndarray, *, voxel_size_m: float, dilation_xy: int, dilation_z: int) -> np.ndarray:
    """Build actor-local BEV pillars with a bounded vertical occupancy band."""
    points = np.asarray(positions, np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("positions must be a non-empty [N,3] array")
    base = np.unique(np.floor(points / voxel_size_m).astype(np.int64), axis=0)
    offsets = np.asarray([(x, y, z) for x in range(-dilation_xy, dilation_xy + 1)
                          for y in range(-dilation_xy, dilation_xy + 1)
                          for z in range(-dilation_z, dilation_z + 1)], np.int64)
    return np.unique((base[:, None, :] + offsets[None, :, :]).reshape(-1, 3), axis=0)


def _local_ray_segments(origins: np.ndarray, directions: np.ndarray, poses: np.ndarray,
                        dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return actor-local ray origin/direction and entry/exit distances to one cuboid."""
    world_o, world_d, actor = (np.asarray(value, np.float32) for value in (origins, directions, poses))
    size = np.asarray(dimensions, np.float32)
    if world_o.ndim != 2 or world_o.shape[1] != 3 or world_d.shape != world_o.shape or actor.shape != (len(world_o), 4, 4):
        raise ValueError("rays and actor poses are inconsistent")
    rotation, translation = actor[:, :3, :3], actor[:, :3, 3]
    local_o = np.einsum("ni,nij->nj", world_o - translation, rotation)
    local_d = np.einsum("ni,nij->nj", world_d, rotation)
    local_d /= np.maximum(np.linalg.norm(local_d, axis=1, keepdims=True), 1e-8)
    half = size[None, :] * .5
    parallel = np.abs(local_d) <= 1e-7
    outside_parallel = parallel & (np.abs(local_o) > half)
    inverse = np.where(parallel, 1.0, 1.0 / np.where(parallel, 1.0, local_d))
    lower, upper = (-half - local_o) * inverse, (half - local_o) * inverse
    near, far = np.minimum(lower, upper), np.maximum(lower, upper)
    near, far = np.where(parallel, -np.inf, near), np.where(parallel, np.inf, far)
    entry, exit_ = np.max(near, axis=1), np.min(far, axis=1)
    entry = np.maximum(entry, .01)
    valid = ~outside_parallel.any(axis=1) & np.isfinite(entry) & np.isfinite(exit_) & (exit_ > entry + 1e-5)
    return local_o, local_d, entry, np.where(valid, exit_, entry)


def ray_support_from_bev(origins: np.ndarray, directions: np.ndarray, actor_poses: np.ndarray,
                         dimensions: np.ndarray, occupied_voxels: np.ndarray, *, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-march only within the track cuboid and return supported/box/depth maps.

    The actor pose may differ per ray, preserving rolling-shutter actor motion.
    A ray is supported only if it reaches a raw-LiDAR-backed local voxel.  The
    cuboid hit is reported separately so the audit can quantify how much a BEV
    grid contracts an over-broad track-box silhouette.
    """
    local_o, local_d, entry, exit_ = _local_ray_segments(origins, directions, actor_poses, dimensions)
    box = exit_ > entry + 1e-5
    supported = np.zeros(len(local_o), dtype=bool)
    depth = np.full(len(local_o), np.nan, np.float32)
    hashes = np.unique(_voxel_hash(occupied_voxels))
    lengths = np.maximum(exit_ - entry, 0.)
    steps = np.ceil(lengths / max(voxel_size_m * .65, 1e-4)).astype(np.int64)
    maximum = int(steps.max(initial=0))
    for index in range(maximum):
        active = box & ~supported & (index < steps)
        if not active.any():
            continue
        distance = entry[active] + np.minimum((index + .5) * voxel_size_m * .65, lengths[active] - 1e-5)
        points = local_o[active] + local_d[active] * distance[:, None]
        keys = np.floor(points / voxel_size_m).astype(np.int64)
        hit = np.isin(_voxel_hash(keys), hashes, assume_unique=False)
        selected = np.flatnonzero(active)[hit]
        supported[selected] = True
        depth[selected] = distance[hit]
    return supported, box, depth


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray, region: np.ndarray) -> dict[str, float | int]:
    """Report recall, precision and ring leakage without treating unknown pixels as truth."""
    pred, truth, valid = (np.asarray(value, bool) for value in (prediction, target, region))
    if pred.shape != truth.shape or valid.shape != pred.shape:
        raise ValueError("prediction, target and region shapes must match")
    p, t = pred & valid, truth & valid
    intersection = int((p & t).sum()); union = int((p | t).sum())
    target_pixels, prediction_pixels = int(t.sum()), int(p.sum())
    ring = valid & ~truth
    return {
        "target_pixels": target_pixels, "prediction_pixels": prediction_pixels,
        "iou": float(intersection / union) if union else 0.,
        "recall": float(intersection / target_pixels) if target_pixels else 0.,
        "precision": float(intersection / prediction_pixels) if prediction_pixels else 0.,
        "area_ratio": float(prediction_pixels / target_pixels) if target_pixels else 0.,
        "ring_leakage": float((p & ring).sum() / ring.sum()) if ring.any() else 0.,
    }


def _record_instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((value for value in record["instances"] if int(value["track_id"]) == track_id and value.get("mask_origin") == "source"), None)


def _actor_poses_at_timestamps(track: dict[str, Any], timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(timestamps, np.int64)
    unique, inverse = np.unique(values, return_inverse=True)
    poses = []
    valid = []
    for timestamp in unique:
        pose = _interpolate_track_pose(track, int(timestamp))
        valid.append(pose is not None)
        poses.append(np.eye(4, dtype=np.float32) if pose is None else np.asarray(pose, np.float32))
    return np.asarray(poses, np.float32)[inverse], np.asarray(valid, bool)[inverse]


def audit_bev_occupancy_gate(options: BevOccupancyGateOptions) -> Path:
    """Evaluate train-LiDAR BEV support on strictly held-out source masks."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite BEV audit: {options.output}")
    manifest_path = options.trusted_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted_manifest must be complete trusted supervision")
    track = next((value for value in manifest["tracks"] if int(value["track_id"]) == options.track_id), None)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("track_id must name a rigid trusted actor")
    asset_path = options.actor_asset.resolve(); asset = CanonicalActorAsset.load(asset_path)
    if options.track_id not in asset.actor_ids:
        raise ValueError("actor asset does not contain the requested track")
    index = asset.actor_ids.index(options.track_id)
    positions = asset.positions[asset.actor_indices == index]
    if not len(positions):
        raise ValueError("actor asset contains no points for requested track")
    occupied = dilated_bev_voxels(positions, voxel_size_m=options.voxel_size_m,
                                  dilation_xy=options.dilation_cells_xy, dilation_z=options.dilation_cells_z)

    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    models: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if not options.reference_start <= reference < options.reference_end or reference % 2 != options.reference_parity:
            continue
        for record in frame["cameras"]:
            camera_id = str(record["camera_id"])
            if camera_id not in options.camera_ids:
                continue
            instance = _record_instance(record, options.track_id)
            if instance is None:
                continue
            sensor = loader.get_camera_sensor(camera_id)
            image = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), np.uint8)
            mask = np.asarray(Image.open(roots[str(instance["mask_root"])] / str(instance["mask"])).convert("L"), np.uint8) > 0
            if mask.shape != image.shape[:2]:
                raise ValueError(f"source mask/image shape mismatch: {camera_id} ref={reference}")
            yy, xx = np.where(mask)
            if not len(xx):
                continue
            x0, x1 = max(int(xx.min()) - options.roi_margin_pixels, 0), min(int(xx.max()) + options.roi_margin_pixels + 1, mask.shape[1])
            y0, y1 = max(int(yy.min()) - options.roi_margin_pixels, 0), min(int(yy.max()) + options.roi_margin_pixels + 1, mask.shape[0])
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            pixels = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).astype(np.float32)
            model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            returned = model.image_points_to_world_rays_shutter_pose(
                pixels, np.asarray(record["camera_to_world_start"], np.float64), np.asarray(record["camera_to_world_end"], np.float64),
                int(record["timestamp_start_us"]), int(record["timestamp_end_us"]), return_timestamps=True,
            )
            rays = returned.world_rays.detach().float().cpu().numpy()
            timestamps = returned.timestamps_us.detach().cpu().numpy().astype(np.int64)
            poses, pose_valid = _actor_poses_at_timestamps(track, timestamps)
            finite = np.isfinite(rays).all(axis=1) & pose_valid
            support, cuboid, depth = ray_support_from_bev(rays[:, :3], rays[:, 3:], poses,
                                                           np.asarray(track["length_width_height"], np.float32), occupied,
                                                           voxel_size_m=options.voxel_size_m)
            support &= finite; cuboid &= finite
            region = np.ones((y1 - y0, x1 - x0), bool)
            target = mask[y0:y1, x0:x1]
            reports.append({
                "reference_frame_index": reference, "camera_id": camera_id,
                "source_frame_index": int(record["source_frame_index"]), "roi_xyxy": [x0, y0, x1, y1],
                "bev": binary_mask_metrics(support.reshape(target.shape), target, region),
                "cuboid": binary_mask_metrics(cuboid.reshape(target.shape), target, region),
                "native_ray_valid_fraction": float(finite.mean()), "bev_depth_median": float(np.nanmedian(depth)) if np.isfinite(depth).any() else None,
            })
            print(f"BEV occupancy ref={reference:03d} camera={camera_id}: "
                  f"recall={reports[-1]['bev']['recall']:.3f} iou={reports[-1]['bev']['iou']:.3f}", flush=True)
    if not reports:
        raise RuntimeError("no held-out source-mask observation matches the requested BEV audit")

    def mean(path: tuple[str, str]) -> float:
        return float(np.mean([float(report[path[0]][path[1]]) for report in reports]))
    output = {
        "schema": SCHEMA, "version": VERSION, "trusted_manifest": str(manifest_path), "actor_asset": str(asset_path),
        "track_id": options.track_id, "reference_range": [options.reference_start, options.reference_end],
        "reference_parity": options.reference_parity, "camera_ids": list(options.camera_ids),
        "occupancy": {"coordinate_frame": "actor_local", "source": "raw-LiDAR multi-view canonical actor points only",
                      "voxel_size_m": options.voxel_size_m, "dilation_cells_xy": options.dilation_cells_xy,
                      "dilation_cells_z": options.dilation_cells_z, "raw_point_count": int(len(positions)),
                      "occupied_voxel_count": int(len(occupied))},
        "observations": reports,
        "mean": {"bev_iou": mean(("bev", "iou")), "bev_recall": mean(("bev", "recall")),
                 "bev_precision": mean(("bev", "precision")), "bev_area_ratio": mean(("bev", "area_ratio")),
                 "bev_ring_leakage": mean(("bev", "ring_leakage")), "cuboid_iou": mean(("cuboid", "iou")),
                 "cuboid_area_ratio": mean(("cuboid", "area_ratio"))},
        "limitations": [
            "This is a held-out source-view occupancy gate, not a novel-view RGB renderer.",
            "Only raw-LiDAR-backed canonical voxels are occupied; unsupported surfaces remain unknown/transparent.",
            "A positive result is required before a BEV gate can affect a dynamic actor renderer or target 7V output.",
        ],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote LiDAR-BEV occupancy gate audit: {options.output}", flush=True)
    return options.output
