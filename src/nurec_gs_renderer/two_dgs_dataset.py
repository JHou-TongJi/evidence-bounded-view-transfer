"""Build a bounded COLMAP-style 2DGS proxy dataset from an NCore chunk.

The official 2D Gaussian Splatting implementation only consumes undistorted,
global-shutter COLMAP pinhole cameras.  NCore camera images must therefore be
resampled with the native FTheta inverse model into narrow virtual pinhole
views.  This is an interoperability baseline, *not* the final renderer: its
pose is evaluated at exposure midpoint and its output must be checked again
with the repository's native FTheta/rolling-shutter renderer before use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .camera import compose_ncore_camera_pose
from .ply_io import load_gaussian_ply
from .pose_sources import _create_ncore_loader, _ncore_exposure_timestamps
from .street_dataset import (
    REAR_PROXY_CAMERA_IDS,
    STREET_CAMERA_IDS,
    _chunk_reference_indices,
    _nearest_index,
    _rectify_frame,
    pinhole_intrinsics,
    tile_to_source_pose,
    virtual_pinhole_rays_in_source,
)


@dataclass(frozen=True)
class TwoDgsDatasetOptions:
    ncore_path: Path
    static_ply: Path
    output: Path
    chunk_index: int = 0
    camera_ids: tuple[str, ...] = STREET_CAMERA_IDS
    width: int = 480
    height: int = 270
    horizontal_fov_deg: float = 55.0
    tile_yaws_deg: tuple[float, ...] = (0.0,)
    rear_tile_yaws_deg: tuple[float, ...] = (0.0,)
    near_field_camera_ids: tuple[str, ...] = ()
    near_field_tile_yaws_deg: tuple[float, ...] = ()
    near_field_tile_pitch_deg: float = -12.0
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    holdout_stride: int = 8
    max_initial_points: int = 100_000
    rectify_device: str = "cuda"
    dynamic_mask_manifest: Path | None = None
    filter_dynamic_initialization: bool = False
    dynamic_initialization_min_observations: int = 2
    road_lidar_seed: Path | None = None
    replace_road_initialization: bool = False
    road_depth_supervision: bool = False
    road_depth_stride: int = 5
    road_depth_holdout_every: int = 5
    road_depth_holdout_offset: int = 4
    road_depth_min_range_m: float = 3.0
    road_depth_max_range_m: float = 60.0
    road_depth_ground_local_z_min_m: float = -3.2
    road_depth_ground_local_z_max_m: float = -1.4
    road_depth_lidar_window: int = 2
    road_depth_max_lidar_delta_us: int = 250_000
    road_depth_lidar_decay_us: int = 100_000
    road_depth_ego_exclusion_radius_m: float = 2.5

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("chunk index and resolution must be positive")
        if not self.camera_ids or any(camera not in STREET_CAMERA_IDS for camera in self.camera_ids):
            raise ValueError(f"camera_ids must be a non-empty subset of {STREET_CAMERA_IDS}")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        if not 20.0 <= self.horizontal_fov_deg < 175.0:
            raise ValueError("horizontal_fov_deg must be in [20, 175)")
        if not self.tile_yaws_deg or not self.rear_tile_yaws_deg:
            raise ValueError("tile yaw lists must be non-empty")
        if not all(np.isfinite(yaw) and abs(yaw) < 90.0 for yaw in (*self.tile_yaws_deg, *self.rear_tile_yaws_deg)):
            raise ValueError("tile yaws must lie in (-90, 90)")
        if any(camera not in STREET_CAMERA_IDS for camera in self.near_field_camera_ids):
            raise ValueError(f"near_field_camera_ids must be a subset of {STREET_CAMERA_IDS}")
        if len(set(self.near_field_camera_ids)) != len(self.near_field_camera_ids):
            raise ValueError("near_field_camera_ids must be unique")
        if bool(self.near_field_camera_ids) != bool(self.near_field_tile_yaws_deg):
            raise ValueError("near-field cameras and yaws must be supplied together")
        if not all(np.isfinite(yaw) and abs(yaw) < 90.0 for yaw in self.near_field_tile_yaws_deg):
            raise ValueError("near-field tile yaws must lie in (-90, 90)")
        if not np.isfinite(self.near_field_tile_pitch_deg) or not -45.0 < self.near_field_tile_pitch_deg < 45.0:
            raise ValueError("near-field tile pitch must lie in (-45, 45)")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.stride <= 0 or self.holdout_stride < 2 or self.max_initial_points <= 0:
            raise ValueError("stride, holdout_stride and max_initial_points must be positive")
        if self.road_lidar_seed is None and self.replace_road_initialization:
            raise ValueError("replace_road_initialization requires road_lidar_seed")
        if self.filter_dynamic_initialization and self.dynamic_mask_manifest is None:
            raise ValueError("filter_dynamic_initialization requires dynamic_mask_manifest")
        if self.dynamic_initialization_min_observations <= 0:
            raise ValueError("dynamic_initialization_min_observations must be positive")
        if self.road_depth_stride <= 0 or self.road_depth_holdout_every < 2:
            raise ValueError("road depth stride must be positive and holdout period at least two")
        if not 0 <= self.road_depth_holdout_offset < self.road_depth_holdout_every:
            raise ValueError("road depth holdout offset must be within its period")
        if self.road_depth_min_range_m <= 0.0 or self.road_depth_max_range_m <= self.road_depth_min_range_m:
            raise ValueError("road depth range limits are invalid")
        if self.road_depth_ground_local_z_max_m <= self.road_depth_ground_local_z_min_m:
            raise ValueError("road depth local-z band is invalid")
        if self.road_depth_lidar_window < 0 or self.road_depth_max_lidar_delta_us <= 0 or self.road_depth_lidar_decay_us <= 0:
            raise ValueError("road depth LiDAR window and time settings are invalid")
        if self.road_depth_ego_exclusion_radius_m < 0.0:
            raise ValueError("road depth ego exclusion radius must be non-negative")


def _dynamic_masks_by_source_observation(manifest_path: Path) -> dict[tuple[str, int, int], list[Path]]:
    """Index only accepted source-priority instance masks by physical observation.

    These masks are deliberately used only to *remove* pixels from static
    supervision.  They never create RGB supervision for a dynamic actor.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ncore-dynamic-reconstruction-mask-refinement":
        raise ValueError(f"{manifest_path} is not a dynamic-mask-refinement manifest")
    indexed: dict[tuple[str, int, int], list[Path]] = {}
    for camera in payload.get("cameras", []):
        camera_id = str(camera.get("camera_id", ""))
        for observation in camera.get("observations", []):
            relative = observation.get("refined_mask")
            if relative is None or observation.get("mask_origin") not in {"source", "sam2"}:
                continue
            if not str(observation.get("status", "")).startswith("accepted"):
                continue
            key = (camera_id, int(observation["reference_frame_index"]), int(observation["source_frame_index"]))
            indexed.setdefault(key, []).append(manifest_path.parent / str(relative))
    if not indexed:
        raise ValueError(f"{manifest_path} contains no accepted dynamic masks")
    return indexed


def _rectify_binary_mask(
    sensor: object,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    device: str,
    yaw_deg: float,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Resample a raw FTheta binary mask into the same virtual pinhole tile."""
    import torch
    import torch.nn.functional as functional
    from ncore.impl.sensors.camera import FThetaCameraModel

    # PIL-backed masks can be read-only views.  grid_sample receives a Torch
    # tensor and must not inherit that undefined writable state.
    source = np.array(mask, dtype=np.uint8, copy=True)
    if source.ndim != 2:
        raise ValueError(f"dynamic mask must be HxW, got {source.shape}")
    model = FThetaCameraModel(sensor.model_parameters, device=device)
    rays_np = virtual_pinhole_rays_in_source(width, height, intrinsic, yaw_deg, pitch_deg).reshape(-1, 3)
    rays = torch.from_numpy(rays_np).to(device)
    returned = model.camera_rays_to_pixels(rays)
    pixels = returned.pixels.reshape(height, width, 2).to(dtype=torch.float32)
    valid = returned.valid_flag.reshape(height, width)
    source_h, source_w = source.shape
    grid = pixels.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(source_w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(source_h - 1, 1) - 1.0
    tensor = torch.from_numpy(source).to(device=device, dtype=torch.float32)[None, None]
    sampled = functional.grid_sample(tensor, grid[None], mode="nearest", padding_mode="zeros", align_corners=True)[0, 0]
    return ((sampled > 0.5) & valid & torch.isfinite(grid).all(dim=-1)).byte().cpu().numpy()


def _yaw_token(yaw_deg: float) -> str:
    return ("p" if yaw_deg >= 0 else "m") + f"{abs(yaw_deg):.1f}".rstrip("0").rstrip(".").replace(".", "d")


def _pitch_suffix(pitch_deg: float) -> str:
    """Keep legacy yaw-only proxy filenames unchanged."""
    return "" if abs(float(pitch_deg)) < 1e-7 else f"_pitch_{_yaw_token(pitch_deg)}"


def _rotation_to_colmap_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP ``qw,qx,qy,qz`` from an OpenCV world-to-camera rotation."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.asarray((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                        (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            q = np.asarray(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                            (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale))
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            q = np.asarray(((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                            0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            q = np.asarray(((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                            (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def _load_road_lidar_seed(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"xyz", "rgb", "support", "metadata"}.issubset(data.files):
                raise ValueError("road LiDAR seed is missing xyz/rgb/support/metadata")
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            rgb = np.asarray(data["rgb"], dtype=np.float32)
            metadata = json.loads(str(data["metadata"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load road LiDAR seed {path}: {exc}") from exc
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or len(xyz) == 0:
        raise ValueError("road LiDAR seed xyz/rgb must be non-empty Nx3 arrays")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("road LiDAR seed contains invalid coordinates or RGB")
    if metadata.get("schema") != "ncore-2dgs-road-lidar-seed" or int(metadata.get("version", -1)) != 1:
        raise ValueError("road LiDAR seed schema/version is unsupported")
    return xyz, rgb, metadata


def _rasterize_road_depth_target(
    world: np.ndarray,
    normal_world: np.ndarray,
    point_confidence: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    static_mask: np.ndarray,
    *,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffer static road LiDAR returns into one virtual pinhole camera.

    The returned depth is camera ``+z`` in metres, matching 2DGS's
    ``surf_depth`` convention.  A raw return is only a sparse anchor: pixels
    without a nearest valid return have zero weight and never enter the loss.
    """
    points = np.asarray(world, dtype=np.float64)
    normal = np.asarray(normal_world, dtype=np.float64).reshape(3)
    point_weights = np.asarray(point_confidence, dtype=np.float64).reshape(-1)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    K = np.asarray(intrinsic, dtype=np.float64)
    mask = np.asarray(static_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3 or len(point_weights) != len(points):
        raise ValueError("world road points must be Nx3")
    if pose.shape != (4, 4) or K.shape != (3, 3) or mask.ndim != 2:
        raise ValueError("invalid camera/intrinsic/static-mask shapes for road depth target")
    if (not np.isfinite(points).all() or not np.isfinite(normal).all() or not np.isfinite(point_weights).all()
            or not np.isfinite(pose).all() or not np.isfinite(K).all() or np.any(point_weights < 0.0)):
        raise ValueError("road depth target inputs must be finite")
    h, w = mask.shape
    depth = np.zeros((h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    if len(points) == 0:
        return depth, confidence, normal_map
    world_to_camera = np.linalg.inv(pose)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    uvw = camera @ K.T
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.rint(uvw[:, 0] / uvw[:, 2]).astype(np.int64)
        v = np.rint(uvw[:, 1] / uvw[:, 2]).astype(np.int64)
    usable = (
        np.isfinite(camera).all(axis=1)
        & (z > 0.10)
        & (z <= float(max_depth_m))
        & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    )
    if not np.any(usable):
        return depth, confidence, normal_map
    u, v, z, point_weights = u[usable], v[usable], z[usable], point_weights[usable]
    flat = v * w + u
    # ``minimum.at`` gives deterministic nearest-surface ownership when one
    # LiDAR scan has multiple returns in the same proxy pixel.
    z_buffer = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat, z)
    # Retain temporally adjacent returns only when they support the same
    # nearest road surface; a farther surface is not averaged into camera-z.
    supported = (z <= z_buffer[flat] + (0.05 + 0.02 * z_buffer[flat])) & mask[v, u]
    if not np.any(supported):
        return depth, confidence, normal_map
    flat, point_weights = flat[supported], point_weights[supported]
    support_weight = np.bincount(flat, weights=point_weights, minlength=h * w).astype(np.float32)
    accepted = mask.reshape(-1) & np.isfinite(z_buffer)
    depth.reshape(-1)[accepted] = z_buffer[accepted].astype(np.float32)
    confidence.reshape(-1)[accepted] = np.clip(support_weight[accepted] / 3.0, 0.0, 1.0)
    unit_normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    normal_map.reshape(-1, 3)[accepted] = unit_normal.astype(np.float32)
    return depth, confidence, normal_map


def _road_lidar_world_points(
    lidar: object,
    timestamp_us: int,
    *,
    min_range_m: float,
    max_range_m: float,
    ground_local_z_min_m: float,
    ground_local_z_max_m: float,
    ego_exclusion_radius_m: float,
    lidar_index: int | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return one scan's safe ground returns and a robust world normal."""
    if lidar_index is None:
        lidar_index = int(lidar.get_closest_frame_index(int(timestamp_us)))
    lidar_timestamp = int(lidar.get_frame_timestamp_us(lidar_index))
    local = np.asarray(lidar.get_frame_point_cloud(lidar_index, False, False).xyz_m_end, dtype=np.float64)
    ranges = np.linalg.norm(local, axis=1)
    keep = (
        np.isfinite(local).all(axis=1)
        & (ranges >= max(float(min_range_m), float(ego_exclusion_radius_m))) & (ranges <= float(max_range_m))
        & (local[:, 2] >= float(ground_local_z_min_m))
        & (local[:, 2] <= float(ground_local_z_max_m))
    )
    ground = local[keep]
    T_lidar_world = np.asarray(lidar.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
    world = ground @ T_lidar_world[:3, :3].T + T_lidar_world[:3, 3]
    if len(ground) < 3:
        return lidar_index, lidar_timestamp, world.astype(np.float32), np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    # A global scan-level PCA normal is intentionally conservative: it gives
    # the road supervision a planar tendency without inventing curb geometry.
    sample = ground[::max(1, len(ground) // 20_000)]
    centred = sample - np.median(sample, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    local_normal = vectors[-1]
    normal_world = T_lidar_world[:3, :3] @ local_normal
    normal_world /= max(float(np.linalg.norm(normal_world)), 1e-12)
    return lidar_index, lidar_timestamp, world.astype(np.float32), normal_world.astype(np.float32)


def _road_lidar_multiscan_world_points(
    lidar: object,
    timestamp_us: int,
    *,
    min_range_m: float,
    max_range_m: float,
    ground_local_z_min_m: float,
    ground_local_z_max_m: float,
    ego_exclusion_radius_m: float,
    scan_window: int,
    max_delta_us: int,
    decay_us: int,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Fuse a bounded LiDAR time window; temporal distance lowers confidence."""
    centre = int(lidar.get_closest_frame_index(int(timestamp_us)))
    indices: list[int] = []
    worlds: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    closest_normal: np.ndarray | None = None
    fallback_normal: np.ndarray | None = None
    for candidate in range(max(0, centre - scan_window), min(int(lidar.frames_count), centre + scan_window + 1)):
        index, lidar_time, world, normal = _road_lidar_world_points(
            lidar, timestamp_us, min_range_m=min_range_m, max_range_m=max_range_m,
            ground_local_z_min_m=ground_local_z_min_m, ground_local_z_max_m=ground_local_z_max_m,
            ego_exclusion_radius_m=ego_exclusion_radius_m, lidar_index=candidate,
        )
        delta = abs(lidar_time - int(timestamp_us))
        if delta > max_delta_us or len(world) == 0:
            continue
        indices.append(index)
        worlds.append(world)
        weights.append(np.full(len(world), np.exp(-float(delta) / float(decay_us)), dtype=np.float32))
        if fallback_normal is None:
            fallback_normal = normal
        if index == centre:
            closest_normal = normal
    if not worlds:
        return [], np.empty((0, 3), np.float32), np.empty((0,), np.float32), np.asarray((0.0, 0.0, 1.0), np.float32)
    if closest_normal is None:
        assert fallback_normal is not None
        closest_normal = fallback_normal
    return indices, np.concatenate(worlds, axis=0), np.concatenate(weights, axis=0), closest_normal


def _initial_cloud_candidates(
    static_ply: Path,
    max_points: int,
    *,
    replace_road_initialization: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return a deterministic, bounded static 3DGS initial cloud.

    This intentionally filters only the *initializer*.  Dynamic instance masks
    remain exclusion masks for RGB supervision, and never become appearance or
    geometry supervision for 2DGS.
    """
    from plyfile import PlyData

    scene = load_gaussian_ply(static_ply)
    keep = scene.opacities >= 0.20
    removed_road = 0
    if replace_road_initialization:
        source = PlyData.read(static_ply)["vertex"].data
        if "road_mask" not in source.dtype.names or len(source) != scene.count:
            raise ValueError("replace_road_initialization requires source PLY road_mask aligned with Gaussians")
        road = np.asarray(source["road_mask"], dtype=np.uint8) > 0
        removed_road = int(np.count_nonzero(keep & road))
        keep &= ~road
    positions = scene.means[keep]
    # InstantNuRec and Graphdeco use RGB = SH_C0 * DC + 0.5 for degree-zero SH.
    rgb = np.clip(scene.sh_coeffs[keep, 0] * 0.28209479177387814 + 0.5, 0.0, 1.0)
    if len(positions) > max_points:
        # Deterministic stratified subsampling keeps the initial cloud bounded.
        chosen = np.linspace(0, len(positions) - 1, max_points, dtype=np.int64)
        positions, rgb = positions[chosen], rgb[chosen]
    return positions, rgb, {
        "source_gaussians": int(scene.count), "opacity_kept_before_dynamic_filter": int(keep.sum()),
        "removed_road_gaussians": removed_road, "candidate_points_before_dynamic_filter": int(len(positions)),
    }


def _dynamic_initialization_mask_hits(
    positions_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    dynamic_mask: np.ndarray,
) -> np.ndarray:
    """Return candidate points whose pinhole projection falls in a dynamic mask.

    The caller counts hits across different physical source observations before
    removing a point.  A single mask is intentionally insufficient by default:
    it can simply be a moving foreground object temporarily occluding a valid
    static surface.  This is a conservative initializer-cleaning mechanism,
    not a dynamic reconstruction method.
    """
    points = np.asarray(positions_world, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    K = np.asarray(intrinsic, dtype=np.float64)
    mask = np.asarray(dynamic_mask, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or pose.shape != (4, 4) or K.shape != (3, 3) or mask.ndim != 2:
        raise ValueError("invalid point/camera/mask shapes for dynamic initialization filter")
    world_to_camera = np.linalg.inv(pose)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-4)
    pixels = np.zeros((len(points), 2), dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels[:, 0] = np.rint(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(np.int64, copy=False)
        pixels[:, 1] = np.rint(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(np.int64, copy=False)
    height, width = mask.shape
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    hits = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(valid)
    hits[indices] = mask[pixels[indices, 1], pixels[indices, 0]] != 0
    return hits


def _write_initial_ply(
    path: Path,
    positions: np.ndarray,
    rgb: np.ndarray,
    initialisation: dict[str, object],
    *,
    road_lidar_seed: Path | None,
) -> dict[str, object]:
    """Write a simple xyz/RGB COLMAP initial cloud after conservative filtering."""
    from plyfile import PlyData, PlyElement

    positions = np.asarray(positions, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or rgb.shape != positions.shape or len(positions) == 0:
        raise ValueError("initial positions/RGB must be non-empty Nx3 arrays")
    road_seed_count = 0
    road_metadata: dict[str, object] | None = None
    if road_lidar_seed is not None:
        road_positions, road_rgb, road_metadata = _load_road_lidar_seed(road_lidar_seed)
        road_seed_count = int(len(road_positions))
        positions = np.concatenate((positions, road_positions), axis=0)
        rgb = np.concatenate((rgb, road_rgb), axis=0)
    vertices = np.empty(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                               ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["red"], vertices["green"], vertices["blue"] = (rgb * 255.0 + 0.5).astype(np.uint8).T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    return {
        **initialisation,
        "opacity_kept": int(len(positions) - road_seed_count),
        "road_lidar_seed": None if road_lidar_seed is None else str(road_lidar_seed.resolve()),
        "road_lidar_seed_points": road_seed_count, "road_lidar_metadata": road_metadata,
        "written_points": int(len(vertices)),
    }


def build_two_dgs_dataset(options: TwoDgsDatasetOptions) -> Path:
    """Write COLMAP text metadata, rectified images and an initial 2DGS PLY."""
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset directory: {output}")
    images_dir, sparse_dir = output / "images", output / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    road_depth_dir = output / "road_depth" if options.road_depth_supervision else None
    if road_depth_dir is not None:
        road_depth_dir.mkdir()

    loader = _create_ncore_loader(options.ncore_path)
    dynamic_masks = (
        None if options.dynamic_mask_manifest is None
        else _dynamic_masks_by_source_observation(options.dynamic_mask_manifest.resolve())
    )
    initial_positions, initial_rgb, initialisation = _initial_cloud_candidates(
        options.static_ply, options.max_initial_points,
        replace_road_initialization=options.replace_road_initialization,
    )
    dynamic_initialization_hits = np.zeros(len(initial_positions), dtype=np.uint16)
    dynamic_initialization_observations = 0
    sensors = {camera: loader.get_camera_sensor(camera) for camera in STREET_CAMERA_IDS}
    lidar = loader.get_lidar_sensor("lidar_top_360fov") if options.road_depth_supervision else None
    selection_options = type("Selection", (), {"chunk_index": options.chunk_index, "frame_start": options.frame_start,
                                                  "frame_end": options.frame_end, "stride": options.stride})()
    selected = _chunk_reference_indices(loader, sensors, selection_options)
    reference_sensor = sensors[STREET_CAMERA_IDS[0]]
    reference_end = _ncore_exposure_timestamps(reference_sensor)[:, 1].astype(np.int64)
    intrinsic = pinhole_intrinsics(options.width, options.height, options.horizontal_fov_deg)
    camera_specs = [
        (camera, float(yaw), 0.0)
        for camera in options.camera_ids
        for yaw in (options.rear_tile_yaws_deg if camera in REAR_PROXY_CAMERA_IDS else options.tile_yaws_deg)
    ]
    camera_specs.extend(
        (camera, float(yaw), float(options.near_field_tile_pitch_deg))
        for camera in options.near_field_camera_ids
        for yaw in options.near_field_tile_yaws_deg
    )
    if len({(camera, yaw, pitch) for camera, yaw, pitch in camera_specs}) != len(camera_specs):
        raise ValueError("base and near-field proxy tiles must be distinct")

    camera_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for camera_id, (_, yaw, pitch) in enumerate(camera_specs, start=1):
        camera_lines.append(f"{camera_id} PINHOLE {options.width} {options.height} {intrinsic[0,0]:.9f} {intrinsic[1,1]:.9f} {intrinsic[0,2]:.9f} {intrinsic[1,2]:.9f}")
    image_lines = ["# Image list with two lines of data per image:", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    records: list[dict[str, object]] = []
    image_id = 1
    road_depth_summary = {"train_images": 0, "holdout_images": 0, "valid_pixels": 0, "lidar_frames": set()}
    print(f"Preparing 2DGS proxy: {len(selected)} reference frames × {len(camera_specs)} pinhole tiles at {options.width}×{options.height}", flush=True)
    for output_frame, reference_index in enumerate(selected):
        timestamp = int(reference_end[reference_index])
        road_world: np.ndarray | None = None
        road_normal_world: np.ndarray | None = None
        road_lidar_indices: list[int] | None = None
        road_point_confidence: np.ndarray | None = None
        depth_eligible = options.road_depth_supervision and output_frame % options.road_depth_stride == 0
        if depth_eligible:
            assert lidar is not None
            road_lidar_indices, road_world, road_point_confidence, road_normal_world = _road_lidar_multiscan_world_points(
                lidar, timestamp,
                min_range_m=options.road_depth_min_range_m,
                max_range_m=options.road_depth_max_range_m,
                ground_local_z_min_m=options.road_depth_ground_local_z_min_m,
                ground_local_z_max_m=options.road_depth_ground_local_z_max_m,
                ego_exclusion_radius_m=options.road_depth_ego_exclusion_radius_m,
                scan_window=options.road_depth_lidar_window,
                max_delta_us=options.road_depth_max_lidar_delta_us,
                decay_us=options.road_depth_lidar_decay_us,
            )
        for camera_index, (source_camera, yaw_deg, pitch_deg) in enumerate(camera_specs):
            sensor = sensors[source_camera]
            exposures = _ncore_exposure_timestamps(sensor)
            source_index = _nearest_index(exposures[:, 1].astype(np.int64), timestamp)
            raw = np.asarray(sensor.get_frame_image_array(source_index), dtype=np.uint8)
            rectified, valid = _rectify_frame(
                sensor,
                raw,
                intrinsic,
                width=options.width,
                height=options.height,
                device=options.rectify_device,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
            )
            # Do not silently treat invalid fisheye areas as RGB supervision.
            rectified[valid == 0] = 0
            filename = f"{output_frame:04d}_{camera_index:02d}_{source_camera}_{_yaw_token(yaw_deg)}{_pitch_suffix(pitch_deg)}.png"
            dynamic_mask = np.zeros((options.height, options.width), dtype=np.uint8)
            if dynamic_masks is not None:
                raw_masks = dynamic_masks.get((source_camera, int(reference_index), int(source_index)), ())
                if raw_masks:
                    raw_union: np.ndarray | None = None
                    for mask_path in raw_masks:
                        if not mask_path.is_file():
                            raise FileNotFoundError(mask_path)
                        # PIL keeps the source descriptor open until its image
                        # object is closed.  A full clip has thousands of
                        # observations, so leaving this implicit exhausts the
                        # process file-descriptor limit before dataset output
                        # is complete.
                        with Image.open(mask_path) as opened_mask:
                            raw = np.asarray(opened_mask.convert("L"), dtype=np.uint8)
                        raw_union = raw if raw_union is None else np.maximum(raw_union, raw)
                    dynamic_mask = _rectify_binary_mask(
                        sensor,
                        raw_union,
                        intrinsic,
                        width=options.width,
                        height=options.height,
                        device=options.rectify_device,
                        yaw_deg=yaw_deg,
                        pitch_deg=pitch_deg,
                    )
            static_mask = ((valid != 0) & (dynamic_mask == 0)).astype(np.uint8)
            if dynamic_masks is None:
                Image.fromarray(rectified, mode="RGB").save(images_dir / filename)
            else:
                rgba = np.concatenate((rectified, (static_mask[..., None] * 255).astype(np.uint8)), axis=-1)
                Image.fromarray(rgba, mode="RGBA").save(images_dir / filename)
            midpoint = int((int(exposures[source_index, 0]) + int(exposures[source_index, 1])) // 2)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig)) @ tile_to_source_pose(yaw_deg, pitch_deg)
            # Count at most one projection for each physical source camera and
            # reference frame.  Overlapping virtual tiles are only different
            # views of the same raw camera image and are not independent
            # evidence that a Gaussian belongs to a dynamic instance.
            if (options.filter_dynamic_initialization and np.any(dynamic_mask)
                    and abs(yaw_deg) < 1e-7 and abs(pitch_deg) < 1e-7):
                dynamic_initialization_hits += _dynamic_initialization_mask_hits(
                    initial_positions, camera_to_world, intrinsic, dynamic_mask,
                ).astype(np.uint16)
                dynamic_initialization_observations += 1
            world_to_camera = np.linalg.inv(camera_to_world)
            qvec = _rotation_to_colmap_quaternion(world_to_camera[:3, :3])
            camera_id = camera_index + 1
            image_lines.append("%d %s %s %s %s %s %s %s %d %s" % (
                image_id, *[f"{value:.12g}" for value in qvec], *[f"{value:.12g}" for value in world_to_camera[:3, 3]], camera_id, filename))
            image_lines.append("")
            road_depth_record: dict[str, object] | None = None
            if depth_eligible:
                assert (road_depth_dir is not None and road_world is not None and road_normal_world is not None
                        and road_lidar_indices is not None and road_point_confidence is not None)
                depth, confidence, normal_world = _rasterize_road_depth_target(
                    road_world, road_normal_world, road_point_confidence, camera_to_world, intrinsic, static_mask,
                    max_depth_m=options.road_depth_max_range_m,
                )
                depth_slot = output_frame // options.road_depth_stride
                split = "holdout" if depth_slot % options.road_depth_holdout_every == options.road_depth_holdout_offset else "train"
                depth_filename = f"{Path(filename).stem}.npz"
                np.savez_compressed(
                    road_depth_dir / depth_filename,
                    depth=np.asarray(depth, np.float32), confidence=np.asarray(confidence, np.float32),
                    normal_world=np.asarray(normal_world, np.float32),
                    split=np.asarray(split), lidar_frame_indices=np.asarray(road_lidar_indices, np.int32),
                )
                valid_pixels = int(np.count_nonzero(confidence > 0.0))
                road_depth_summary[f"{split}_images"] += 1
                road_depth_summary["valid_pixels"] += valid_pixels
                road_depth_summary["lidar_frames"].update(road_lidar_indices)
                road_depth_record = {"file": f"road_depth/{depth_filename}", "split": split,
                                     "lidar_frame_indices": road_lidar_indices, "valid_pixels": valid_pixels}
            records.append({"image_id": image_id, "reference_frame": int(reference_index), "source_frame": int(source_index),
                            "source_camera": source_camera, "tile_yaw_deg": yaw_deg, "tile_pitch_deg": pitch_deg, "timestamp_us": midpoint,
                            "image": f"images/{filename}", "camera_to_world": camera_to_world.tolist(),
                            "valid_fraction": float(valid.mean()), "dynamic_mask_fraction": float(dynamic_mask.mean()),
                            "static_supervision_fraction": float(static_mask.mean()), "road_depth": road_depth_record})
            image_id += 1
        print(f"[{output_frame + 1}/{len(selected)}] front source frame={int(reference_index)}", flush=True)
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    if options.filter_dynamic_initialization:
        dynamic_removed = dynamic_initialization_hits >= options.dynamic_initialization_min_observations
        # Keep a filtering failure explicit.  An empty initializer is a data
        # corruption, not a condition the official 2DGS trainer can diagnose.
        if not np.any(~dynamic_removed):
            raise ValueError("dynamic initialization filter removed every candidate point")
        initialisation.update({
            "dynamic_initialization_filter": {
                "enabled": True,
                "evidence": "central virtual-pinhole projection into accepted dynamic masks; one count per source camera/reference frame",
                "minimum_independent_observations": options.dynamic_initialization_min_observations,
                "observations_with_nonempty_dynamic_mask": dynamic_initialization_observations,
                "removed_points": int(dynamic_removed.sum()),
                "removed_fraction": float(dynamic_removed.mean()),
                "hit_count_quantiles": [float(value) for value in np.quantile(dynamic_initialization_hits, (0.0, .5, .9, .99, 1.0))],
            },
        })
        initial_positions, initial_rgb = initial_positions[~dynamic_removed], initial_rgb[~dynamic_removed]
    else:
        initialisation["dynamic_initialization_filter"] = {"enabled": False}
    init_stats = _write_initial_ply(
        sparse_dir / "points3D.ply", initial_positions, initial_rgb, initialisation,
        road_lidar_seed=options.road_lidar_seed,
    )
    if options.road_depth_supervision:
        road_depth_summary["lidar_frames"] = sorted(int(value) for value in road_depth_summary["lidar_frames"])
        road_depth_summary.update({
            "stride": options.road_depth_stride, "holdout_every": options.road_depth_holdout_every,
            "holdout_offset": options.road_depth_holdout_offset, "range_m": [options.road_depth_min_range_m, options.road_depth_max_range_m],
            "local_z_band_m": [options.road_depth_ground_local_z_min_m, options.road_depth_ground_local_z_max_m],
            "lidar_scan_window": options.road_depth_lidar_window,
            "max_lidar_delta_us": options.road_depth_max_lidar_delta_us,
            "lidar_decay_us": options.road_depth_lidar_decay_us,
            "ego_exclusion_radius_m": options.road_depth_ego_exclusion_radius_m,
            "loss_target": "camera_z_m", "normal_target": "world_unit_normal",
        })
    manifest = {"schema": "ncore-2dgs-pinhole-proxy", "version": 2, "ncore_path": str(options.ncore_path.resolve()),
                "static_ply": str(options.static_ply.resolve()), "chunk_index": options.chunk_index,
                "width": options.width, "height": options.height, "horizontal_fov_deg": options.horizontal_fov_deg,
                "camera_ids": list(options.camera_ids), "tile_yaws_deg": list(options.tile_yaws_deg),
                "rear_tile_yaws_deg": list(options.rear_tile_yaws_deg),
                "near_field_camera_ids": list(options.near_field_camera_ids),
                "near_field_tile_yaws_deg": list(options.near_field_tile_yaws_deg),
                "near_field_tile_pitch_deg": options.near_field_tile_pitch_deg, "records": records,
                "initialisation": init_stats,
                "dynamic_mask_manifest": None if options.dynamic_mask_manifest is None else str(options.dynamic_mask_manifest.resolve()),
                "filter_dynamic_initialization": options.filter_dynamic_initialization,
                "dynamic_initialization_min_observations": options.dynamic_initialization_min_observations,
                "road_lidar_seed": None if options.road_lidar_seed is None else str(options.road_lidar_seed.resolve()),
                "replace_road_initialization": options.replace_road_initialization,
                "road_depth_supervision": road_depth_summary if options.road_depth_supervision else None,
                "rgb_supervision": "all_valid_pixels" if options.dynamic_mask_manifest is None else "valid_pixels_excluding_accepted_source_priority_dynamic_instance_masks",
                "limitations": ["Official 2DGS supports undistorted pinhole cameras only.", "Images are resampled from native NCore FTheta with invalid pixels blacked out; official 2DGS has no validity-mask loss.", "Rolling shutter is approximated by the exposure midpoint.", "Road LiDAR depth is sparse and only constrains retained static ground returns; it does not create unobserved road geometry.", "This proxy cannot be used as an exact-FTheta/rolling final output without a native renderer and independent validation."]}
    manifest_path = output / "ncore_2dgs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote 2DGS NCore pinhole proxy dataset: {manifest_path}", flush=True)
    return manifest_path
