"""Prepare a pinhole training set for the external Street Gaussians checkout.

Street Gaussians is a pinhole/global-shutter implementation.  NCore cameras
are FTheta rolling-shutter cameras, therefore this converter deliberately
creates a *training proxy*: every source frame is re-sampled into a virtual
pinhole camera at the exposure midpoint.  The proxy is suitable for testing
object-centric reconstruction, but it does not replace this repository's
FTheta/rolling-shutter final renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .camera import compose_ncore_camera_pose
from .chunking import chunk_time_range, sequence_sampling_interval
from .ply_io import load_gaussian_ply
from .pose_sources import _create_ncore_loader, _ncore_exposure_timestamps


# Keep the model-facing order fixed.  It resembles the five-camera Waymo
# layout expected by the upstream project, but names remain explicit in the
# generated manifest instead of pretending NCore is Waymo.
STREET_CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
)
REAR_PROXY_CAMERA_IDS = frozenset((
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
))
MANIFEST_NAME = "ncore_street_manifest.json"


@dataclass(frozen=True)
class ProxyCameraSpec:
    """One virtual pinhole camera sampled from an NCore source sensor."""

    camera_id: str
    source_camera_id: str
    yaw_deg: float
    horizontal_fov_deg: float


@dataclass(frozen=True)
class StreetDatasetOptions:
    ncore_path: Path
    static_ply: Path
    output: Path
    chunk_index: int = 0
    width: int = 960
    height: int = 540
    horizontal_fov_deg: float = 100.0
    rear_horizontal_fov_deg: float = 60.0
    proxy_layout: str = "single"
    tile_horizontal_fov_deg: float = 55.0
    rear_tile_horizontal_fov_deg: float = 55.0
    tile_yaws_deg: tuple[float, ...] = (-40.0, 0.0, 40.0)
    rear_tile_yaws_deg: tuple[float, ...] = (0.0,)
    validation_stride: int = 20
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    max_background_points: int = 300_000
    voxel_size_m: float = 0.20
    rectify_device: str = "cuda"
    include_actors: bool = True
    exclude_sky_initialization: bool = True
    write_actor_masks: bool = False
    actor_mask_margin_pixels: int = 8
    actor_mask_segformer_model_dir: Path | None = None
    actor_mask_segformer_device: str = "cuda"
    actor_mask_vehicle_probability_threshold: float = 0.60
    actor_mask_vehicle_margin_threshold: float = 0.05
    actor_mask_vehicle_erosion_pixels: int = 0
    write_lidar_depth: bool = False
    max_lidar_timestamp_delta_us: int = 100_000

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("chunk index and output resolution must be positive")
        if not 20.0 <= self.horizontal_fov_deg < 175.0:
            raise ValueError("horizontal_fov_deg must be in [20, 175)")
        if not 20.0 <= self.rear_horizontal_fov_deg < 175.0:
            raise ValueError("rear_horizontal_fov_deg must be in [20, 175)")
        if self.proxy_layout not in {"single", "tiles"}:
            raise ValueError("proxy_layout must be 'single' or 'tiles'")
        if not 20.0 <= self.tile_horizontal_fov_deg < 175.0:
            raise ValueError("tile_horizontal_fov_deg must be in [20, 175)")
        if not 20.0 <= self.rear_tile_horizontal_fov_deg < 175.0:
            raise ValueError("rear_tile_horizontal_fov_deg must be in [20, 175)")
        if not self.tile_yaws_deg or not self.rear_tile_yaws_deg:
            raise ValueError("tile layouts require at least one yaw")
        if not all(np.isfinite(value) and abs(value) < 90.0 for value in (*self.tile_yaws_deg, *self.rear_tile_yaws_deg)):
            raise ValueError("tile yaw angles must be finite and in (-90, 90) degrees")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.stride <= 0 or self.validation_stride <= 1 or self.max_background_points <= 0 or self.voxel_size_m <= 0:
            raise ValueError("stride, validation_stride, max_background_points and voxel_size_m must be positive")
        if self.actor_mask_margin_pixels < 0 or self.max_lidar_timestamp_delta_us <= 0:
            raise ValueError("actor mask margin and LiDAR timestamp tolerance must be positive")
        if self.actor_mask_segformer_model_dir is not None and not self.write_actor_masks:
            raise ValueError("actor_mask_segformer_model_dir requires write_actor_masks")
        if not 0.0 <= self.actor_mask_vehicle_probability_threshold <= 1.0:
            raise ValueError("actor mask vehicle probability threshold must be in [0, 1]")
        if not -1.0 <= self.actor_mask_vehicle_margin_threshold <= 1.0:
            raise ValueError("actor mask vehicle margin threshold must be in [-1, 1]")
        if self.actor_mask_vehicle_erosion_pixels < 0:
            raise ValueError("actor mask vehicle erosion must be non-negative")


def pinhole_intrinsics(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    return np.asarray(
        [[focal, 0.0, (width - 1.0) / 2.0], [0.0, focal, (height - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def proxy_horizontal_fov_deg(camera_id: str, horizontal_fov_deg: float, rear_horizontal_fov_deg: float) -> float:
    """Choose a conservative pinhole FOV for each NCore camera family.

    The rear 70-degree sensors cannot support the 100-degree proxy used by
    front/cross cameras.  Asking them to do so creates a large black invalid
    region, which must never become RGB supervision.
    """
    return rear_horizontal_fov_deg if camera_id in REAR_PROXY_CAMERA_IDS else horizontal_fov_deg


def proxy_intrinsic_for_camera(
    camera_id: str,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    rear_horizontal_fov_deg: float,
) -> np.ndarray:
    return pinhole_intrinsics(
        width,
        height,
        proxy_horizontal_fov_deg(camera_id, horizontal_fov_deg, rear_horizontal_fov_deg),
    )


def tile_to_source_pose(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Return ``T_tile_source`` for OpenCV-camera yaw and pitch.

    The virtual tile has its own +Z forward axis.  A positive yaw points that
    axis towards positive source-camera X (image right), so composing
    ``T_source_world @ T_tile_source`` produces its world pose.  OpenCV +Y is
    image down: a *negative* pitch points the tile's optical axis roadward
    (towards positive source-camera Y).
    """
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    pitch_cosine, pitch_sine = float(np.cos(pitch)), float(np.sin(pitch))
    output = np.eye(4, dtype=np.float64)
    yaw_rotation = np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )
    pitch_rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, pitch_cosine, -pitch_sine), (0.0, pitch_sine, pitch_cosine)),
        dtype=np.float64,
    )
    output[:3, :3] = yaw_rotation @ pitch_rotation
    return output


def _yaw_token(yaw_deg: float) -> str:
    sign = "p" if yaw_deg >= 0 else "m"
    magnitude = f"{abs(float(yaw_deg)):.1f}".rstrip("0").rstrip(".")
    return f"{sign}{magnitude.replace('.', 'd')}"


def proxy_camera_specs(options: StreetDatasetOptions) -> tuple[ProxyCameraSpec, ...]:
    """Expand source sensors into either legacy single views or narrow tiles."""
    if options.proxy_layout == "single":
        return tuple(
            ProxyCameraSpec(
                camera_id=camera_id,
                source_camera_id=camera_id,
                yaw_deg=0.0,
                horizontal_fov_deg=proxy_horizontal_fov_deg(
                    camera_id, options.horizontal_fov_deg, options.rear_horizontal_fov_deg
                ),
            )
            for camera_id in STREET_CAMERA_IDS
        )
    specs: list[ProxyCameraSpec] = []
    for source_camera_id in STREET_CAMERA_IDS:
        rear = source_camera_id in REAR_PROXY_CAMERA_IDS
        yaws = options.rear_tile_yaws_deg if rear else options.tile_yaws_deg
        fov = options.rear_tile_horizontal_fov_deg if rear else options.tile_horizontal_fov_deg
        for yaw_deg in yaws:
            specs.append(
                ProxyCameraSpec(
                    camera_id=f"{source_camera_id}__tile_{_yaw_token(yaw_deg)}",
                    source_camera_id=source_camera_id,
                    yaw_deg=float(yaw_deg),
                    horizontal_fov_deg=float(fov),
                )
            )
    return tuple(specs)


def virtual_pinhole_rays(width: int, height: int, intrinsic: np.ndarray) -> np.ndarray:
    """OpenCV camera rays at pixel centres, one per output pixel."""
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    rays = np.stack(
        ((xx - intrinsic[0, 2]) / intrinsic[0, 0], (yy - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(xx)),
        axis=-1,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def virtual_pinhole_rays_in_source(
    width: int,
    height: int,
    intrinsic: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Return virtual pinhole rays expressed in the physical source camera."""
    rays = virtual_pinhole_rays(width, height, intrinsic)
    return rays @ tile_to_source_pose(yaw_deg, pitch_deg)[:3, :3].T


def _rectify_frame(
    sensor: object,
    image: np.ndarray,
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    device: str,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an optionally yawed/pitched pinhole tile from a source FTheta image.

    ``virtual_pinhole_rays`` are expressed in the virtual tile's OpenCV
    camera axes.  NCore's FTheta inverse projection, on the other hand,
    expects source-camera rays.  Rotating the rays before projection is
    essential: changing only the output camera pose would attach centre-view
    pixels to a side-looking tile and create inconsistent supervision.
    """
    import torch
    import torch.nn.functional as functional
    from ncore.impl.sensors.camera import FThetaCameraModel

    model = FThetaCameraModel(sensor.model_parameters, device=device)
    rays_np = virtual_pinhole_rays_in_source(width, height, intrinsic, yaw_deg, pitch_deg).reshape(-1, 3)
    rays = torch.from_numpy(rays_np).to(device)
    returned = model.camera_rays_to_pixels(rays)
    pixels = returned.pixels.reshape(height, width, 2).to(dtype=torch.float32)
    valid = returned.valid_flag.reshape(height, width)
    source_h, source_w = image.shape[:2]
    # align_corners=True makes -1/+1 mean the centres of the outer pixels.
    grid = pixels.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(source_w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(source_h - 1, 1) - 1.0
    source = torch.from_numpy(np.array(image, copy=True)).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    sampled = functional.grid_sample(source, grid[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    valid = valid & torch.isfinite(grid).all(dim=-1) & (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    sampled[:, ~valid] = 0.0
    return (
        (sampled.permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()),
        valid.byte().cpu().numpy(),
    )


def _chunk_reference_indices(loader: object, sensors: dict[str, object], options: StreetDatasetOptions) -> np.ndarray:
    timestamps = {name: _ncore_exposure_timestamps(sensor)[:, 1].astype(np.int64) for name, sensor in sensors.items()}
    start, end = sequence_sampling_interval(loader, sensors, timestamps)
    layout = chunk_time_range(start, end, options.chunk_index, n_frames_per_chunk=18, max_frame_gap_timestamp_us=750_000)
    source = timestamps[STREET_CAMERA_IDS[0]]
    end_inclusive = layout.chunk_index + 1 == layout.chunk_count
    selected = np.flatnonzero((source >= layout.start_us) & (source <= layout.end_us if end_inclusive else source < layout.end_us))
    selected = selected[slice(options.frame_start, options.frame_end, options.stride)]
    if len(selected) == 0:
        raise ValueError("requested frame window selects no source frames")
    return selected.astype(np.int64)


def _nearest_index(timestamps: np.ndarray, timestamp: int) -> int:
    pos = int(np.searchsorted(timestamps, timestamp))
    candidates = [max(0, min(len(timestamps) - 1, pos - 1)), max(0, min(len(timestamps) - 1, pos))]
    return min(candidates, key=lambda index: abs(int(timestamps[index]) - timestamp))


def find_vehicle_label_ids(id2label: dict[Any, Any]) -> tuple[int, ...]:
    """Resolve ADE-style rigid road-vehicle labels from model metadata.

    The exact class IDs are model-dependent.  Keeping this lookup label-based
    makes the dataset builder fail explicitly if a different local checkpoint
    cannot provide a compatible vehicle class instead of silently training on
    broad cuboid rectangles again.
    """
    supported = {"car", "bus", "truck", "van"}
    matches = sorted(
        int(index)
        for index, label in id2label.items()
        if str(label).strip().casefold() in supported
    )
    if not matches:
        raise ValueError("SegFormer model has no supported vehicle labels (car, bus, truck, van)")
    return tuple(matches)


class SegformerVehicleSegmenter:
    """Local-only semantic vehicle scores for cuboid-mask refinement."""

    def __init__(self, model_dir: Path, device: str) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"SegFormer model directory does not exist: {model_dir}")
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - optional environment dependency
            raise RuntimeError("semantic actor masks require pip install '.[sky]'") from exc
        self.torch = torch
        self.device = torch.device(device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(f"failed to load local SegFormer snapshot from {model_dir}") from exc
        self.model.to(self.device).eval()
        self.vehicle_label_ids = find_vehicle_label_ids(self.model.config.id2label)
        self.vehicle_labels = tuple(str(self.model.config.id2label[index]) for index in self.vehicle_label_ids)

    def vehicle_scores(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return vehicle probability and vehicle-vs-other semantic margin."""
        return self.vehicle_scores_batch((image,))[0]

    def vehicle_scores_batch(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 8,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch local SegFormer inference while preserving each input resolution."""
        if not images:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len({(image.width, image.height) for image in images}) != 1:
            raise ValueError("vehicle_scores_batch requires images with one shared resolution")
        torch = self.torch
        output: list[tuple[np.ndarray, np.ndarray]] = []
        height, width = images[0].height, images[0].width
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=(height, width), mode="bilinear", align_corners=False
                )
                probabilities = logits.softmax(dim=1)
                vehicle_probability = probabilities[:, list(self.vehicle_label_ids)].sum(dim=1)
                non_vehicle = probabilities.clone()
                non_vehicle[:, list(self.vehicle_label_ids)] = -1.0
                margin = vehicle_probability - non_vehicle.max(dim=1).values
            output.extend(zip(
                vehicle_probability.float().cpu().numpy(),
                margin.float().cpu().numpy(),
                strict=True,
            ))
        return output


def _write_simple_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["nx"], data["ny"], data["nz"] = 0.0, 0.0, 0.0
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    data["red"], data["green"], data["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def _background_initialization(
    path: Path,
    output: Path,
    voxel_size_m: float,
    max_points: int,
    *,
    exclude_sky: bool,
) -> dict[str, int]:
    scene = load_gaussian_ply(path)
    rgb = np.clip(scene.sh_coeffs[:, 0] * 0.28209479177387814 + 0.5, 0.0, 1.0)
    selected = np.ones(scene.count, dtype=bool)
    sky_points_excluded = 0
    if exclude_sky:
        from plyfile import PlyData

        vertex = PlyData.read(str(path))["vertex"].data
        if "sky_mask" in (vertex.dtype.names or ()):
            selected = np.asarray(vertex["sky_mask"], dtype=np.float32) < 0.5
            sky_points_excluded = int((~selected).sum())
    means = scene.means[selected]
    rgb = rgb[selected]
    # Deterministic first-in-voxel selection avoids a 2.8M-point initial model
    # while retaining the original NCore world frame.
    voxels = np.floor(means / voxel_size_m).astype(np.int64)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    indices.sort()
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)]
    _write_simple_ply(output, means[indices], rgb[indices])
    return {
        "input_gaussians": scene.count,
        "sky_points_excluded": sky_points_excluded,
        "eligible_gaussians": int(len(means)),
        "voxel_representatives": int(len(np.unique(voxels, axis=0))),
        "written_points": int(len(indices)),
    }


def _extract_vehicle_tracks(ncore_path: Path, selected_timestamps_us: np.ndarray) -> list[dict[str, Any]]:
    """Get rigid NCore cuboid trajectories using the already validated upstream helper."""
    if len(selected_timestamps_us) < 2:
        return []
    try:
        from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
        from instant_nurec.utils.types import HalfClosedInterval
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("actor conversion requires the local InstantNuRec checkout on PYTHONPATH") from exc
    loader = _create_ncore_loader(ncore_path)
    start, end = int(selected_timestamps_us.min()) - 1_000_000, int(selected_timestamps_us.max()) + 1_000_001
    tracks = consolidate_cuboid_tracks(compute_cuboid_df(loader, HalfClosedInterval(start, end)), loader, ["AUTOLABEL"], 0.0, np.eye(4))
    output: list[dict[str, Any]] = []
    for raw_id, track in tracks.items():
        if str(track.get("label_class")) not in {"automobile", "heavy_truck"}:
            continue
        try:
            track_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        timestamps = np.asarray(track.get("timestamps_us"), dtype=np.int64)
        poses = np.asarray(track.get("poses"), dtype=np.float64)
        if poses.shape != (len(timestamps), 4, 4) or len(timestamps) < 2:
            continue
        movement = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
        if movement < 0.50:
            continue
        # InstantNuRec's consolidated track schema calls this field
        # ``dimension``.  Falling back to a passenger-car box for every actor
        # silently made trucks too small and caused Street's box pruning to
        # erase whole actors.
        size = np.asarray(
            track.get("dimension", track.get("size", track.get("sizes", [4.8, 2.0, 1.8]))),
            dtype=np.float64,
        ).reshape(-1)
        if len(size) != 3 or np.any(size <= 0):
            size = np.asarray([4.8, 2.0, 1.8], dtype=np.float64)
        output.append({"track_id": track_id, "label": str(track["label_class"]), "timestamps_us": timestamps.tolist(), "actor_to_world": poses.tolist(), "length_width_height": size.tolist()})
    return sorted(output, key=lambda item: item["track_id"])


def _interpolate_track_pose(track: dict[str, Any], timestamp_us: int) -> np.ndarray | None:
    """Linearly interpolate a tracked actor pose, without extrapolating it."""
    timestamps = np.asarray(track["timestamps_us"], dtype=np.int64)
    poses = np.asarray(track["actor_to_world"], dtype=np.float64)
    if timestamp_us < int(timestamps[0]) or timestamp_us > int(timestamps[-1]):
        return None
    right = int(np.searchsorted(timestamps, timestamp_us, side="left"))
    if right == 0 or int(timestamps[right]) == timestamp_us:
        return poses[right].copy()
    left = right - 1
    fraction = (timestamp_us - int(timestamps[left])) / (int(timestamps[right]) - int(timestamps[left]))
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (1.0 - fraction) * poses[left, :3, 3] + fraction * poses[right, :3, 3]
    u, _, vh = np.linalg.svd((1.0 - fraction) * poses[left, :3, :3] + fraction * poses[right, :3, :3])
    result[:3, :3] = u @ vh
    if np.linalg.det(result[:3, :3]) < 0:
        u[:, -1] *= -1
        result[:3, :3] = u @ vh
    return result


def _actor_supervision_mask(
    tracks: list[dict[str, Any]],
    timestamp_us: int,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    margin_pixels: int,
) -> np.ndarray:
    """Conservatively rasterize visible rigid cuboid bounds into a binary mask.

    This is deliberately a box mask, not an instance segmentation label.  It
    prevents a frozen/no-background actor model from being scored on road and
    sky pixels, while the configurable margin preserves vehicle boundaries.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    for track in tracks:
        actor_to_world = _interpolate_track_pose(track, timestamp_us)
        if actor_to_world is None:
            continue
        dimensions = np.asarray(track["length_width_height"], dtype=np.float64)
        local = signs * (dimensions[None, :] / 2.0)
        corners_world = local @ actor_to_world[:3, :3].T + actor_to_world[:3, 3]
        corners_camera = corners_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        front = corners_camera[:, 2] > 0.10
        if front.sum() < 2:
            continue
        projected = corners_camera[front, :2] / corners_camera[front, 2:3]
        projected = projected @ intrinsic[:2, :2].T + intrinsic[:2, 2]
        xmin, ymin = np.floor(projected.min(axis=0)).astype(int) - margin_pixels
        xmax, ymax = np.ceil(projected.max(axis=0)).astype(int) + margin_pixels
        if xmax < 0 or ymax < 0 or xmin >= width or ymin >= height:
            continue
        xmin, xmax = max(0, xmin), min(width - 1, xmax)
        ymin, ymax = max(0, ymin), min(height - 1, ymax)
        if xmin <= xmax and ymin <= ymax:
            mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return mask


def _semantic_actor_supervision_mask(
    cuboid_mask: np.ndarray,
    valid_mask: np.ndarray,
    vehicle_probability: np.ndarray,
    vehicle_margin: np.ndarray,
    *,
    probability_threshold: float,
    margin_threshold: float,
    erosion_pixels: int,
) -> np.ndarray:
    """Keep only high-confidence rigid-vehicle pixels inside cuboid bounds.

    This intentionally has no rectangle fallback: an uncertain semantic pixel
    is excluded from actor supervision rather than being turned into moving
    road/vegetation.  The NCore valid mask is included here as well as in the
    RGB loss so exported LiDAR supervision has exactly the same support.
    """
    shape = np.asarray(cuboid_mask).shape
    if any(np.asarray(value).shape != shape for value in (valid_mask, vehicle_probability, vehicle_margin)):
        raise ValueError("cuboid, valid, vehicle probability and vehicle margin shapes must match")
    semantic = (
        (np.asarray(vehicle_probability, dtype=np.float32) >= probability_threshold)
        & (np.asarray(vehicle_margin, dtype=np.float32) >= margin_threshold)
    )
    if erosion_pixels:
        from .sky_build import erode_boolean_mask

        semantic = erode_boolean_mask(semantic, erosion_pixels)
    return (
        np.asarray(cuboid_mask, dtype=bool)
        & np.asarray(valid_mask, dtype=bool)
        & semantic
    ).astype(np.uint8)


def _project_lidar_depth(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Z-buffer sparse LiDAR points into a virtual pinhole camera in metres."""
    world_to_camera = np.linalg.inv(camera_to_world)
    points_camera = points_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (depth > 0.10)
    if not valid.any():
        return np.zeros((height, width), dtype=np.float32)
    points_camera = points_camera[valid]
    depth = depth[valid]
    projected = points_camera[:, :2] / depth[:, None]
    projected = projected @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    xx = np.rint(projected[:, 0]).astype(np.int64)
    yy = np.rint(projected[:, 1]).astype(np.int64)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flat, yy[valid] * width + xx[valid], depth[valid].astype(np.float32))
    flat[~np.isfinite(flat)] = 0.0
    return flat.reshape(height, width)


def build_street_dataset(options: StreetDatasetOptions) -> Path:
    """Write a self-contained NCore proxy dataset and return its manifest path."""
    import ncore.data

    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir()
    masks_dir = output / "masks"
    masks_dir.mkdir()
    actor_masks_dir = output / "actor_masks" if options.write_actor_masks else None
    if actor_masks_dir is not None:
        actor_masks_dir.mkdir()
    lidar_depth_dir = output / "lidar_depth" if options.write_lidar_depth else None
    if lidar_depth_dir is not None:
        lidar_depth_dir.mkdir()
    vehicle_segmenter = (
        None
        if options.actor_mask_segformer_model_dir is None
        else SegformerVehicleSegmenter(
            options.actor_mask_segformer_model_dir,
            options.actor_mask_segformer_device,
        )
    )
    loader = _create_ncore_loader(options.ncore_path)
    sensors = {name: loader.get_camera_sensor(name) for name in STREET_CAMERA_IDS}
    proxy_cameras = proxy_camera_specs(options)
    selected = _chunk_reference_indices(loader, sensors, options)
    reference_end = _ncore_exposure_timestamps(sensors[STREET_CAMERA_IDS[0]])[:, 1].astype(np.int64)
    reference_exposures = _ncore_exposure_timestamps(sensors[STREET_CAMERA_IDS[0]])
    selected_midpoints = ((reference_exposures[selected, 0].astype(np.int64) + reference_exposures[selected, 1].astype(np.int64)) // 2)
    tracks = _extract_vehicle_tracks(options.ncore_path, selected_midpoints) if options.include_actors else []
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov") if options.write_lidar_depth else None
    default_intrinsic = pinhole_intrinsics(options.width, options.height, options.horizontal_fov_deg)
    frames: list[dict[str, Any]] = []
    source_indices: dict[str, list[int]] = {spec.camera_id: [] for spec in proxy_cameras}
    print(
        f"Preparing {len(selected)} NCore reference frames × {len(proxy_cameras)} proxy cameras "
        f"at {options.width}×{options.height}"
        + (" with SegFormer vehicle-mask refinement" if vehicle_segmenter is not None else ""),
        flush=True,
    )
    for output_index, reference_index in enumerate(selected):
        reference_timestamp = int(reference_end[reference_index])
        lidar_world: np.ndarray | None = None
        lidar_frame_index: int | None = None
        lidar_timestamp_us: int | None = None
        if lidar_sensor is not None:
            lidar_frame_index = int(lidar_sensor.get_closest_frame_index(reference_timestamp))
            lidar_timestamp_us = int(lidar_sensor.get_frame_timestamp_us(lidar_frame_index))
            if abs(lidar_timestamp_us - reference_timestamp) <= options.max_lidar_timestamp_delta_us:
                lidar_points = lidar_sensor.get_frame_point_cloud(lidar_frame_index, False, False).xyz_m_end
                lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index), dtype=np.float64)
                lidar_world = np.asarray(lidar_points, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        for camera_index, proxy_camera in enumerate(proxy_cameras):
            sensor = sensors[proxy_camera.source_camera_id]
            intrinsic = pinhole_intrinsics(options.width, options.height, proxy_camera.horizontal_fov_deg)
            exposures = _ncore_exposure_timestamps(sensor)
            source_index = _nearest_index(exposures[:, 1].astype(np.int64), reference_timestamp)
            source_indices[proxy_camera.camera_id].append(source_index)
            image = np.asarray(sensor.get_frame_image_array(source_index), dtype=np.uint8)
            rectified, valid = _rectify_frame(
                sensor,
                image,
                intrinsic,
                width=options.width,
                height=options.height,
                device=options.rectify_device,
                yaw_deg=proxy_camera.yaw_deg,
            )
            name = f"{output_index:06d}_{camera_index}.png"
            Image.fromarray(rectified, mode="RGB").save(images_dir / name)
            Image.fromarray((valid * 255).astype(np.uint8), mode="L").save(masks_dir / name)
            midpoint = int((int(exposures[source_index, 0]) + int(exposures[source_index, 1])) // 2)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            tile_to_source = tile_to_source_pose(proxy_camera.yaw_deg)
            source_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            camera_to_world = source_to_world @ tile_to_source
            camera_to_rig = np.asarray(sensor.T_sensor_rig) @ tile_to_source
            frame = {
                "frame_index": output_index, "camera_index": camera_index, "camera_id": proxy_camera.camera_id,
                "source_camera_id": proxy_camera.source_camera_id, "tile_yaw_deg": proxy_camera.yaw_deg,
                "source_frame_index": source_index, "timestamp_us": midpoint, "image": f"images/{name}",
                "valid_mask": f"masks/{name}", "intrinsic": intrinsic.tolist(),
                "horizontal_fov_deg": proxy_camera.horizontal_fov_deg,
                "is_validation": output_index % options.validation_stride == 0,
                "camera_to_world": camera_to_world.tolist(), "camera_to_rig": camera_to_rig.tolist(),
                "valid_fraction": float(valid.mean()),
            }
            actor_mask: np.ndarray | None = None
            if actor_masks_dir is not None:
                cuboid_mask = _actor_supervision_mask(
                    tracks, midpoint, camera_to_world, intrinsic, options.width, options.height,
                    options.actor_mask_margin_pixels,
                )
                actor_mask = cuboid_mask
                frame["cuboid_mask_fraction"] = float(cuboid_mask.mean())
                if vehicle_segmenter is not None:
                    segmentation_started = time.perf_counter()
                    vehicle_probability, vehicle_margin = vehicle_segmenter.vehicle_scores(
                        Image.fromarray(rectified, mode="RGB")
                    )
                    actor_mask = _semantic_actor_supervision_mask(
                        cuboid_mask,
                        valid,
                        vehicle_probability,
                        vehicle_margin,
                        probability_threshold=options.actor_mask_vehicle_probability_threshold,
                        margin_threshold=options.actor_mask_vehicle_margin_threshold,
                        erosion_pixels=options.actor_mask_vehicle_erosion_pixels,
                    )
                    frame["vehicle_semantic_fraction"] = float(
                        (
                            (vehicle_probability >= options.actor_mask_vehicle_probability_threshold)
                            & (vehicle_margin >= options.actor_mask_vehicle_margin_threshold)
                            & np.asarray(valid, dtype=bool)
                        ).mean()
                    )
                    frame["vehicle_segmentation_seconds"] = time.perf_counter() - segmentation_started
                Image.fromarray((actor_mask * 255).astype(np.uint8), mode="L").save(actor_masks_dir / name)
                frame["object_mask"] = f"actor_masks/{name}"
                frame["object_mask_fraction"] = float(actor_mask.mean())
            if lidar_depth_dir is not None and lidar_world is not None:
                lidar_depth = _project_lidar_depth(lidar_world, camera_to_world, intrinsic, options.width, options.height)
                # Actor-only training never scores the background.  Retaining
                # its dense LiDAR map would add roughly 3 GiB for this chunk
                # without affecting any loss, so keep only projected points
                # within the same conservative vehicle mask and compress it.
                if actor_mask is not None:
                    lidar_depth[actor_mask == 0] = 0.0
                depth_name = name.replace(".png", ".npz")
                np.savez_compressed(lidar_depth_dir / depth_name, depth=lidar_depth)
                frame["lidar_depth"] = f"lidar_depth/{depth_name}"
                frame["lidar_frame_index"] = lidar_frame_index
                frame["lidar_timestamp_us"] = lidar_timestamp_us
                frame["lidar_valid_count"] = int((lidar_depth > 0.0).sum())
            frames.append(frame)
        print(f"[{output_index + 1}/{len(selected)}] source front frame={int(reference_index)} timestamp_us={reference_timestamp}", flush=True)
    bkgd_stats = _background_initialization(
        options.static_ply,
        output / "input_ply" / "points3D_bkgd.ply",
        options.voxel_size_m,
        options.max_background_points,
        exclude_sky=options.exclude_sky_initialization,
    )
    manifest = {
        "schema": "ncore-street-gaussians-dataset", "version": 5,
        "ncore_path": str(options.ncore_path.resolve()), "static_ply": str(options.static_ply.resolve()),
        "chunk_index": options.chunk_index, "camera_ids": [spec.camera_id for spec in proxy_cameras],
        "source_camera_ids": list(STREET_CAMERA_IDS), "reference_proxy_camera_id": proxy_cameras[0].camera_id,
        "intrinsic": default_intrinsic.tolist(), "width": options.width, "height": options.height,
        "horizontal_fov_deg": options.horizontal_fov_deg,
        "per_camera_horizontal_fov_deg": {
            spec.camera_id: spec.horizontal_fov_deg for spec in proxy_cameras
        },
        "validation_stride": options.validation_stride,
        "proxy_camera_model": "pinhole_at_exposure_midpoint_from_ncore_ftheta",
        "proxy_layout": options.proxy_layout,
        "proxy_cameras": [
            {"camera_id": spec.camera_id, "source_camera_id": spec.source_camera_id,
             "tile_yaw_deg": spec.yaw_deg, "horizontal_fov_deg": spec.horizontal_fov_deg}
            for spec in proxy_cameras
        ],
        "frames": frames, "reference_source_indices": selected.tolist(), "per_camera_source_indices": source_indices,
        "tracks": tracks, "background_initialization": bkgd_stats,
        "actor_supervision_mask": {
            "kind": "cuboid_only" if vehicle_segmenter is None else "cuboid_intersection_semantic_vehicle",
            "cuboid_margin_pixels": options.actor_mask_margin_pixels,
            "segformer_model_dir": None if vehicle_segmenter is None else str(options.actor_mask_segformer_model_dir.resolve()),
            "segformer_vehicle_labels": None if vehicle_segmenter is None else list(vehicle_segmenter.vehicle_labels),
            "vehicle_probability_threshold": None if vehicle_segmenter is None else options.actor_mask_vehicle_probability_threshold,
            "vehicle_margin_threshold": None if vehicle_segmenter is None else options.actor_mask_vehicle_margin_threshold,
            "vehicle_erosion_pixels": None if vehicle_segmenter is None else options.actor_mask_vehicle_erosion_pixels,
        },
        "limitations": ["Training images are FTheta-undistorted virtual pinhole proxies.", "Rolling-shutter motion during an exposure is represented only by its midpoint.", "LiDAR depth is sparse and uses the nearest LiDAR frame within the configured timestamp tolerance when enabled.", "Actors are instantiated only over their observed NCore cuboid time range.", "Only valid FTheta-to-pinhole pixels supervise RGB loss.", "A semantic vehicle mask is class-level rather than instance-level; overlapping nearby vehicles can still be ambiguous." if vehicle_segmenter is not None else "Actor supervision masks are conservative projected cuboid bounds, not instance segmentation masks."],
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Street Gaussians proxy dataset: {manifest_path}", flush=True)
    return manifest_path
