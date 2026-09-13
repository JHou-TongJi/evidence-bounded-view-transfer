"""Build conservative static road-surface seeds for the official 2DGS proxy.

The source PLY's road mask is useful for removing its strip-like road
initialisation, but it must not be treated as a depth measurement.  This
module instead keeps only raw top-LiDAR returns in a physically plausible
ground-height band, samples their colour through the native FTheta rolling
camera projection, rejects source-priority dynamic masks, and fuses the
remaining world points in voxels.  The output is an initialisation asset, not
depth supervision or a generated road surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path, _ncore_exposure_timestamps
from .street_dataset import STREET_CAMERA_IDS, _chunk_reference_indices, _nearest_index
from .two_dgs_dataset import _dynamic_masks_by_source_observation


SCHEMA = "ncore-2dgs-road-lidar-seed"
VERSION = 1


@dataclass(frozen=True)
class TwoDgsRoadLidarSeedOptions:
    ncore_path: Path
    output: Path
    dynamic_mask_manifest: Path
    chunk_index: int = 0
    source_camera_id: str = "camera_front_wide_120fov"
    frame_start: int | None = 0
    frame_end: int | None = 299
    stride: int = 5
    voxel_size_m: float = 0.15
    min_voxel_observations: int = 2
    max_points: int = 100_000
    min_range_m: float = 3.0
    max_range_m: float = 60.0
    ground_local_z_min_m: float = -3.2
    ground_local_z_max_m: float = -1.4
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.stride <= 0 or self.max_points <= 0:
            raise ValueError("chunk index, stride and max_points must be positive")
        if self.source_camera_id not in STREET_CAMERA_IDS:
            raise ValueError(f"source_camera_id must be one of {STREET_CAMERA_IDS}")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.voxel_size_m <= 0.0 or self.min_voxel_observations <= 0:
            raise ValueError("voxel settings must be positive")
        if self.min_range_m <= 0.0 or self.max_range_m <= self.min_range_m:
            raise ValueError("LiDAR range limits are invalid")
        if self.ground_local_z_max_m <= self.ground_local_z_min_m:
            raise ValueError("ground local-z band is invalid")
        if self.output.suffix.lower() != ".npz":
            raise ValueError("road LiDAR seed output must use .npz")


def _raw_dynamic_union(paths: tuple[Path, ...], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        if mask.shape != shape:
            raise ValueError(f"dynamic mask {path} shape {mask.shape} does not match source image {shape}")
        result |= mask
    return result


def _project_world_points_native_rolling(model: Any, world: np.ndarray, camera: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices and raw FTheta pixels for visible world points."""
    projected = model.world_points_to_pixels_shutter_pose(
        np.asarray(world, dtype=np.float32),
        np.asarray(camera.world_to_camera, dtype=np.float64),
        np.asarray(camera.world_to_camera_end, dtype=np.float64),
        int(camera.timestamp_start_us),
        int(camera.timestamp_us),
        return_T_world_sensors=True,
        return_valid_indices=True,
    )
    indices = projected.valid_indices.detach().cpu().numpy().astype(np.int64)
    pixels = projected.pixels.detach().float().cpu().numpy().astype(np.float32)
    return indices, pixels


def _fuse_voxels(world: np.ndarray, rgb: np.ndarray, voxel_size_m: float, min_observations: int, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(world) == 0:
        raise ValueError("no coloured LiDAR road returns survived projection and masks")
    keys = np.floor(world / float(voxel_size_m)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    count = np.bincount(inverse, minlength=int(inverse.max()) + 1).astype(np.float32)
    world_sum = np.zeros((len(count), 3), dtype=np.float64)
    rgb_sum = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(world_sum, inverse, world)
    np.add.at(rgb_sum, inverse, rgb)
    keep = count >= float(min_observations)
    centres = (world_sum[keep] / count[keep, None]).astype(np.float32)
    colours = np.clip(rgb_sum[keep] / count[keep, None], 0.0, 1.0).astype(np.float32)
    support = count[keep].astype(np.uint16)
    if len(centres) == 0:
        raise ValueError("no LiDAR road voxels meet min_voxel_observations")
    if len(centres) > max_points:
        # The unique keys are lexicographically ordered, so evenly sampling
        # that order remains deterministic and spatially distributed.
        selected = np.linspace(0, len(centres) - 1, max_points, dtype=np.int64)
        centres, colours, support = centres[selected], colours[selected], support[selected]
    return centres, colours, support


def build_two_dgs_road_lidar_seed(options: TwoDgsRoadLidarSeedOptions) -> Path:
    """Build a hashable, colour-initialised static road seed asset."""
    import torch
    from ncore.impl.sensors.camera import FThetaCameraModel

    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing road LiDAR seed: {output}")
    loader = _create_ncore_loader(options.ncore_path)
    sensors = {camera: loader.get_camera_sensor(camera) for camera in STREET_CAMERA_IDS}
    selection = type("Selection", (), {
        "chunk_index": options.chunk_index,
        "frame_start": options.frame_start,
        "frame_end": options.frame_end,
        "stride": options.stride,
    })()
    references = _chunk_reference_indices(loader, sensors, selection)
    source_sensor = sensors[options.source_camera_id]
    source_exposures = _ncore_exposure_timestamps(source_sensor)
    reference_exposures = _ncore_exposure_timestamps(sensors[STREET_CAMERA_IDS[0]])
    lidar = loader.get_lidar_sensor("lidar_top_360fov")
    model = FThetaCameraModel(source_sensor.model_parameters, device=options.device)
    masks = _dynamic_masks_by_source_observation(options.dynamic_mask_manifest.resolve())
    all_world: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    audit: list[dict[str, int]] = []
    for ordinal, reference_index in enumerate(references):
        reference_timestamp = int(reference_exposures[reference_index, 1])
        source_index = _nearest_index(source_exposures[:, 1].astype(np.int64), reference_timestamp)
        lidar_index = int(lidar.get_closest_frame_index(reference_timestamp))
        local = np.asarray(lidar.get_frame_point_cloud(lidar_index, False, False).xyz_m_end, dtype=np.float32)
        ranges = np.linalg.norm(local, axis=1)
        ground = (
            np.isfinite(local).all(axis=1)
            & (ranges >= options.min_range_m)
            & (ranges <= options.max_range_m)
            & (local[:, 2] >= options.ground_local_z_min_m)
            & (local[:, 2] <= options.ground_local_z_max_m)
        )
        local = local[ground]
        lidar_to_world = np.asarray(lidar.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
        world = local @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        camera = _load_ncore_source_camera_path({"frame_indices": [source_index]}, loader, options.source_camera_id).frames[0]
        image = np.asarray(source_sensor.get_frame_image_array(source_index), dtype=np.uint8)
        source_indices, pixels = _project_world_points_native_rolling(model, world, camera)
        xx = np.rint(pixels[:, 0]).astype(np.int64)
        yy = np.rint(pixels[:, 1]).astype(np.int64)
        within = (xx >= 0) & (xx < image.shape[1]) & (yy >= 0) & (yy < image.shape[0])
        source_indices, xx, yy = source_indices[within], xx[within], yy[within]
        dynamic = _raw_dynamic_union(
            tuple(masks.get((options.source_camera_id, int(reference_index), int(source_index)), ())),
            image.shape[:2],
        )
        static = ~dynamic[yy, xx]
        kept_world = world[source_indices[static]]
        kept_rgb = image[yy[static], xx[static]].astype(np.float32) / 255.0
        all_world.append(kept_world.astype(np.float32))
        all_rgb.append(kept_rgb.astype(np.float32))
        audit.append({
            "reference_index": int(reference_index), "source_frame_index": int(source_index),
            "lidar_frame_index": int(lidar_index), "raw_points": int(len(ground)),
            "ground_points": int(len(local)), "projected_points": int(len(source_indices)),
            "static_coloured_points": int(len(kept_world)),
        })
        if ordinal == 0 or (ordinal + 1) % 10 == 0 or ordinal + 1 == len(references):
            print(
                f"Road LiDAR seed [{ordinal + 1}/{len(references)}] ref={int(reference_index)} "
                f"ground={len(local)} coloured={len(kept_world)}",
                flush=True,
            )
    centres, colours, support = _fuse_voxels(
        np.concatenate(all_world, axis=0), np.concatenate(all_rgb, axis=0), options.voxel_size_m,
        options.min_voxel_observations, options.max_points,
    )
    metadata = {
        "schema": SCHEMA, "version": VERSION, "ncore_path": str(options.ncore_path.resolve()),
        "dynamic_mask_manifest": str(options.dynamic_mask_manifest.resolve()), "chunk_index": options.chunk_index,
        "source_camera_id": options.source_camera_id, "reference_indices": [int(value) for value in references],
        "frame_stride": options.stride, "voxel_size_m": options.voxel_size_m,
        "min_voxel_observations": options.min_voxel_observations, "min_range_m": options.min_range_m,
        "max_range_m": options.max_range_m, "ground_local_z_band_m": [options.ground_local_z_min_m, options.ground_local_z_max_m],
        "seed_count": int(len(centres)), "audit": audit,
        "limitations": [
            "Seeds are conservative raw-LiDAR ground returns; they do not create unobserved road geometry.",
            "Colour is sampled from one native FTheta/rolling source camera and is only an initialisation prior.",
            "Dynamic source-priority masks only reject observed dynamic pixels; they do not prove all retained points are static.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, xyz=np.asarray(centres, np.float32), rgb=np.asarray(colours, np.float32),
        support=np.asarray(support, np.uint16), metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"Wrote 2DGS road LiDAR seed ({len(centres)} voxels): {output}", flush=True)
    return output
