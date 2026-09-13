"""Native NCore static-Gaussian optimisation pilot.

This module is intentionally a bounded replacement for trying to select or
splice InstantNuRec release PLYs.  It keeps original NCore FTheta calibration
and rolling-shutter exposure poses, starts from a supplied static PLY, masks
sky/ego/dynamic pixels, and optimises a selected subset of static Gaussians
against raw RGB plus projected LiDAR depths.  The first version is a pilot:
it does *not* densify or claim full-scene convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .color_correction import GaussianColorCorrection, file_sha256
from .ftheta_actor_optimize import _rasterize_rgb_alpha
from .ply_io import GaussianScene, load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .raw_dynamic_actor_train import _world_depth
from .render import GsplatRenderer, RenderOptions, _write_json_atomic
from .sky_build import SegformerSkySegmenter, _load_ego_mask, dilate_boolean_mask, refine_sky_mask
from .static_leakage import _project_world_points_ftheta


SCHEMA = "ncore-native-static-gaussian-optimisation"
VERSION = 1
DEFAULT_CAMERA_IDS = (
    "camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
    "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_front_tele_30fov", "camera_rear_tele_30fov",
)
_C0 = 0.28209479177387814


@dataclass(frozen=True)
class StaticSceneOptimizeOptions:
    initial_ply: Path
    ncore_path: Path
    dynamic_mask_manifest: Path
    segformer_model_dir: Path
    output_ply: Path | None
    train_reference_indices: tuple[int, ...]
    validation_reference_indices: tuple[int, ...]
    color_correction_output: Path | None = None
    initial_color_correction: Path | None = None
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    device: str = "cuda"
    width: int = 240
    height: int = 135
    trainable_gaussians: int = 100_000
    selection_mode: str = "opacity"
    optimization_mode: str = "joint"
    iterations: int = 250
    learning_rate: float = 2.0e-3
    max_position_delta_m: float = .10
    max_log_scale_delta: float = .20
    max_rgb_delta: float = .20
    max_logit_opacity_delta: float = 1.0
    rgb_weight: float = 1.0
    lidar_depth_weight: float = .05
    position_prior_weight: float = .01
    scale_prior_weight: float = .002
    color_prior_weight: float = .002
    opacity_prior_weight: float = .001
    sky_probability_threshold: float = .70
    sky_margin_threshold: float = .05
    sky_erosion_pixels: int = 2
    dynamic_dilation_pixels: int = 2
    max_lidar_samples_per_observation: int = 256
    lidar_support_radius_m: float = .50
    minimum_lidar_cameras: int = 2
    minimum_lidar_references: int = 2
    seed_voxel_size_m: float = .10
    seed_residual_support_radius_m: float = .30
    seed_scale_m: float = .08
    maximum_seed_gaussians: int = 2_048
    minimum_seed_gaussians: int = 32
    validation_every: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output_ply is not None and self.output_ply.suffix.lower() != ".ply":
            raise ValueError("output_ply must use .ply")
        if self.color_correction_output is not None and self.color_correction_output.suffix.lower() != ".npz":
            raise ValueError("color_correction_output must use .npz")
        if self.initial_color_correction is not None and self.initial_color_correction.suffix.lower() != ".npz":
            raise ValueError("initial_color_correction must use .npz")
        if self.output_ply is None and self.color_correction_output is None:
            raise ValueError("output_ply or color_correction_output is required")
        if not self.train_reference_indices or not self.validation_reference_indices:
            raise ValueError("non-empty disjoint train/validation reference indices are required")
        if set(self.train_reference_indices).intersection(self.validation_reference_indices):
            raise ValueError("train and validation reference indices must be disjoint")
        if any(index < 0 for index in (*self.train_reference_indices, *self.validation_reference_indices)):
            raise ValueError("reference indices must be non-negative")
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be non-empty and unique")
        if self.width <= 0 or self.height <= 0 or self.trainable_gaussians <= 0 or self.iterations < 0:
            raise ValueError("resolution, Gaussian budget and iterations are invalid")
        if self.selection_mode not in {"opacity", "residual"}:
            raise ValueError("selection_mode must be 'opacity' or 'residual'")
        if self.optimization_mode not in {"joint", "color_only", "geometry_lidar", "geometry_lidar_densify"}:
            raise ValueError("optimization_mode must be 'joint', 'color_only', 'geometry_lidar' or 'geometry_lidar_densify'")
        if self.optimization_mode == "color_only" and self.color_correction_output is None:
            raise ValueError("color_only mode requires color_correction_output to preserve geometry exactly")
        if self.learning_rate <= 0.0 or self.max_position_delta_m <= 0.0 or self.max_log_scale_delta <= 0.0:
            raise ValueError("optimisation bounds are invalid")
        if not 0.0 < self.max_rgb_delta <= 1.0 or self.max_logit_opacity_delta <= 0.0:
            raise ValueError("colour/opacity bounds are invalid")
        if not 0.0 <= self.sky_probability_threshold <= 1.0 or not -1.0 <= self.sky_margin_threshold <= 1.0:
            raise ValueError("sky thresholds are invalid")
        if self.sky_erosion_pixels < 0 or self.dynamic_dilation_pixels < 0 or self.max_lidar_samples_per_observation <= 0:
            raise ValueError("mask/LiDAR settings are invalid")
        if self.lidar_support_radius_m <= 0.0 or self.minimum_lidar_cameras <= 0 or self.minimum_lidar_references <= 0:
            raise ValueError("LiDAR support settings are invalid")
        if (self.seed_voxel_size_m <= 0.0 or self.seed_residual_support_radius_m <= 0.0 or self.seed_scale_m <= 0.0
                or self.maximum_seed_gaussians <= 0 or self.minimum_seed_gaussians <= 0
                or self.minimum_seed_gaussians > self.maximum_seed_gaussians):
            raise ValueError("densification seed settings are invalid")
        if self.validation_every <= 0 or any(value < 0.0 for value in (
            self.rgb_weight, self.lidar_depth_weight, self.position_prior_weight, self.scale_prior_weight,
            self.color_prior_weight, self.opacity_prior_weight,
        )):
            raise ValueError("loss settings are invalid")


@dataclass
class StaticObservation:
    split: str
    camera_id: str
    reference_frame_index: int
    source_frame_index: int
    camera: Any
    target_rgb: np.ndarray
    valid_mask: np.ndarray
    lidar_xy: np.ndarray
    lidar_depth_m: np.ndarray
    lidar_world_m: np.ndarray
    mask_fractions: dict[str, float]


@dataclass(frozen=True)
class StaticSurfaceSeeds:
    """Multi-view static LiDAR seeds admitted for the optional geometry stage.

    They are not RGB-only hallucinations: every seed is a voxelised raw LiDAR
    return from a sky/ego/dynamic-excluded pixel, observed from independent
    cameras and reference times, and close to an existing residual Gaussian.
    """
    means: np.ndarray
    rgb: np.ndarray
    support_count: np.ndarray
    report: dict[str, int | float]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0


def _dynamic_mask_index(manifest_path: Path, camera_ids: tuple[str, ...]) -> dict[tuple[str, int], list[Path]]:
    """Use only original source masks; SAM-only propagation never supervises static geometry."""
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("dynamic mask manifest must be complete")
    root = manifest_path.parent
    result: dict[tuple[str, int], list[Path]] = {}
    requested = set(camera_ids)
    for camera in manifest.get("cameras", []):
        if not isinstance(camera, dict) or str(camera.get("camera_id")) not in requested:
            continue
        camera_id = str(camera["camera_id"])
        for observation in camera.get("observations", []):
            if not isinstance(observation, dict) or observation.get("mask_origin") != "source":
                continue
            relative = observation.get("refined_mask")
            if not isinstance(relative, str):
                continue
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            key = (camera_id, int(observation["reference_frame_index"]))
            result.setdefault(key, []).append(path)
    return result


def _source_frame_index(manifest: dict[str, Any], camera_id: str, reference_index: int) -> int:
    for camera in manifest.get("cameras", []):
        if isinstance(camera, dict) and camera.get("camera_id") == camera_id:
            for observation in camera.get("observations", []):
                if isinstance(observation, dict) and int(observation.get("reference_frame_index", -1)) == reference_index:
                    return int(observation["source_frame_index"])
    raise ValueError(f"dynamic-mask manifest lacks {camera_id} reference={reference_index}")


def _union_dynamic_masks(paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for path in paths:
        value = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
        if value.shape != shape:
            raise ValueError(f"dynamic mask/image shape mismatch: {path}: {value.shape} != {shape}")
        result |= value
    return result


def _lidar_depth_samples(loader: object, camera_id: str, camera: Any, source_shape: tuple[int, int],
                         valid_mask: np.ndarray, *, width: int, height: int, device: str,
                         maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the closest raw top-LiDAR scan into exact camera midpoint space."""
    from ncore.impl.sensors.camera import FThetaCameraModel

    lidar = loader.get_lidar_sensor("lidar_top_360fov")
    timestamp = int(camera.timestamp_start_us + (camera.timestamp_us - camera.timestamp_start_us) // 2)
    index = int(lidar.get_closest_frame_index(timestamp))
    cloud = np.asarray(lidar.get_frame_point_cloud(index, False, False).xyz_m_end, np.float32)
    transform = np.asarray(lidar.get_frames_T_sensor_target("world", index), np.float64)
    world = cloud @ transform[:3, :3].T + transform[:3, 3]
    sensor = loader.get_camera_sensor(camera_id)
    model = FThetaCameraModel(sensor.model_parameters, device=device)
    midpoint = camera.world_to_camera
    if camera.world_to_camera_end is not None:
        # The production expected-depth convention evaluates the centre at t=.5.
        from .camera import interpolate_w2c
        rotation, translation = interpolate_w2c(camera.world_to_camera, camera.world_to_camera_end, np.asarray([.5]))
        midpoint = np.eye(4, dtype=np.float64); midpoint[:3, :3] = rotation[0]; midpoint[:3, 3] = translation[0]
    pixels, depths, projected = _project_world_points_ftheta(
        world, midpoint, model, source_width=source_shape[1], source_height=source_shape[0], width=width, height=height, device=device,
    )
    xx = np.where(projected, np.rint(pixels[:, 0]), -1).astype(np.int64)
    yy = np.where(projected, np.rint(pixels[:, 1]), -1).astype(np.int64)
    projected &= (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    projected &= valid_mask[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]
    candidates = np.flatnonzero(projected & np.isfinite(depths) & (depths > .1))
    if len(candidates):
        # Keep only the front-most LiDAR return per rendered pixel, matching
        # the expected-depth supervision convention.
        order = candidates[np.lexsort((depths[candidates], yy[candidates] * width + xx[candidates]))]
        flat = yy[order] * width + xx[order]
        candidates = order[np.r_[True, flat[1:] != flat[:-1]]]
    if len(candidates) > maximum:
        rng = np.random.default_rng(seed)
        candidates = rng.choice(candidates, size=maximum, replace=False)
    return (
        np.stack([xx[candidates], yy[candidates]], axis=1).astype(np.int64),
        depths[candidates].astype(np.float32),
        world[candidates].astype(np.float32),
    )


def load_static_observations(options: StaticSceneOptimizeOptions) -> list[StaticObservation]:
    dynamic_manifest = _read_json(options.dynamic_mask_manifest)
    if Path(dynamic_manifest.get("ncore_path", "")).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic-mask manifest belongs to a different NCore clip")
    index = _dynamic_mask_index(options.dynamic_mask_manifest, options.camera_ids)
    loader = _create_ncore_loader(options.ncore_path)
    segmenter = SegformerSkySegmenter(options.segformer_model_dir, options.device)
    observations: list[StaticObservation] = []
    pairs = [("train", value) for value in options.train_reference_indices] + [("validation", value) for value in options.validation_reference_indices]
    for split, reference_index in pairs:
        for camera_id in options.camera_ids:
            source_index = _source_frame_index(dynamic_manifest, camera_id, reference_index)
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [source_index], "output": {"width": options.width, "height": options.height}}, loader, camera_id
            ).frames[0]
            sensor = loader.get_camera_sensor(camera_id)
            raw = np.asarray(sensor.get_frame_image_array(source_index), np.uint8)
            image = Image.fromarray(raw, mode="RGB")
            rgb = np.asarray(image.resize((options.width, options.height), Image.Resampling.BILINEAR), np.float32) / 255.0
            dynamic = _resize_mask(_union_dynamic_masks(index.get((camera_id, reference_index), []), raw.shape[:2]), options.width, options.height)
            dynamic = dilate_boolean_mask(dynamic, options.dynamic_dilation_pixels)
            ego = _load_ego_mask(sensor, (options.width, options.height))
            sky_probability, sky_margin = segmenter.sky_scores(image.resize((options.width, options.height), Image.Resampling.BILINEAR))
            sky, _ = refine_sky_mask(
                sky_probability, sky_margin, rgb, ego, sky_threshold=options.sky_probability_threshold,
                margin_threshold=options.sky_margin_threshold, erosion_pixels=options.sky_erosion_pixels,
                edge_exclusion_pixels=0, edge_gradient_threshold=0.0,
            )
            valid = ~(dynamic | sky | ego)
            if int(valid.sum()) < 64:
                raise RuntimeError(f"static supervision mask is too small: {camera_id} ref={reference_index}")
            xy, depth, world = _lidar_depth_samples(
                loader, camera_id, camera, raw.shape[:2], valid, width=options.width, height=options.height,
                device=options.device, maximum=options.max_lidar_samples_per_observation,
                seed=options.seed + reference_index * 31 + len(observations),
            )
            observations.append(StaticObservation(
                split=split, camera_id=camera_id, reference_frame_index=reference_index, source_frame_index=source_index,
                camera=camera, target_rgb=rgb, valid_mask=valid, lidar_xy=xy, lidar_depth_m=depth, lidar_world_m=world,
                mask_fractions={"dynamic": float(dynamic.mean()), "sky": float(sky.mean()), "ego": float(ego.mean()), "valid": float(valid.mean())},
            ))
    del segmenter
    if not any(item.split == "train" for item in observations) or not any(item.split == "validation" for item in observations):
        raise RuntimeError("static observations require both train and validation samples")
    return observations


def select_trainable_gaussians(opacities: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministically prioritise visible Gaussian support for a bounded pilot."""
    values = np.asarray(opacities, np.float32)
    if values.ndim != 1 or maximum <= 0:
        raise ValueError("opacities must be one-dimensional and maximum positive")
    count = min(len(values), maximum)
    return np.sort(np.argsort(values, kind="stable")[-count:]).astype(np.int64)


def select_residual_gaussians(renderer: GsplatRenderer, observations: list[StaticObservation], maximum: int) -> np.ndarray:
    """Choose Gaussian parameters that actually explain static RGB residuals.

    Release PLY opacity is a poor proxy for which parameters produce the road
    stripes or view-specific tree/building errors.  Attribute RGB residual to
    DC coefficients per training view, allocate each view an equal candidate
    quota, and merge the normalized gradients.  This is a selection step only:
    no source image, mask, or PLY is changed here.
    """
    if not observations:
        raise ValueError("residual selection needs training observations")
    torch = renderer.torch
    count = int(renderer.means.shape[0])
    quota = min(count, max(1, int(math.ceil(maximum / len(observations)))))
    score = torch.zeros(count, device=renderer.means.device, dtype=torch.float32)
    for observation in observations:
        colors = renderer.colors.detach().clone().requires_grad_(True)
        rendered, alpha = _rasterize_rgb_alpha(
            renderer, observation.camera, renderer.means, renderer.quats, renderer.scales, renderer.opacities, colors,
        )
        target = torch.from_numpy(observation.target_rgb).to(rendered.device)
        valid = torch.from_numpy(observation.valid_mask).to(rendered.device)
        prediction = (rendered / alpha[..., None].clamp_min(.05)).clamp(0.0, 1.0)
        loss = torch.sqrt((prediction - target).square() + 1e-6).mean(dim=-1)[valid].mean()
        loss.backward()
        gradient = colors.grad[:, 0].norm(dim=-1)
        positive = int((gradient > 0.0).sum().item())
        selected = min(quota, positive)
        if selected:
            values, indices = torch.topk(gradient, selected, sorted=False)
            score.index_add_(0, indices, values / values.median().clamp_min(1e-12))
        print(
            f"Static residual attribution camera={observation.camera_id} ref={observation.reference_frame_index} "
            f"loss={float(loss):.6f} positive={positive:,} selected={selected:,}",
            flush=True,
        )
        del colors, rendered, alpha, target, valid, prediction, loss, gradient
    usable = int((score > 0.0).sum().item())
    if usable == 0:
        raise RuntimeError("residual attribution selected no trainable Gaussian")
    chosen = min(maximum, usable)
    _, indices = torch.topk(score, chosen, sorted=False)
    if chosen < maximum:
        # Preserve the requested budget without assigning a zero-residual
        # parameter twice; these fallback points are deliberately lowest risk.
        chosen_mask = torch.zeros(count, device=score.device, dtype=torch.bool)
        chosen_mask[indices] = True
        fallback = torch.where(~chosen_mask)[0]
        needed = min(maximum - chosen, len(fallback))
        _, top = torch.topk(renderer.opacities[fallback], needed, sorted=False)
        indices = torch.cat([indices, fallback[top]])
    return torch.sort(indices).values.detach().cpu().numpy().astype(np.int64)


def select_multiview_lidar_supported_gaussians(
    means: np.ndarray,
    candidate_indices: np.ndarray,
    observations: list[StaticObservation],
    *,
    radius_m: float,
    minimum_cameras: int,
    minimum_references: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Keep only residual candidates with independent image-visible LiDAR support.

    A LiDAR return counts only when it survived that observation's static
    image mask.  Camera and reference frame support are recorded separately,
    so repeated scans from one direction cannot authorise a geometry update.
    """
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("candidate_indices must be non-empty and one-dimensional")
    if radius_m <= 0.0 or minimum_cameras <= 0 or minimum_references <= 0:
        raise ValueError("invalid LiDAR support gate")
    positions = np.asarray(means, np.float32)[candidates]
    camera_ids = {value: index for index, value in enumerate(sorted({item.camera_id for item in observations}))}
    reference_ids = {value: index for index, value in enumerate(sorted({item.reference_frame_index for item in observations}))}
    if len(camera_ids) > 63 or len(reference_ids) > 63:
        raise ValueError("LiDAR support bitset supports at most 63 cameras/references")
    camera_bits = np.zeros(len(candidates), dtype=np.uint64)
    reference_bits = np.zeros(len(candidates), dtype=np.uint64)
    radius_squared = float(radius_m * radius_m)
    support_observations = 0
    for observation in observations:
        points = np.asarray(observation.lidar_world_m, np.float32)
        if not len(points):
            continue
        support_observations += 1
        nearby = np.zeros(len(candidates), dtype=bool)
        for start in range(0, len(candidates), 512):
            block = positions[start:start + 512]
            squared = ((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
            nearby[start:start + len(block)] = squared.min(axis=1) <= radius_squared
        camera_bits[nearby] |= np.uint64(1) << np.uint64(camera_ids[observation.camera_id])
        reference_bits[nearby] |= np.uint64(1) << np.uint64(reference_ids[observation.reference_frame_index])
    camera_count = sum(((camera_bits >> np.uint64(index)) & np.uint64(1)).astype(np.int16) for index in range(len(camera_ids)))
    reference_count = sum(((reference_bits >> np.uint64(index)) & np.uint64(1)).astype(np.int16) for index in range(len(reference_ids)))
    valid = (camera_count >= minimum_cameras) & (reference_count >= minimum_references)
    selected = candidates[valid]
    return selected, {
        "candidate_count": int(len(candidates)), "selected_count": int(len(selected)), "support_observations": int(support_observations),
        "radius_m": float(radius_m), "minimum_cameras": int(minimum_cameras), "minimum_references": int(minimum_references),
        "max_camera_support": int(camera_count.max(initial=0)), "max_reference_support": int(reference_count.max(initial=0)),
    }


def build_multiview_lidar_surface_seeds(
    residual_means: np.ndarray,
    observations: list[StaticObservation],
    *,
    voxel_size_m: float,
    residual_support_radius_m: float,
    minimum_cameras: int,
    minimum_references: int,
    maximum: int,
) -> StaticSurfaceSeeds:
    """Construct conservative new surface seeds from independently observed LiDAR.

    The function deliberately never projects arbitrary RGB pixels into 3-D.
    A voxel must have raw LiDAR support from independent cameras and times,
    and must be close to a residual-attributed existing Gaussian.  The latter
    prevents densifying a perfectly explained road/sky region merely because
    it is richly sampled by the lidar.
    """
    residual = np.asarray(residual_means, np.float32)
    if residual.ndim != 2 or residual.shape[1] != 3 or not len(residual):
        raise ValueError("residual_means must be non-empty [N,3]")
    if (voxel_size_m <= 0.0 or residual_support_radius_m <= 0.0 or minimum_cameras <= 0
            or minimum_references <= 0 or maximum <= 0):
        raise ValueError("surface seed parameters are invalid")
    camera_ids = {value: index for index, value in enumerate(sorted({item.camera_id for item in observations}))}
    reference_ids = {value: index for index, value in enumerate(sorted({item.reference_frame_index for item in observations}))}
    if len(camera_ids) > 63 or len(reference_ids) > 63:
        raise ValueError("surface seed bitset supports at most 63 cameras/references")
    buckets: dict[tuple[int, int, int], dict[str, Any]] = {}
    input_returns = 0
    for observation in observations:
        world = np.asarray(observation.lidar_world_m, np.float32)
        xy = np.asarray(observation.lidar_xy, np.int64)
        if not len(world):
            continue
        if xy.shape != (len(world), 2):
            raise ValueError("LiDAR world/pixel samples have inconsistent shapes")
        colours = observation.target_rgb[xy[:, 1], xy[:, 0]]
        input_returns += len(world)
        voxels = np.floor(world / voxel_size_m).astype(np.int64)
        for point, colour, voxel in zip(world, colours, voxels, strict=True):
            key = tuple(int(value) for value in voxel)
            bucket = buckets.setdefault(key, {
                "points": [], "colours": [], "camera_bits": np.uint64(0), "reference_bits": np.uint64(0),
            })
            bucket["points"].append(point)
            bucket["colours"].append(colour)
            bucket["camera_bits"] |= np.uint64(1) << np.uint64(camera_ids[observation.camera_id])
            bucket["reference_bits"] |= np.uint64(1) << np.uint64(reference_ids[observation.reference_frame_index])
    candidates: list[tuple[np.ndarray, np.ndarray, int, int, int]] = []
    for bucket in buckets.values():
        cameras = int(int(bucket["camera_bits"]).bit_count())
        references = int(int(bucket["reference_bits"]).bit_count())
        if cameras < minimum_cameras or references < minimum_references:
            continue
        point = np.mean(np.asarray(bucket["points"], np.float32), axis=0)
        # Small blocks avoid a large [voxel,residual] allocation when the
        # caller uses every valid seven-camera observation.
        squared = np.full((), np.inf, np.float32)
        for start in range(0, len(residual), 4096):
            squared = min(squared, np.min(((residual[start:start + 4096] - point) ** 2).sum(axis=1)))
        if float(squared) > residual_support_radius_m * residual_support_radius_m:
            continue
        colour = np.median(np.asarray(bucket["colours"], np.float32), axis=0)
        candidates.append((point, colour, len(bucket["points"]), cameras, references))
    # More independent support wins; a deterministic lexicographic tie break
    # keeps repeated pilots reproducible.
    candidates.sort(key=lambda value: (-value[3], -value[4], -value[2], *value[0].tolist()))
    candidates = candidates[:maximum]
    if candidates:
        means = np.stack([value[0] for value in candidates]).astype(np.float32)
        rgb = np.stack([value[1] for value in candidates]).astype(np.float32)
        support = np.asarray([value[2] for value in candidates], np.uint16)
    else:
        means = np.empty((0, 3), np.float32); rgb = np.empty((0, 3), np.float32); support = np.empty(0, np.uint16)
    return StaticSurfaceSeeds(
        means=means, rgb=rgb, support_count=support,
        report={
            "input_lidar_returns": int(input_returns), "voxel_count": int(len(buckets)), "selected_count": int(len(means)),
            "voxel_size_m": float(voxel_size_m), "residual_support_radius_m": float(residual_support_radius_m),
            "minimum_cameras": int(minimum_cameras), "minimum_references": int(minimum_references),
        },
    )


def _parameter_scene(renderer: GsplatRenderer, indices: Any, position_delta: Any, log_scale_delta: Any,
                     rgb_delta: Any, opacity_delta: Any, bounds: StaticSceneOptimizeOptions,
                     *, seed_base: tuple[Any, Any, Any, Any, Any] | None = None,
                     seed_position_delta: Any | None = None, seed_log_scale_delta: Any | None = None,
                     seed_opacity_delta: Any | None = None) -> tuple[Any, Any, Any, Any, Any]:
    torch = renderer.torch
    base_rgb = renderer.colors[:, 0, :] * _C0 + .5
    means = renderer.means.index_copy(0, indices, renderer.means[indices] + bounds.max_position_delta_m * position_delta.tanh())
    quats = renderer.quats
    scales = renderer.scales.index_copy(0, indices, renderer.scales[indices] * torch.exp(bounds.max_log_scale_delta * log_scale_delta.tanh()))
    rgb = base_rgb.index_copy(0, indices, (base_rgb[indices] + bounds.max_rgb_delta * rgb_delta.tanh()).clamp(0.0, 1.0))
    logits = torch.logit(renderer.opacities.clamp(1e-5, 1 - 1e-5))
    opacities = torch.sigmoid(logits.index_copy(0, indices, logits[indices] + bounds.max_logit_opacity_delta * opacity_delta.tanh()))
    colors = ((rgb - .5) / _C0)[:, None, :]
    if seed_base is not None:
        if seed_position_delta is None or seed_log_scale_delta is None or seed_opacity_delta is None:
            raise ValueError("seed deltas are required with seed_base")
        seed_means, seed_quats, seed_scales, seed_opacities, seed_colors = seed_base
        seed_means = seed_means + bounds.max_position_delta_m * seed_position_delta.tanh()
        seed_scales = seed_scales * torch.exp(bounds.max_log_scale_delta * seed_log_scale_delta.tanh())
        seed_logits = torch.logit(seed_opacities.clamp(1e-5, 1 - 1e-5))
        seed_opacities = torch.sigmoid(seed_logits + bounds.max_logit_opacity_delta * seed_opacity_delta.tanh())
        means = torch.cat((means, seed_means), dim=0)
        quats = torch.cat((quats, seed_quats), dim=0)
        scales = torch.cat((scales, seed_scales), dim=0)
        opacities = torch.cat((opacities, seed_opacities), dim=0)
        colors = torch.cat((colors, seed_colors), dim=0)
    return means, quats, scales, opacities, colors


def _observation_loss(renderer: GsplatRenderer, observation: StaticObservation, parameters: tuple[Any, Any, Any, Any, Any],
                      options: StaticSceneOptimizeOptions, *, include_lidar_depth: bool = True) -> tuple[Any, Any, Any]:
    torch = renderer.torch
    means, quats, scales, opacities, colors = parameters
    rendered, alpha = _rasterize_rgb_alpha(renderer, observation.camera, means, quats, scales, opacities, colors)
    target = torch.from_numpy(observation.target_rgb).to(rendered.device)
    valid = torch.from_numpy(observation.valid_mask).to(rendered.device)
    rgb = (rendered / alpha[..., None].clamp_min(.05)).clamp(0.0, 1.0)
    rgb_loss = torch.sqrt((rgb - target).square() + 1e-6).mean(dim=-1)[valid].mean()
    depth_loss = rgb_loss.new_zeros(())
    if include_lidar_depth and len(observation.lidar_xy):
        expected_depth, _ = _world_depth(renderer, observation.camera, means, quats, scales, opacities)
        xy = torch.from_numpy(observation.lidar_xy).to(rendered.device)
        lidar = torch.from_numpy(observation.lidar_depth_m).to(rendered.device)
        predicted = expected_depth[xy[:, 1], xy[:, 0]]
        finite = torch.isfinite(predicted) & torch.isfinite(lidar)
        if bool(finite.any()):
            depth_loss = torch.sqrt((predicted[finite] - lidar[finite]).square() + .01).mean()
    return rgb_loss, depth_loss, alpha


def _evaluate(observations: list[StaticObservation], renderer: GsplatRenderer, parameters: tuple[Any, Any, Any, Any, Any],
              options: StaticSceneOptimizeOptions) -> dict[str, float]:
    torch = renderer.torch
    rgb_values: list[float] = []; depth_values: list[float] = []
    with torch.no_grad():
        for observation in observations:
            rgb, depth, _ = _observation_loss(renderer, observation, parameters, options)
            rgb_values.append(float(rgb)); depth_values.append(float(depth))
    return {"rgb_mae": float(np.mean(rgb_values)), "lidar_depth_mae_m": float(np.mean(depth_values)), "observation_count": float(len(observations))}


def _write_ply(scene: GaussianScene, means: np.ndarray, scales: np.ndarray, quats: np.ndarray, opacities: np.ndarray,
               colors: np.ndarray, path: Path) -> None:
    """Write a DC-only Graphdeco-compatible PLY; source release assets are DC-only."""
    if scene.sh_degree != 0:
        raise ValueError("static pilot currently supports DC-only PLYs")
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plyfile is required") from exc
    dtype = [(name, "f4") for name in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    value = np.empty(len(means), dtype=dtype)
    value["x"], value["y"], value["z"] = means[:, 0], means[:, 1], means[:, 2]
    dc = (colors - .5) / _C0
    value["f_dc_0"], value["f_dc_1"], value["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    value["opacity"] = np.log(np.clip(opacities, 1e-5, 1 - 1e-5) / np.clip(1 - opacities, 1e-5, 1 - 1e-5))
    value["scale_0"], value["scale_1"], value["scale_2"] = np.log(scales[:, 0]), np.log(scales[:, 1]), np.log(scales[:, 2])
    value["rot_0"], value["rot_1"], value["rot_2"], value["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".ply", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        PlyData([PlyElement.describe(value, "vertex")], text=False).write(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def optimize_static_scene(options: StaticSceneOptimizeOptions) -> dict[str, Any]:
    """Run a bounded native-FTheta static optimisation pilot and write a replacement PLY."""
    import torch

    output_path = options.color_correction_output if options.optimization_mode == "color_only" else options.output_ply
    if output_path is None:  # guarded by StaticSceneOptimizeOptions, retained for type narrowing
        raise ValueError("no output path")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    report_path = output_path.with_suffix(".report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    torch.manual_seed(options.seed); np.random.seed(options.seed)
    scene = load_gaussian_ply(options.initial_ply)
    if options.initial_color_correction is not None:
        correction = GaussianColorCorrection.load(options.initial_color_correction)
        correction.validate_source(options.initial_ply, scene.count)
        scene = correction.apply_to_scene(scene)
    if scene.sh_degree != 0:
        raise ValueError("static pilot currently supports DC-only input PLYs")
    observations = load_static_observations(options)
    train = [item for item in observations if item.split == "train"]
    validation = [item for item in observations if item.split == "validation"]
    renderer = GsplatRenderer(scene, RenderOptions(device=options.device, rasterize_mode="antialiased"))
    device = renderer.means.device
    selected_np = (
        select_trainable_gaussians(scene.opacities, options.trainable_gaussians)
        if options.selection_mode == "opacity"
        else select_residual_gaussians(renderer, train, options.trainable_gaussians)
    )
    lidar_support: dict[str, int | float] | None = None
    if options.optimization_mode == "geometry_lidar":
        selected_np, lidar_support = select_multiview_lidar_supported_gaussians(
            scene.means, selected_np, train, radius_m=options.lidar_support_radius_m,
            minimum_cameras=options.minimum_lidar_cameras, minimum_references=options.minimum_lidar_references,
        )
        if not len(selected_np):
            raise RuntimeError("no residual Gaussian passed the multi-view LiDAR geometry gate")
    surface_seeds: StaticSurfaceSeeds | None = None
    if options.optimization_mode == "geometry_lidar_densify":
        selected_np, lidar_support = select_multiview_lidar_supported_gaussians(
            scene.means, selected_np, train, radius_m=options.lidar_support_radius_m,
            minimum_cameras=options.minimum_lidar_cameras, minimum_references=options.minimum_lidar_references,
        )
        if not len(selected_np):
            raise RuntimeError("no residual Gaussian passed the multi-view LiDAR geometry gate")
        surface_seeds = build_multiview_lidar_surface_seeds(
            scene.means[selected_np], train, voxel_size_m=options.seed_voxel_size_m,
            residual_support_radius_m=options.seed_residual_support_radius_m,
            minimum_cameras=options.minimum_lidar_cameras, minimum_references=options.minimum_lidar_references,
            maximum=options.maximum_seed_gaussians,
        )
        if len(surface_seeds.means) < options.minimum_seed_gaussians:
            raise RuntimeError(
                f"only {len(surface_seeds.means)} multi-view LiDAR surface seeds passed the gate; "
                f"need at least {options.minimum_seed_gaussians}"
            )
    print(
        f"Static Gaussian selection mode={options.selection_mode} selected={len(selected_np):,}/{scene.count:,}",
        flush=True,
    )
    selected = torch.from_numpy(selected_np).to(device)
    geometry_trainable = options.optimization_mode != "color_only"
    color_trainable = options.optimization_mode not in {"geometry_lidar", "geometry_lidar_densify"}
    position_delta = torch.zeros((len(selected), 3), device=device, requires_grad=geometry_trainable)
    log_scale_delta = torch.zeros_like(position_delta, requires_grad=geometry_trainable)
    rgb_delta = torch.zeros_like(position_delta, requires_grad=color_trainable)
    opacity_delta = torch.zeros(len(selected), device=device, requires_grad=geometry_trainable)
    seed_base: tuple[Any, Any, Any, Any, Any] | None = None
    seed_position_delta: Any | None = None
    seed_log_scale_delta: Any | None = None
    seed_opacity_delta: Any | None = None
    if surface_seeds is not None:
        count = len(surface_seeds.means)
        seed_base = (
            torch.from_numpy(surface_seeds.means).to(device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(count, 1),
            torch.full((count, 3), options.seed_scale_m, dtype=torch.float32, device=device),
            torch.full((count,), .10, dtype=torch.float32, device=device),
            torch.from_numpy(((surface_seeds.rgb - .5) / _C0)[:, None, :]).to(device),
        )
        seed_position_delta = torch.zeros((count, 3), device=device, requires_grad=True)
        seed_log_scale_delta = torch.zeros_like(seed_position_delta, requires_grad=True)
        seed_opacity_delta = torch.zeros(count, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [value for value in (position_delta, log_scale_delta, rgb_delta, opacity_delta,
                             seed_position_delta, seed_log_scale_delta, seed_opacity_delta)
         if value is not None and value.requires_grad],
        lr=options.learning_rate,
    )
    def current_parameters(*, include_seeds: bool) -> tuple[Any, Any, Any, Any, Any]:
        return _parameter_scene(
            renderer, selected, position_delta, log_scale_delta, rgb_delta, opacity_delta, options,
            seed_base=seed_base if include_seeds else None,
            seed_position_delta=seed_position_delta if include_seeds else None,
            seed_log_scale_delta=seed_log_scale_delta if include_seeds else None,
            seed_opacity_delta=seed_opacity_delta if include_seeds else None,
        )
    # Preserve the exact input scene as a valid early-stop candidate.  A short
    # pilot can otherwise write a worse PLY merely because its first update is
    # the only recorded validation point.
    initial_parameters = current_parameters(include_seeds=False)
    initial_validation = _evaluate(validation, renderer, initial_parameters, options)
    best: tuple[float, tuple[Any, ...], bool] = (
        initial_validation["rgb_mae"],
        tuple(value.detach().clone() for value in (position_delta, log_scale_delta, rgb_delta, opacity_delta,
                                                   seed_position_delta, seed_log_scale_delta, seed_opacity_delta)
              if value is not None),
        False,
    )
    history: list[dict[str, float]] = [{"step": 0.0, "loss": float("nan"), "rgb": float("nan"), "lidar": float("nan"), **initial_validation}]
    for step in range(options.iterations):
        parameters = current_parameters(include_seeds=surface_seeds is not None)
        observation = train[step % len(train)]
        rgb_loss, depth_loss, _ = _observation_loss(
            renderer, observation, parameters, options, include_lidar_depth=geometry_trainable,
        )
        prior = options.color_prior_weight * rgb_delta.square().mean()
        if geometry_trainable:
            prior = (prior + options.position_prior_weight * position_delta.square().mean()
                     + options.scale_prior_weight * log_scale_delta.square().mean()
                     + options.opacity_prior_weight * opacity_delta.square().mean())
        loss = options.rgb_weight * rgb_loss + options.lidar_depth_weight * depth_loss + prior
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 0 or (step + 1) % options.validation_every == 0 or step + 1 == options.iterations:
            evaluated = current_parameters(include_seeds=surface_seeds is not None)
            validation_metrics = _evaluate(validation, renderer, evaluated, options)
            history.append({"step": float(step + 1), "loss": float(loss.detach()), "rgb": float(rgb_loss.detach()), "lidar": float(depth_loss.detach()), **validation_metrics})
            if validation_metrics["rgb_mae"] < best[0]:
                best = (
                    validation_metrics["rgb_mae"],
                    tuple(value.detach().clone() for value in (position_delta, log_scale_delta, rgb_delta, opacity_delta,
                                                               seed_position_delta, seed_log_scale_delta, seed_opacity_delta)
                          if value is not None),
                    surface_seeds is not None,
                )
            print(f"Static FTheta optimise [{step + 1:05d}/{options.iterations}] loss={float(loss):.6f} rgb={float(rgb_loss):.6f} lidar={float(depth_loss):.6f} val={validation_metrics['rgb_mae']:.6f}", flush=True)
    stored = iter(best[1])
    position_delta.data.copy_(next(stored)); log_scale_delta.data.copy_(next(stored)); rgb_delta.data.copy_(next(stored)); opacity_delta.data.copy_(next(stored))
    if seed_position_delta is not None and seed_log_scale_delta is not None and seed_opacity_delta is not None:
        seed_position_delta.data.copy_(next(stored)); seed_log_scale_delta.data.copy_(next(stored)); seed_opacity_delta.data.copy_(next(stored))
    means, quats, scales, opacities, colors_dc = current_parameters(include_seeds=best[2])
    means_np, scales_np, opacity_np = (value.detach().cpu().numpy().astype(np.float32) for value in (means, scales, opacities))
    rgb_np = (colors_dc[:, 0, :].detach().cpu().numpy() * _C0 + .5).astype(np.float32)
    correction_count = 0
    if options.optimization_mode == "color_only":
        if options.color_correction_output is None:  # guarded by options validation
            raise ValueError("color_only mode requires a correction output")
        corrected_dc = colors_dc[selected, 0, :].detach().cpu().numpy().astype(np.float32)
        original_dc = scene.sh_coeffs[selected_np, 0].astype(np.float32, copy=True)
        change = np.linalg.norm(corrected_dc - original_dc, axis=1)
        correction = GaussianColorCorrection(
            indices=selected_np.astype(np.int64, copy=False), original_f_dc=original_dc, corrected_f_dc=corrected_dc,
            confidence=(change / max(float(change.max(initial=0.0)), 1e-12)).astype(np.float32),
            support_count=np.ones(len(selected_np), dtype=np.uint16), ply_sha256=file_sha256(options.initial_ply),
            gaussian_count=scene.count,
            metadata={"schema": SCHEMA, "version": VERSION, "optimization_mode": "color_only", "selection_mode": options.selection_mode},
        )
        correction.save(options.color_correction_output)
        correction_count = correction.count
    else:
        if options.output_ply is None:  # guarded by options validation
            raise ValueError("geometry optimisation requires output_ply")
        _write_ply(scene, means_np, scales_np, quats.detach().cpu().numpy().astype(np.float32), opacity_np, rgb_np, options.output_ply)
    final_parameters = current_parameters(include_seeds=best[2])
    report: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "initial_ply": str(options.initial_ply.resolve()),
        "initial_ply_sha256": _sha256(options.initial_ply),
        "initial_color_correction": None if options.initial_color_correction is None else str(options.initial_color_correction.resolve()),
        "output_ply": None if options.output_ply is None else str(options.output_ply.resolve()),
        "color_correction_output": None if options.color_correction_output is None else str(options.color_correction_output.resolve()),
        "color_correction_count": correction_count,
        "dynamic_mask_manifest": str(options.dynamic_mask_manifest.resolve()), "dynamic_mask_manifest_sha256": _sha256(options.dynamic_mask_manifest),
        "segformer_model_dir": str(options.segformer_model_dir.resolve()), "camera_ids": list(options.camera_ids),
        "train_reference_indices": list(options.train_reference_indices), "validation_reference_indices": list(options.validation_reference_indices),
        "resolution": [options.width, options.height], "selection_mode": options.selection_mode,
        "optimization_mode": options.optimization_mode, "lidar_support": lidar_support,
        "surface_seeds": None if surface_seeds is None else surface_seeds.report,
        "best_includes_surface_seeds": bool(best[2]),
        "trainable_gaussians": int(len(selected)), "total_gaussians": scene.count,
        "mask_fractions": [{"camera_id": item.camera_id, "reference_frame_index": item.reference_frame_index, "split": item.split, **item.mask_fractions} for item in observations],
        "train": _evaluate(train, renderer, final_parameters, options), "validation": _evaluate(validation, renderer, final_parameters, options),
        "initial_validation": initial_validation, "best_validation_rgb_mae": best[0], "history": history,
        "geometry_trainable": geometry_trainable,
        "geometry_unchanged_by_optimization": not geometry_trainable,
        "limitations": [
            "Pilot optimises a fixed selected subset; it does not densify, prune or add unobserved geometry.",
            "Sky, ego and original-source dynamic masks are excluded from RGB/LiDAR supervision.",
            "A successful smoke is not authorisation for a long run, sequence render or L4 production.",
        ],
    }
    _write_json_atomic(report, report_path)
    label = "color correction" if options.optimization_mode == "color_only" else "PLY"
    print(f"Wrote native FTheta static pilot {label}: {output_path} (best_validation_rgb={best[0]:.6f})", flush=True)
    return report
