"""Object-centric canonical Gaussian optimisation in native NCore camera space.

Unlike the Street Gaussians bridge, this experiment never rectifies source
images into a pinhole proxy.  Every training call uses the original NCore
FTheta camera and its start/end rolling-shutter poses through gsplat's
``with_eval3d`` path.  The InstantNuRec static PLY is frozen; only a compact,
explicit subset of canonical rigid-actor Gaussians is optimised.

This is deliberately conservative.  It does not claim that a cuboid is a
pixel-perfect instance annotation.  RGB supervision is restricted to
``cuboid ∩ high-confidence vehicle semantics`` and three regularizers prevent
the actor layer from absorbing the frozen background:

* alpha outside a dilated vehicle ROI;
* alpha behind a frozen static foreground according to initial expected depth;
* residual static contribution inside a visible vehicle ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask, vehicle_roi
from .canonical_actor import (
    CANONICAL_SCHEMA,
    CANONICAL_VERSION,
    ActorTrajectory,
    CanonicalActorAsset,
    _rotation_matrix_from_wxyz,
    _wxyz_from_rotation_matrix,
)
from .camera import compose_ncore_camera_pose
from .ply_io import GaussianScene, load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path, _ncore_exposure_timestamps
from .render import GsplatRenderer, RenderOptions
from .sky_build import dilate_boolean_mask
from .static_leakage import _project_lidar_depth_ftheta
from .street_dataset import SegformerVehicleSegmenter, _nearest_index


DIRECT_DEFAULT_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
)


@dataclass(frozen=True)
class FThetaActorOptimizeOptions:
    static_ply: Path
    initial_actors: Path
    actor_dataset_manifest: Path
    ncore_path: Path
    segformer_model_dir: Path
    output: Path
    frame_indices: tuple[int, ...] = ()
    camera_ids: tuple[str, ...] = DIRECT_DEFAULT_CAMERAS
    validation_frame_indices: tuple[int, ...] = ()
    device: str = "cuda"
    width: int = 480
    height: int = 270
    max_gaussians_per_actor: int = 12_000
    iterations: int = 3_000
    learning_rate: float = 2.0e-3
    max_position_delta_m: float = 0.20
    max_color_delta: float = 0.25
    roi_probability_threshold: float = 0.60
    roi_margin_threshold: float = 0.05
    cuboid_margin_pixels: int = 8
    roi_dilation_pixels: int = 8
    static_occluder_depth_margin_m: float = 0.30
    clean_static_mask_dir: Path | None = None
    clean_static_mask_slots: tuple[int, ...] = ()
    validation_clean_static_mask_slots: tuple[int, ...] = ()
    min_source_mask_coverage: float = 0.0
    min_lidar_pixels: int = 0
    max_lidar_timestamp_delta_us: int = 100_000
    camera_appearance_enabled: bool = False
    camera_appearance_max_delta: float = 0.12
    camera_appearance_bias_prior_weight: float = 0.01
    camera_appearance_slope_prior_weight: float = 0.02
    rgb_weight: float = 1.0
    outside_alpha_weight: float = 0.10
    occluded_alpha_weight: float = 0.10
    static_leak_weight: float = 0.05
    position_prior_weight: float = 0.02
    color_prior_weight: float = 0.01
    opacity_prior_weight: float = 0.002
    checkpoint_every: int = 250
    resume_checkpoint: Path | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output.suffix.lower() != ".npz":
            raise ValueError("output must use the .npz extension")
        if (not self.frame_indices and not self.clean_static_mask_slots) or not self.camera_ids:
            raise ValueError("at least one reference frame or clean-static slot and camera are required")
        if self.frame_indices and self.clean_static_mask_slots:
            raise ValueError("choose reference frame indices or exact clean-static mask slots, not both")
        if self.validation_frame_indices and self.validation_clean_static_mask_slots:
            raise ValueError("choose validation reference frames or validation clean-static mask slots, not both")
        if len(set((*self.frame_indices, *self.validation_frame_indices))) != len(self.frame_indices) + len(self.validation_frame_indices):
            raise ValueError("training and validation frame indices must not overlap or repeat")
        if len(set((*self.clean_static_mask_slots, *self.validation_clean_static_mask_slots))) != len(self.clean_static_mask_slots) + len(self.validation_clean_static_mask_slots):
            raise ValueError("training and validation clean-static mask slots must not overlap or repeat")
        if (self.clean_static_mask_slots or self.validation_clean_static_mask_slots) and self.clean_static_mask_dir is None:
            raise ValueError("clean-static mask slots require clean_static_mask_dir")
        if self.width <= 0 or self.height <= 0 or self.max_gaussians_per_actor <= 0:
            raise ValueError("resolution and Gaussian budget must be positive")
        if self.iterations < 0 or self.learning_rate <= 0.0:
            raise ValueError("iterations and learning rate are invalid")
        if self.max_position_delta_m <= 0.0 or not 0.0 < self.max_color_delta <= 1.0:
            raise ValueError("position/color bounds are invalid")
        if self.cuboid_margin_pixels < 0 or self.roi_dilation_pixels < 0:
            raise ValueError("ROI margins must be non-negative")
        if self.static_occluder_depth_margin_m < 0.0:
            raise ValueError("static occluder depth margin must be non-negative")
        if not 0.0 <= self.min_source_mask_coverage <= 1.0:
            raise ValueError("min_source_mask_coverage must be in [0,1]")
        if self.min_lidar_pixels < 0 or self.max_lidar_timestamp_delta_us <= 0:
            raise ValueError("LiDAR visibility gate parameters are invalid")
        if not 0.0 < self.camera_appearance_max_delta <= 1.0:
            raise ValueError("camera_appearance_max_delta must be in (0,1]")
        if self.camera_appearance_bias_prior_weight < 0.0 or self.camera_appearance_slope_prior_weight < 0.0:
            raise ValueError("camera appearance prior weights must be non-negative")
        if self.min_source_mask_coverage > 0.0 and self.clean_static_mask_dir is None:
            raise ValueError("min_source_mask_coverage requires clean_static_mask_dir")
        if self.camera_appearance_enabled and self.resume_checkpoint is not None:
            raise ValueError("camera appearance diagnostic does not support checkpoint resume")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if any(value < 0.0 for value in (
            self.rgb_weight, self.outside_alpha_weight, self.occluded_alpha_weight,
            self.static_leak_weight, self.position_prior_weight, self.color_prior_weight,
            self.opacity_prior_weight,
        )):
            raise ValueError("loss weights must be non-negative")


@dataclass
class DirectObservation:
    camera_id: str
    reference_frame_index: int
    source_frame_index: int
    camera: Any
    target_rgb: np.ndarray
    static_rgb: np.ndarray
    vehicle_roi: np.ndarray
    dilated_vehicle_roi: np.ndarray
    visible_roi: np.ndarray
    static_alpha: np.ndarray
    static_depth: np.ndarray
    actor_initial_depth: np.ndarray
    clean_static_mask_coverage: float | None
    lidar_visible_pixels: int
    timestamp_midpoint_us: int


def _load_tracks(path: Path, actor_ids: set[int]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    all_tracks = manifest.get("tracks")
    if not isinstance(all_tracks, list):
        raise ValueError(f"actor dataset manifest lacks tracks: {path}")
    tracks = [track for track in all_tracks if int(track.get("track_id", -1)) in actor_ids]
    if not tracks:
        raise ValueError("none of the canonical actor IDs occur in the frozen dataset manifest")
    return tracks


def _subset_asset(asset: CanonicalActorAsset, maximum_per_actor: int) -> CanonicalActorAsset:
    """Keep the strongest compact actor set for tractable exact-camera training."""
    selected: list[np.ndarray] = []
    for actor_index in range(len(asset.actor_ids)):
        members = np.flatnonzero(asset.actor_indices == actor_index)
        # The Street bridge has visibility/source counts even for regularized
        # actors.  Opacity is primary, while those fields make ties stable and
        # favour genuinely multi-view inputs when available.
        score = (
            asset.opacities[members].astype(np.float64)
            + 1.0e-4 * asset.visibility_counts[members]
            + 1.0e-6 * asset.source_counts[members]
        )
        keep = members[np.argsort(score)[::-1][:maximum_per_actor]]
        selected.append(np.sort(keep))
    indices = np.concatenate(selected)
    metadata = dict(asset.metadata)
    metadata.update({
        "optimizer_status": "direct_ftheta_rolling_shutter_initialization",
        "source_gaussian_count": asset.count,
        "max_gaussians_per_actor": maximum_per_actor,
    })
    subset = CanonicalActorAsset(
        positions=asset.positions[indices], rotations_wxyz=asset.rotations_wxyz[indices],
        scales=asset.scales[indices], rgb=asset.rgb[indices], opacities=asset.opacities[indices],
        actor_indices=asset.actor_indices[indices], visibility_counts=asset.visibility_counts[indices],
        source_counts=asset.source_counts[indices], actor_ids=asset.actor_ids,
        trajectories=asset.trajectories, metadata=metadata,
    )
    subset.validate()
    return subset


def _clean_static_mask_lookup(path: Path | None) -> dict[tuple[str, int], Path]:
    """Load exact source-image masks keyed by NCore camera/frame.

    The gate intentionally does not resample mask slots: a source image is
    trusted only when that exact camera frame was an input to clean-static
    reconstruction and its stored mask is present.
    """
    if path is None:
        return {}
    manifest_path = path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "nurec-clean-static-mask" or int(manifest.get("version", -1)) != 1:
        raise ValueError(f"unsupported clean-static mask manifest: {manifest_path}")
    entries: dict[tuple[str, int], Path] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError(f"malformed clean-static mask entry: {manifest_path}")
        camera_id, frame_index, relative = entry.get("camera_id"), entry.get("frame_index"), entry.get("path")
        if not isinstance(camera_id, str) or not isinstance(frame_index, int) or not isinstance(relative, str):
            raise ValueError(f"malformed clean-static mask entry: {manifest_path}")
        target = path / relative
        if not target.is_file():
            raise FileNotFoundError(f"clean-static mask listed but missing: {target}")
        entries[(camera_id, frame_index)] = target
    if not entries:
        raise ValueError(f"clean-static mask manifest has no entries: {manifest_path}")
    return entries


def _clean_static_slot_frames(path: Path | None) -> dict[tuple[str, int], int]:
    """Return exact NCore frame indices keyed by (camera_id, manifest slot)."""
    if path is None:
        return {}
    manifest_path = path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "nurec-clean-static-mask" or int(manifest.get("version", -1)) != 1:
        raise ValueError(f"unsupported clean-static mask manifest: {manifest_path}")
    entries: dict[tuple[str, int], int] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError(f"malformed clean-static mask entry: {manifest_path}")
        camera_id, slot, frame_index = entry.get("camera_id"), entry.get("slot"), entry.get("frame_index")
        if not isinstance(camera_id, str) or not isinstance(slot, int) or not isinstance(frame_index, int):
            raise ValueError(f"malformed clean-static mask slot fields: {manifest_path}")
        entries[(camera_id, slot)] = frame_index
    return entries


def visibility_gate_mask(
    roi: np.ndarray,
    source_mask: np.ndarray,
    static_alpha: np.ndarray,
    static_depth: np.ndarray,
    actor_depth: np.ndarray,
    *,
    depth_margin_m: float,
) -> np.ndarray:
    """Keep source-excluded semantic pixels not proven static-foreground occluded."""
    roi = np.asarray(roi, dtype=bool)
    source_mask = np.asarray(source_mask, dtype=bool)
    static_alpha = np.asarray(static_alpha, dtype=np.float32)
    static_depth = np.asarray(static_depth, dtype=np.float32)
    actor_depth = np.asarray(actor_depth, dtype=np.float32)
    if any(value.shape != roi.shape for value in (source_mask, static_alpha, static_depth, actor_depth)):
        raise ValueError("visibility gate inputs must share one HxW shape")
    static_front = (
        (static_alpha >= 0.02) & np.isfinite(static_depth) & np.isfinite(actor_depth)
        & (static_depth < actor_depth - depth_margin_m)
    )
    return roi & source_mask & ~static_front


def _camera_kwargs(renderer: GsplatRenderer, camera: Any) -> tuple[dict[str, object], bool]:
    torch = renderer.torch
    kwargs: dict[str, object] = {"camera_model": camera.camera_model}
    packed = renderer.options.packed
    if camera.ftheta is not None:
        ftheta = camera.ftheta
        kwargs.update(
            with_ut=True,
            ftheta_coeffs=renderer.FThetaCameraDistortionParameters(
                reference_poly=renderer.FThetaPolynomialType[ftheta.reference_poly],
                pixeldist_to_angle_poly=ftheta.pixeldist_to_angle_poly,
                angle_to_pixeldist_poly=ftheta.angle_to_pixeldist_poly,
                max_angle=ftheta.max_angle,
                linear_cde=ftheta.linear_cde,
            ),
            rolling_shutter=renderer.RollingShutterType[ftheta.shutter_type],
        )
        packed = False
        if ftheta.is_rolling:
            kwargs["viewmats_rs"] = torch.from_numpy(camera.world_to_camera_end).to(renderer.means.device)[None]
    return kwargs, packed


def _rasterize_rgb_alpha(
    renderer: GsplatRenderer,
    camera: Any,
    means: Any,
    quats: Any,
    scales: Any,
    opacities: Any,
    colors: Any,
) -> tuple[Any, Any]:
    """Differentiable native-camera RGB/alpha rasterization.

    ``with_eval3d`` is intentionally forced for rolling FTheta cameras.  This
    matches the production renderer's validated path and avoids the known 2D
    rolling-shutter footprint failure in gsplat 1.5.3.
    """
    torch = renderer.torch
    kwargs, packed = _camera_kwargs(renderer, camera)
    intrinsics = camera.intrinsics
    common = dict(
        means=means, quats=quats, scales=scales, opacities=opacities,
        viewmats=torch.from_numpy(camera.world_to_camera).to(renderer.means.device)[None],
        Ks=torch.from_numpy(intrinsics.K).to(renderer.means.device)[None],
        width=intrinsics.width, height=intrinsics.height,
        near_plane=renderer.options.near_plane, far_plane=renderer.options.far_plane,
        packed=packed, backgrounds=None, rasterize_mode=renderer.options.rasterize_mode,
    )
    extra = {"with_eval3d": True} if camera.ftheta is not None and camera.ftheta.is_rolling else {}
    rendered, alpha, _ = renderer.rasterization(
        colors=colors, sh_degree=renderer.sh_degree, render_mode="RGB", **extra, **common, **kwargs
    )
    return rendered[0, ..., :3], alpha[0, ..., 0].clamp(0.0, 1.0)


def _world_actor_parameters(asset: CanonicalActorAsset, camera: Any, *, device: Any) -> tuple[np.ndarray, np.ndarray]:
    """World rotations/quaternions for a camera exposure midpoint.

    The trainable values stay actor-local.  Cuboid trajectories are fixed in
    v1, so quaternion transforms are a deterministic, non-gradient part of
    the per-view setup while local positions retain gradients.
    """
    timestamp = int(camera.timestamp_start_us + (camera.timestamp_us - camera.timestamp_start_us) // 2) if camera.timestamp_start_us is not None else int(camera.timestamp_us)
    quaternions = np.empty_like(asset.rotations_wxyz)
    rotations = np.empty((len(asset.actor_ids), 3, 3), dtype=np.float32)
    for actor_index, trajectory in enumerate(asset.trajectories):
        times = trajectory.timestamps_us
        right = min(max(int(np.searchsorted(times, timestamp, side="right")), 1), len(times) - 1)
        left = right - 1
        fraction = np.clip((timestamp - int(times[left])) / max(int(times[right] - times[left]), 1), 0.0, 1.0)
        blended = (1.0 - fraction) * trajectory.actor_to_world[left, :3, :3] + fraction * trajectory.actor_to_world[right, :3, :3]
        u, _, vh = np.linalg.svd(blended)
        rotation = u @ vh
        rotations[actor_index] = rotation
        members = asset.actor_indices == actor_index
        for index in np.flatnonzero(members):
            quaternions[index] = _wxyz_from_rotation_matrix(rotation @ _rotation_matrix_from_wxyz(asset.rotations_wxyz[index]))
    return rotations, quaternions


def _world_actor_means(asset: CanonicalActorAsset, local_positions: Any, camera: Any, *, torch: Any) -> tuple[Any, Any]:
    timestamp = int(camera.timestamp_start_us + (camera.timestamp_us - camera.timestamp_start_us) // 2) if camera.timestamp_start_us is not None else int(camera.timestamp_us)
    positions = torch.empty_like(local_positions)
    translations = []
    rotations, quaternions = _world_actor_parameters(asset, camera, device=local_positions.device)
    for actor_index, trajectory in enumerate(asset.trajectories):
        times = trajectory.timestamps_us
        right = min(max(int(np.searchsorted(times, timestamp, side="right")), 1), len(times) - 1)
        left = right - 1
        fraction = np.clip((timestamp - int(times[left])) / max(int(times[right] - times[left]), 1), 0.0, 1.0)
        translation = (1.0 - fraction) * trajectory.actor_to_world[left, :3, 3] + fraction * trajectory.actor_to_world[right, :3, 3]
        translations.append(translation)
        members = asset.actor_indices == actor_index
        rotation = torch.from_numpy(rotations[actor_index]).to(device=local_positions.device, dtype=local_positions.dtype)
        translation_tensor = torch.from_numpy(translation).to(device=local_positions.device, dtype=local_positions.dtype)
        positions[members] = local_positions[members] @ rotation.T + translation_tensor
    return positions, torch.from_numpy(quaternions).to(device=local_positions.device, dtype=local_positions.dtype)


def _build_observations(
    options: FThetaActorOptimizeOptions,
    renderer: GsplatRenderer,
    actor_renderer: GsplatRenderer,
    asset: CanonicalActorAsset,
    tracks: list[dict[str, Any]],
) -> tuple[list[DirectObservation], list[DirectObservation]]:
    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(options.ncore_path)
    reference = loader.get_camera_sensor("camera_front_wide_120fov")
    reference_exposures = _ncore_exposure_timestamps(reference)
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    model_by_camera: dict[str, Any] = {}
    mask_by_camera_frame = _clean_static_mask_lookup(options.clean_static_mask_dir)
    mask_slot_frames = _clean_static_slot_frames(options.clean_static_mask_dir)
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov") if options.min_lidar_pixels else None
    observations: list[DirectObservation] = []
    validation: list[DirectObservation] = []
    requested = ([(index, False, None) for index in options.frame_indices]
                 + [(index, True, None) for index in options.validation_frame_indices]
                 + [(mask_slot_frames[("camera_front_wide_120fov", slot)], False, slot) for slot in options.clean_static_mask_slots]
                 + [(mask_slot_frames[("camera_front_wide_120fov", slot)], True, slot) for slot in options.validation_clean_static_mask_slots])
    for reference_index, is_validation, mask_slot in requested:
        if not 0 <= reference_index < len(reference_exposures):
            raise ValueError(f"reference frame index out of range: {reference_index}")
        reference_time = int(reference_exposures[reference_index, 1])
        for camera_id in options.camera_ids:
            sensor = loader.get_camera_sensor(camera_id)
            exposures = _ncore_exposure_timestamps(sensor)
            source_index = (
                mask_slot_frames[(camera_id, mask_slot)] if mask_slot is not None
                else _nearest_index(exposures[:, 1].astype(np.int64), reference_time)
            )
            config = {"frame_indices": [source_index], "output": {"width": options.width, "height": options.height}}
            camera = _load_ncore_source_camera_path(config, loader, camera_id).frames[0]
            source = np.asarray(sensor.get_frame_image_array(source_index), dtype=np.uint8)
            image = Image.fromarray(source, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR)
            target = np.asarray(image, dtype=np.float32) / 255.0
            probability, margin = segmenter.vehicle_scores(image)
            midpoint = int((int(exposures[source_index, 0]) + int(exposures[source_index, 1])) // 2)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            model = model_by_camera.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            cuboid = _ftheta_cuboid_mask(
                tracks, midpoint, camera_to_world, model, height=options.height, width=options.width,
                source_height=int(source.shape[0]), source_width=int(source.shape[1]),
                margin_pixels=options.cuboid_margin_pixels, device=options.device,
            ).astype(bool)
            roi = vehicle_roi(cuboid, np.ones_like(cuboid, dtype=bool), probability, margin,
                              probability_threshold=options.roi_probability_threshold,
                              margin_threshold=options.roi_margin_threshold)
            if not roi.any():
                print(f"Skip {camera_id} frame={source_index}: no confident vehicle ROI", flush=True)
                continue
            source_mask_path = mask_by_camera_frame.get((camera_id, source_index))
            if source_mask_path is None:
                if options.min_source_mask_coverage > 0.0:
                    print(f"Skip {camera_id} frame={source_index}: not an exact clean-static source mask slot", flush=True)
                    continue
                source_mask = np.ones_like(roi, dtype=bool)
                mask_coverage: float | None = None
            else:
                source_mask_native = np.asarray(Image.open(source_mask_path).convert("L"), dtype=np.uint8) > 0
                if source_mask_native.shape != source.shape[:2]:
                    raise ValueError(f"clean-static source mask shape differs from NCore image: {source_mask_path}")
                source_mask = np.asarray(
                    Image.fromarray(source_mask_native.astype(np.uint8) * 255, mode="L").resize(
                        (options.width, options.height), Image.Resampling.NEAREST
                    ), dtype=np.uint8
                ) > 0
                mask_coverage = float(source_mask[roi].mean())
                if mask_coverage < options.min_source_mask_coverage:
                    print(
                        f"Skip {camera_id} frame={source_index}: source exclusion coverage={mask_coverage:.3f} "
                        f"< {options.min_source_mask_coverage:.3f}", flush=True,
                    )
                    continue
            dilated = dilate_boolean_mask(roi, options.roi_dilation_pixels)
            with renderer.torch.no_grad():
                static = renderer.render(camera, include_depth=True)
                actor = actor_renderer.render(camera, include_depth=True)
            static_alpha = static.alpha >= 0.02
            actor_depth = np.asarray(actor.depth, dtype=np.float32)
            static_depth = np.asarray(static.depth, dtype=np.float32)
            visible = visibility_gate_mask(
                roi, source_mask, static_alpha, static_depth, actor_depth,
                depth_margin_m=options.static_occluder_depth_margin_m,
            )
            lidar_visible_pixels = 0
            if lidar_sensor is not None:
                lidar_index = int(lidar_sensor.get_closest_frame_index(midpoint))
                lidar_time = int(lidar_sensor.get_frame_timestamp_us(lidar_index))
                if abs(lidar_time - midpoint) <= options.max_lidar_timestamp_delta_us:
                    lidar_points = lidar_sensor.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                    lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
                    lidar_world = np.asarray(lidar_points, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
                    lidar_depth = _project_lidar_depth_ftheta(
                        lidar_world, np.linalg.inv(camera_to_world), model,
                        source_width=int(source.shape[1]), source_height=int(source.shape[0]),
                        width=options.width, height=options.height, device=options.device,
                    )
                    lidar_visible_pixels = int((visible & (lidar_depth > 0.0)).sum())
                if lidar_visible_pixels < options.min_lidar_pixels:
                    print(
                        f"Skip {camera_id} frame={source_index}: LiDAR support={lidar_visible_pixels} "
                        f"< {options.min_lidar_pixels}", flush=True,
                    )
                    continue
            if not visible.any():
                print(f"Skip {camera_id} frame={source_index}: no source-visible non-occluded ROI", flush=True)
                continue
            record = DirectObservation(
                camera_id=camera_id, reference_frame_index=reference_index,
                source_frame_index=source_index, camera=camera,
                target_rgb=target.astype(np.float16), static_rgb=static.rgb.astype(np.float16),
                vehicle_roi=roi, dilated_vehicle_roi=dilated,
                visible_roi=visible, static_alpha=static_alpha, static_depth=static_depth,
                actor_initial_depth=actor_depth, clean_static_mask_coverage=mask_coverage,
                lidar_visible_pixels=lidar_visible_pixels, timestamp_midpoint_us=midpoint,
            )
            (validation if is_validation else observations).append(record)
            print(
                f"Prepared {'val' if is_validation else 'train'} {camera_id} frame={source_index}: "
                f"roi={int(roi.sum())} visible={int(visible.sum())} mask={mask_coverage} lidar={lidar_visible_pixels}", flush=True,
            )
    if not observations:
        raise RuntimeError("no direct FTheta/rolling-shutter training observations were usable")
    return observations, validation


def _asset_from_parameters(asset: CanonicalActorAsset, local_positions: Any, rgb_delta: Any, opacity_delta: Any, options: FThetaActorOptimizeOptions, metadata: dict[str, object]) -> CanonicalActorAsset:
    positions = (asset.positions + options.max_position_delta_m * np.tanh(local_positions.detach().cpu().numpy())).astype(np.float32)
    rgb = np.clip(asset.rgb + options.max_color_delta * np.tanh(rgb_delta.detach().cpu().numpy()), 0.0, 1.0).astype(np.float32)
    logits = np.log(np.clip(asset.opacities, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - asset.opacities, 1e-4, 1.0))
    opacities = (1.0 / (1.0 + np.exp(-(logits + opacity_delta.detach().cpu().numpy())))).astype(np.float32)
    result_metadata = dict(asset.metadata)
    result_metadata.update(metadata)
    result = CanonicalActorAsset(
        positions=positions, rotations_wxyz=asset.rotations_wxyz, scales=asset.scales, rgb=rgb,
        opacities=opacities, actor_indices=asset.actor_indices, visibility_counts=asset.visibility_counts,
        source_counts=asset.source_counts, actor_ids=asset.actor_ids, trajectories=asset.trajectories,
        metadata=result_metadata,
    )
    result.validate()
    return result


def _appearance_coordinate(timestamp_us: int, center_us: float, scale_us: float) -> float:
    return float(np.clip((float(timestamp_us) - center_us) / max(scale_us, 1.0), -2.0, 2.0))


def _camera_appearance_offset(
    camera_id: str,
    timestamp_us: int,
    coefficients: Any,
    camera_indices: dict[str, int],
    *,
    center_us: float,
    scale_us: float,
    options: FThetaActorOptimizeOptions,
) -> Any:
    """Bounded camera-conditioned affine-in-time RGB residual in sRGB units."""
    index = camera_indices[camera_id]
    coordinate = _appearance_coordinate(timestamp_us, center_us, scale_us)
    return options.camera_appearance_max_delta * coefficients[index, 0].add(coordinate * coefficients[index, 1]).tanh()


def _validation_rgb_loss(
    validation: list[DirectObservation],
    asset: CanonicalActorAsset,
    renderer: GsplatRenderer,
    *,
    base_local: Any,
    base_rgb: Any,
    base_logit: Any,
    local_delta: Any,
    rgb_delta: Any,
    opacity_delta: Any,
    camera_appearance: Any | None,
    camera_indices: dict[str, int],
    appearance_center_us: float,
    appearance_scale_us: float,
    options: FThetaActorOptimizeOptions,
) -> float | None:
    """Evaluate frozen hold-out views with the same actor-over-static rule."""
    if not validation:
        return None
    torch = renderer.torch
    with torch.no_grad():
        local = base_local + options.max_position_delta_m * torch.tanh(local_delta)
        actor_opacity = torch.sigmoid(base_logit + opacity_delta)
        values: list[float] = []
        for observation in validation:
            actor_rgb = base_rgb + options.max_color_delta * torch.tanh(rgb_delta)
            if camera_appearance is not None:
                actor_rgb = actor_rgb + _camera_appearance_offset(
                    observation.camera_id, observation.timestamp_midpoint_us, camera_appearance, camera_indices,
                    center_us=appearance_center_us, scale_us=appearance_scale_us, options=options,
                )
            actor_colors = ((actor_rgb.clamp(0.0, 1.0) - 0.5) / 0.28209479177387814)[:, None, :]
            means, actor_quats = _world_actor_means(asset, local, observation.camera, torch=torch)
            actor_premultiplied, actor_alpha = _rasterize_rgb_alpha(
                renderer, observation.camera, means, actor_quats,
                torch.from_numpy(asset.scales).to(renderer.means.device), actor_opacity, actor_colors,
            )
            static_rgb = torch.from_numpy(observation.static_rgb.astype(np.float32)).to(renderer.means.device)
            target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(renderer.means.device)
            visible = torch.from_numpy(observation.visible_roi).to(renderer.means.device)
            prediction = actor_premultiplied + (1.0 - actor_alpha[..., None]) * static_rgb
            values.append(float(torch.sqrt((prediction - target).square() + 1e-6).mean(dim=-1)[visible].mean()))
    return float(np.mean(values))


def _checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".checkpoint.pt")


def optimize_ftheta_actors(options: FThetaActorOptimizeOptions) -> CanonicalActorAsset:
    """Optimise a compact canonical actor asset in exact FTheta/RS views."""
    import torch

    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite actor output: {options.output}")
    # The asset, its resumable checkpoint, and the final JSON report are all
    # sibling files.  Creating the parent here keeps a new output experiment
    # self-contained and avoids failing only after the first checkpoint.
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    static_scene = load_gaussian_ply(options.static_ply)
    initial_full = CanonicalActorAsset.load(options.initial_actors)
    initial = _subset_asset(initial_full, options.max_gaussians_per_actor)
    tracks = _load_tracks(options.actor_dataset_manifest, set(initial.actor_ids))
    static_renderer = GsplatRenderer(static_scene, RenderOptions(device=options.device, static_opacity_scale=1.0))
    actor_renderer = GsplatRenderer(
        static_scene, RenderOptions(device=options.device, static_opacity_scale=0.0), canonical_actor_asset=initial
    )
    train, validation = _build_observations(options, static_renderer, actor_renderer, initial, tracks)
    device = static_renderer.means.device
    local_delta = torch.zeros_like(torch.from_numpy(initial.positions).to(device), requires_grad=True)
    rgb_delta = torch.zeros_like(torch.from_numpy(initial.rgb).to(device), requires_grad=True)
    opacity_delta = torch.zeros_like(torch.from_numpy(initial.opacities).to(device), requires_grad=True)
    appearance_camera_ids = tuple(sorted({item.camera_id for item in (*train, *validation)}))
    camera_indices = {camera_id: index for index, camera_id in enumerate(appearance_camera_ids)}
    appearance_times = np.asarray([item.timestamp_midpoint_us for item in train], dtype=np.float64)
    appearance_center_us = float(appearance_times.mean())
    appearance_scale_us = float(max(np.abs(appearance_times - appearance_center_us).max(), 1.0))
    camera_appearance = (
        torch.zeros((len(appearance_camera_ids), 2, 3), dtype=local_delta.dtype, device=device, requires_grad=True)
        if options.camera_appearance_enabled else None
    )
    parameters = [local_delta, rgb_delta, opacity_delta]
    if camera_appearance is not None:
        parameters.append(camera_appearance)
    optimizer = torch.optim.Adam(parameters, lr=options.learning_rate)
    start_step = 0
    best_validation_rgb = float("inf")
    best_validation_step: int | None = None
    best_parameters: tuple[Any, Any, Any] | None = None
    if options.resume_checkpoint is not None:
        checkpoint = torch.load(options.resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("initial_asset_count") != initial.count:
            raise ValueError("checkpoint was made from a different compact initial asset")
        local_delta.data.copy_(checkpoint["local_delta"])
        rgb_delta.data.copy_(checkpoint["rgb_delta"])
        opacity_delta.data.copy_(checkpoint["opacity_delta"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_validation_rgb = float(checkpoint.get("best_validation_rgb", best_validation_rgb))
        best_validation_step = checkpoint.get("best_validation_step")
        if checkpoint.get("best_parameters") is not None:
            saved = checkpoint["best_parameters"]
            best_parameters = tuple(item.to(device) for item in saved)
    base_local = torch.from_numpy(initial.positions).to(device)
    base_rgb = torch.from_numpy(initial.rgb).to(device)
    base_opacity = torch.from_numpy(initial.opacities).to(device)
    base_logit = torch.logit(base_opacity.clamp(1e-4, 1.0 - 1e-4))
    history: list[dict[str, float]] = []
    for step in range(start_step, options.iterations):
        observation = train[step % len(train)]
        local = base_local + options.max_position_delta_m * torch.tanh(local_delta)
        means, actor_quats = _world_actor_means(initial, local, observation.camera, torch=torch)
        actor_rgb = base_rgb + options.max_color_delta * torch.tanh(rgb_delta)
        if camera_appearance is not None:
            actor_rgb = actor_rgb + _camera_appearance_offset(
                observation.camera_id, observation.timestamp_midpoint_us, camera_appearance, camera_indices,
                center_us=appearance_center_us, scale_us=appearance_scale_us, options=options,
            )
        actor_rgb = actor_rgb.clamp(0.0, 1.0)
        actor_colors = ((actor_rgb - 0.5) / 0.28209479177387814)[:, None, :]
        actor_opacity = torch.sigmoid(base_logit + opacity_delta)
        actor_premultiplied, actor_alpha = _rasterize_rgb_alpha(
            static_renderer, observation.camera, means, actor_quats,
            torch.from_numpy(initial.scales).to(device), actor_opacity, actor_colors,
        )
        # Frozen static geometry is rendered outside the autograd graph.  A
        # layer-over composition is mathematically the intended visibility
        # rule when a dynamic actor is in front: its premultiplied RGB replaces
        # only the remaining static transmittance.  Crucially, this avoids
        # retaining 2.8M frozen static Gaussian intermediates for actor BWD.
        static_rgb = torch.from_numpy(observation.static_rgb.astype(np.float32)).to(device)
        joint_rgb = actor_premultiplied + (1.0 - actor_alpha[..., None]) * static_rgb
        target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(device)
        visible = torch.from_numpy(observation.visible_roi).to(device)
        outside = torch.from_numpy(~observation.dilated_vehicle_roi).to(device)
        static_alpha = torch.from_numpy(observation.static_alpha).to(device)
        occluded = (
            (~observation.dilated_vehicle_roi) & observation.static_alpha
            & np.isfinite(observation.static_depth) & np.isfinite(observation.actor_initial_depth)
            & (observation.actor_initial_depth > observation.static_depth + options.static_occluder_depth_margin_m)
        )
        occluded = torch.from_numpy(occluded).to(device)
        charbonnier = torch.sqrt((joint_rgb - target).square() + 1e-6).mean(dim=-1)
        rgb_loss = charbonnier[visible].mean()
        outside_loss = actor_alpha[outside].mean() if bool(outside.any()) else actor_alpha.new_zeros(())
        occluded_loss = actor_alpha[occluded].mean() if bool(occluded.any()) else actor_alpha.new_zeros(())
        # The only allowed way to remove static dynamic leakage is to explain a
        # semantic actor pixel with a source-confirmed, non-occluded visible
        # actor.  This term never suppresses the static PLY itself.
        static_leak_loss = ((1.0 - actor_alpha) * static_alpha.float())[visible].mean() if bool(visible.any()) else actor_alpha.new_zeros(())
        position_prior = local_delta.square().mean()
        color_prior = rgb_delta.square().mean()
        opacity_prior = opacity_delta.square().mean()
        appearance_bias_prior = camera_appearance[:, 0].square().mean() if camera_appearance is not None else actor_alpha.new_zeros(())
        appearance_slope_prior = camera_appearance[:, 1].square().mean() if camera_appearance is not None else actor_alpha.new_zeros(())
        loss = (
            options.rgb_weight * rgb_loss + options.outside_alpha_weight * outside_loss
            + options.occluded_alpha_weight * occluded_loss + options.static_leak_weight * static_leak_loss
            + options.position_prior_weight * position_prior + options.color_prior_weight * color_prior
            + options.opacity_prior_weight * opacity_prior
            + options.camera_appearance_bias_prior_weight * appearance_bias_prior
            + options.camera_appearance_slope_prior_weight * appearance_slope_prior
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        record = {"step": float(step + 1), "loss": float(loss.detach()), "rgb": float(rgb_loss.detach()),
                  "outside_alpha": float(outside_loss.detach()), "occluded_alpha": float(occluded_loss.detach()),
                  "static_leak": float(static_leak_loss.detach()), "appearance_bias": float(appearance_bias_prior.detach()),
                  "appearance_slope": float(appearance_slope_prior.detach())}
        history.append(record)
        if step == start_step or (step + 1) % 10 == 0 or step + 1 == options.iterations:
            print(
                f"Direct FTheta optimize [{step + 1:05d}/{options.iterations}] {observation.camera_id}:{observation.source_frame_index} "
                f"loss={record['loss']:.6f} rgb={record['rgb']:.6f} outside={record['outside_alpha']:.6f} "
                f"occluded={record['occluded_alpha']:.6f} leak={record['static_leak']:.6f}", flush=True,
            )
        if (step + 1) % options.checkpoint_every == 0 or step + 1 == options.iterations:
            validation_rgb = _validation_rgb_loss(
                validation, initial, static_renderer, base_local=base_local, base_rgb=base_rgb,
                base_logit=base_logit, local_delta=local_delta, rgb_delta=rgb_delta,
                opacity_delta=opacity_delta, camera_appearance=camera_appearance, camera_indices=camera_indices,
                appearance_center_us=appearance_center_us, appearance_scale_us=appearance_scale_us, options=options,
            )
            if validation_rgb is not None:
                record["validation_rgb"] = validation_rgb
                print(f"  holdout_rgb={validation_rgb:.6f}", flush=True)
                if validation_rgb < best_validation_rgb:
                    best_validation_rgb = validation_rgb
                    best_validation_step = step + 1
                    best_parameters = tuple(parameter.detach().clone() for parameter in (local_delta, rgb_delta, opacity_delta))
            torch.save({
                "step": step + 1, "initial_asset_count": initial.count, "local_delta": local_delta.detach(),
                "rgb_delta": rgb_delta.detach(), "opacity_delta": opacity_delta.detach(), "optimizer": optimizer.state_dict(),
                "best_validation_rgb": best_validation_rgb, "best_validation_step": best_validation_step,
                "best_parameters": best_parameters,
            }, _checkpoint_path(options.output))
    metadata = {
        "schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION,
        "optimizer_status": "direct native NCore FTheta/rolling-shutter object optimization",
        "initial_actors": str(options.initial_actors.resolve()), "static_ply": str(options.static_ply.resolve()),
        "actor_dataset_manifest": str(options.actor_dataset_manifest.resolve()), "ncore_path": str(options.ncore_path.resolve()),
        "camera_ids": list(options.camera_ids), "training_frame_indices": list(options.frame_indices),
        "validation_frame_indices": list(options.validation_frame_indices),
        "training_clean_static_mask_slots": list(options.clean_static_mask_slots),
        "validation_clean_static_mask_slots": list(options.validation_clean_static_mask_slots),
        "resolution": [options.width, options.height],
        "loss_constraints": {"outside_alpha": options.outside_alpha_weight, "occluded_alpha": options.occluded_alpha_weight,
                             "static_leak": options.static_leak_weight, "static_occluder_depth_margin_m": options.static_occluder_depth_margin_m},
        "visibility_gates": {
            "clean_static_mask_dir": None if options.clean_static_mask_dir is None else str(options.clean_static_mask_dir.resolve()),
            "min_source_mask_coverage": options.min_source_mask_coverage,
            "min_lidar_pixels": options.min_lidar_pixels,
            "max_lidar_timestamp_delta_us": options.max_lidar_timestamp_delta_us,
        },
        "render_compositing": "actor_premultiplied + (1-actor_alpha) * frozen_static_rgb; static outside autograd",
        "selected_step": best_validation_step if best_parameters is not None else options.iterations,
        "best_validation_rgb": best_validation_rgb if best_parameters is not None else None,
        "history_tail": history[-100:], "train_observation_count": len(train), "validation_observation_count": len(validation),
    }
    if camera_appearance is not None:
        appearance_path = options.output.with_suffix(".appearance.npz")
        np.savez_compressed(
            appearance_path, camera_ids=np.asarray(appearance_camera_ids),
            coefficients=camera_appearance.detach().cpu().numpy().astype(np.float32),
            center_us=np.asarray(appearance_center_us, dtype=np.float64),
            scale_us=np.asarray(appearance_scale_us, dtype=np.float64),
            max_delta=np.asarray(options.camera_appearance_max_delta, dtype=np.float32),
        )
        metadata["camera_appearance_diagnostic"] = {
            "path": str(appearance_path), "camera_ids": list(appearance_camera_ids),
            "time_center_us": appearance_center_us, "time_scale_us": appearance_scale_us,
            "max_delta": options.camera_appearance_max_delta,
            "note": "diagnostic sidecar; not yet accepted for production renderer or sequence output",
        }
    selected_parameters = best_parameters or (local_delta, rgb_delta, opacity_delta)
    result = _asset_from_parameters(initial, *selected_parameters, options, metadata)
    result.save(options.output)
    report_path = options.output.with_suffix(".report.json")
    report_path.write_text(json.dumps({"output": str(options.output), "gaussian_count": result.count,
                                       "actor_ids": list(result.actor_ids), "history": history,
                                       "train_observations": [{"camera_id": item.camera_id, "source_frame_index": item.source_frame_index, "visible_roi_pixels": int(item.visible_roi.sum()), "clean_static_mask_coverage": item.clean_static_mask_coverage, "lidar_visible_pixels": item.lidar_visible_pixels} for item in train],
                                       "validation_observations": [{"camera_id": item.camera_id, "source_frame_index": item.source_frame_index, "visible_roi_pixels": int(item.visible_roi.sum()), "clean_static_mask_coverage": item.clean_static_mask_coverage, "lidar_visible_pixels": item.lidar_visible_pixels} for item in validation]}, indent=2), encoding="utf-8")
    print(f"Wrote direct FTheta optimized canonical actors: {options.output}", flush=True)
    return result
