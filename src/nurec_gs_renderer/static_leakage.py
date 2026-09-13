"""Conservative multi-view attribution of dynamic leakage in a static PLY.

The earlier proximity experiment asks whether a static Gaussian happens to be
near a dynamic Gaussian.  That is not enough in a road scene: a valid pole,
parked car, or background wall can be close to a moving object.  This module
instead requires evidence in the original observations: a Gaussian must be
inside the tracked object volume, project to the object-specific semantic ROI,
be depth-compatible with the *visible static* rendering, and (by default) be
consistent with sparse LiDAR at least once.  The output remains a reversible,
PLY-hash-bound opacity sidecar.

It is deliberately an attribution/validation tool, not a replacement for a
clean-background reconstruction.  A successful sidecar only proves that
retraining the static background with those dynamic pixels masked is warranted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask, _read_actor_tracks, vehicle_roi
from .camera import compose_ncore_camera_pose
from .color_correction import file_sha256
from .opacity_correction import GaussianOpacityCorrection
from .opacity_decontaminate import _load_semantic_masks
from .ply_io import GaussianScene, load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path, _ncore_exposure_timestamps
from .render import GsplatRenderer, RenderOptions
from .street_dataset import SegformerVehicleSegmenter, _interpolate_track_pose, _nearest_index


DEFAULT_CAMERA_IDS = (
    "camera_cross_left_120fov",
    "camera_rear_left_70fov",
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
)


@dataclass(frozen=True)
class StaticLeakageOptions:
    ply_path: Path
    ncore_path: Path
    actor_dataset_manifest: Path
    segformer_model_dir: Path
    output: Path
    track_id: int
    reference_frame_indices: tuple[int, ...]
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    device: str = "cuda"
    width: int = 480
    height: int = 270
    cuboid_margin_m: float = 0.10
    cuboid_margin_pixels: int = 8
    vehicle_probability_threshold: float = 0.60
    vehicle_margin_threshold: float = 0.05
    static_depth_tolerance_m: float = 0.50
    lidar_depth_tolerance_m: float = 0.75
    max_lidar_timestamp_delta_us: int = 100_000
    min_view_support: int = 3
    min_lidar_support: int = 0
    opacity_scale: float = 0.50
    exclude_road: bool = True
    exclude_sky: bool = True

    def __post_init__(self) -> None:
        if self.output.suffix.lower() != ".npz":
            raise ValueError("output must use the .npz extension")
        if self.track_id < 0 or not self.reference_frame_indices or any(value < 0 for value in self.reference_frame_indices):
            raise ValueError("track_id and reference_frame_indices are invalid")
        if tuple(sorted(set(self.reference_frame_indices))) != self.reference_frame_indices:
            raise ValueError("reference_frame_indices must be unique and increasing")
        if not self.camera_ids or self.width <= 0 or self.height <= 0:
            raise ValueError("at least one camera and positive output resolution are required")
        if self.cuboid_margin_m < 0 or self.cuboid_margin_pixels < 0:
            raise ValueError("cuboid margins must be non-negative")
        if self.static_depth_tolerance_m <= 0 or self.lidar_depth_tolerance_m <= 0:
            raise ValueError("depth tolerances must be positive")
        if self.max_lidar_timestamp_delta_us <= 0 or self.min_view_support <= 0 or self.min_lidar_support < 0:
            raise ValueError("support and LiDAR timestamp settings are invalid")
        if not 0.0 <= self.opacity_scale <= 1.0:
            raise ValueError("opacity_scale must be in [0,1]")


def select_cuboid_volume_candidates(
    means_world: np.ndarray,
    track: dict[str, Any],
    timestamps_us: np.ndarray,
    *,
    margin_m: float,
) -> np.ndarray:
    """Return static points that occupy this actor's expanded volume at any time."""
    means = np.asarray(means_world, dtype=np.float32)
    selected = np.zeros(len(means), dtype=bool)
    dimensions = np.asarray(track["length_width_height"], dtype=np.float32)
    half = dimensions * 0.5 + float(margin_m)
    for timestamp in np.asarray(timestamps_us, dtype=np.int64):
        pose = _interpolate_track_pose(track, int(timestamp))
        if pose is None:
            continue
        # actor_to_world uses row-vector form: p_world = p_local R^T + t.
        # Therefore p_local = (p_world - t) R.
        local = (means - pose[:3, 3].astype(np.float32)) @ pose[:3, :3].astype(np.float32)
        selected |= np.all(np.abs(local) <= half[None], axis=1)
    return np.flatnonzero(selected).astype(np.int64)


def select_multiview_leakage_candidates(
    candidate_indices: np.ndarray,
    view_support: np.ndarray,
    lidar_support: np.ndarray,
    *,
    min_view_support: int,
    min_lidar_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the non-negotiable multi-view and sparse-depth support gates."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    view = np.asarray(view_support)
    lidar = np.asarray(lidar_support)
    if view.shape != candidate_indices.shape or lidar.shape != candidate_indices.shape:
        raise ValueError("candidate and support arrays must have matching shape")
    selected = view >= min_view_support
    if min_lidar_support:
        selected &= lidar >= min_lidar_support
    indices = candidate_indices[selected]
    lidar_factor = (
        np.ones(int(selected.sum()), dtype=np.float32)
        if min_lidar_support == 0
        else np.minimum(lidar[selected] / min_lidar_support, 1.0)
    )
    confidence = np.clip(
        np.sqrt(np.minimum(view[selected] / max(min_view_support, 1), 1.0) * lidar_factor),
        0.0,
        1.0,
    ).astype(np.float32)
    return indices, confidence


def _project_world_points_ftheta(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    model: object,
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points with NCore's FTheta model at the exposure midpoint."""
    import torch

    points_camera = (
        np.asarray(points_world, dtype=np.float32) @ np.asarray(world_to_camera[:3, :3], dtype=np.float32).T
        + np.asarray(world_to_camera[:3, 3], dtype=np.float32)
    )
    depths = points_camera[:, 2].astype(np.float32)
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points_camera).all(axis=1) & (depths > 0.10)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        for start in range(0, len(valid_indices), 200_000):
            indices = valid_indices[start : start + 200_000]
            camera = points_camera[indices]
            rays = camera / np.linalg.norm(camera, axis=1, keepdims=True).clip(1e-8)
            returned = model.camera_rays_to_pixels(torch.from_numpy(rays).to(device))
            values = returned.pixels.detach().float().cpu().numpy()
            values[:, 0] = (values[:, 0] + 0.5) * (width / source_width) - 0.5
            values[:, 1] = (values[:, 1] + 0.5) * (height / source_height) - 0.5
            pixels[indices] = values
            valid[indices] &= returned.valid_flag.detach().cpu().numpy().astype(bool)
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, depths, valid


def _project_lidar_depth_ftheta(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    model: object,
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    device: str,
) -> np.ndarray:
    pixels, depths, valid = _project_world_points_ftheta(
        points_world, world_to_camera, model, source_width=source_width, source_height=source_height,
        width=width, height=height, device=device,
    )
    xx = np.where(valid, np.rint(pixels[:, 0]), -1).astype(np.int64)
    yy = np.where(valid, np.rint(pixels[:, 1]), -1).astype(np.int64)
    valid &= (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    flat = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(flat, yy[valid] * width + xx[valid], depths[valid])
    flat[~np.isfinite(flat)] = 0.0
    return flat.reshape(height, width)


def _matching_track(path: Path, track_id: int) -> dict[str, Any]:
    tracks = _read_actor_tracks(path)
    matches = [track for track in tracks if int(track["track_id"]) == int(track_id)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one track_id={track_id} in actor manifest, got {len(matches)}")
    return matches[0]


def build_multiview_static_leakage_correction(options: StaticLeakageOptions) -> GaussianOpacityCorrection:
    """Build a conservative, source-hash-bound static leakage opacity sidecar."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite opacity correction: {options.output}")
    from ncore.impl.sensors.camera import FThetaCameraModel

    scene = load_gaussian_ply(options.ply_path)
    track = _matching_track(options.actor_dataset_manifest, options.track_id)
    loader = _create_ncore_loader(options.ncore_path)
    reference_sensor = loader.get_camera_sensor("camera_front_wide_120fov")
    reference_exposures = _ncore_exposure_timestamps(reference_sensor)
    if max(options.reference_frame_indices) >= len(reference_exposures):
        raise ValueError("reference frame index exceeds front-wide camera length")
    reference_midpoints = (
        reference_exposures[np.asarray(options.reference_frame_indices), 0].astype(np.int64)
        + reference_exposures[np.asarray(options.reference_frame_indices), 1].astype(np.int64)
    ) // 2
    candidates = select_cuboid_volume_candidates(
        scene.means, track, reference_midpoints, margin_m=options.cuboid_margin_m
    )
    if not len(candidates):
        raise RuntimeError("no static Gaussian lies in the requested actor trajectory volume")
    road, sky = _load_semantic_masks(
        options.ply_path, scene.count, exclude_road=options.exclude_road, exclude_sky=options.exclude_sky
    )
    candidate_ok = ~road[candidates] & ~sky[candidates]
    candidates = candidates[candidate_ok]
    if not len(candidates):
        raise RuntimeError("all trajectory-volume candidates were excluded as road or sky")

    renderer = GsplatRenderer(scene, RenderOptions(device=options.device))
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in options.camera_ids}
    models = {camera_id: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera_id, sensor in sensors.items()}
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov")
    view_support = np.zeros(len(candidates), dtype=np.uint16)
    lidar_support = np.zeros(len(candidates), dtype=np.uint16)
    lidar_conflict = np.zeros(len(candidates), dtype=np.uint16)
    observation_count = 0
    usable_observation_count = 0
    records: list[dict[str, object]] = []

    for reference_index, reference_time in zip(options.reference_frame_indices, reference_exposures[np.asarray(options.reference_frame_indices), 1], strict=True):
        for camera_id, sensor in sensors.items():
            observation_count += 1
            exposures = _ncore_exposure_timestamps(sensor)
            source_index = _nearest_index(exposures[:, 1].astype(np.int64), int(reference_time))
            source = np.asarray(sensor.get_frame_image_array(source_index), dtype=np.uint8)
            image = Image.fromarray(source, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR)
            probability, margin = segmenter.vehicle_scores(image)
            midpoint = int((int(exposures[source_index, 0]) + int(exposures[source_index, 1])) // 2)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [source_index], "output": {"width": options.width, "height": options.height}}, loader, camera_id
            ).frames[0]
            cuboid = _ftheta_cuboid_mask(
                [track], midpoint, camera_to_world, models[camera_id], height=options.height, width=options.width,
                source_height=int(source.shape[0]), source_width=int(source.shape[1]),
                margin_pixels=options.cuboid_margin_pixels, device=options.device,
            ).astype(bool)
            roi = vehicle_roi(cuboid, np.ones_like(cuboid, dtype=bool), probability, margin,
                              probability_threshold=options.vehicle_probability_threshold,
                              margin_threshold=options.vehicle_margin_threshold)
            if not roi.any():
                records.append({"reference_frame_index": int(reference_index), "camera_id": camera_id,
                                "source_frame_index": int(source_index), "roi_pixels": 0, "accepted": 0})
                continue
            rendered = renderer.render(camera, include_depth=True)
            static_depth = np.asarray(rendered.depth, dtype=np.float32)
            static_alpha = np.asarray(rendered.alpha, dtype=np.float32)
            pixels, point_depth, valid = _project_world_points_ftheta(
                scene.means[candidates], np.linalg.inv(camera_to_world), models[camera_id],
                source_width=int(source.shape[1]), source_height=int(source.shape[0]), width=options.width,
                height=options.height, device=options.device,
            )
            xx = np.where(valid, np.rint(pixels[:, 0]), -1).astype(np.int64)
            yy = np.where(valid, np.rint(pixels[:, 1]), -1).astype(np.int64)
            valid &= (xx >= 0) & (xx < options.width) & (yy >= 0) & (yy < options.height)
            valid_indices = np.flatnonzero(valid)
            accepted = np.zeros(len(candidates), dtype=bool)
            if len(valid_indices):
                x, y = xx[valid_indices], yy[valid_indices]
                rendered_depth = static_depth[y, x]
                allowed = (
                    roi[y, x] & (static_alpha[y, x] >= 0.02) & np.isfinite(rendered_depth)
                    & (np.abs(point_depth[valid_indices] - rendered_depth) <= options.static_depth_tolerance_m)
                )
                accepted[valid_indices] = allowed
            lidar_accepted = np.zeros(len(candidates), dtype=bool)
            lidar_conflicted = np.zeros(len(candidates), dtype=bool)
            lidar_index = int(lidar_sensor.get_closest_frame_index(int(reference_time)))
            lidar_time = int(lidar_sensor.get_frame_timestamp_us(lidar_index))
            lidar_valid = abs(lidar_time - int(reference_time)) <= options.max_lidar_timestamp_delta_us
            if lidar_valid and accepted.any():
                lidar_points = lidar_sensor.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
                lidar_world = np.asarray(lidar_points, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
                lidar_depth = _project_lidar_depth_ftheta(
                    lidar_world, np.linalg.inv(camera_to_world), models[camera_id], source_width=int(source.shape[1]),
                    source_height=int(source.shape[0]), width=options.width, height=options.height, device=options.device,
                )
                indices = np.flatnonzero(accepted)
                sample_depth = lidar_depth[yy[indices], xx[indices]]
                lidar_accepted[indices] = (sample_depth > 0.0) & (
                    np.abs(point_depth[indices] - sample_depth) <= options.lidar_depth_tolerance_m
                )
                lidar_conflicted[indices] = (sample_depth > 0.0) & ~lidar_accepted[indices]
                # Sparse LiDAR is a positive/negative measurement, not a
                # dense segmentation map.  Missing hits leave the multi-view
                # semantic/static-depth evidence usable; a contradictory hit
                # vetoes that view for this Gaussian.
                accepted[lidar_conflicted] = False
            view_support += accepted.astype(np.uint16)
            lidar_support += lidar_accepted.astype(np.uint16)
            lidar_conflict += lidar_conflicted.astype(np.uint16)
            usable_observation_count += 1
            records.append({
                "reference_frame_index": int(reference_index), "camera_id": camera_id,
                "source_frame_index": int(source_index), "roi_pixels": int(roi.sum()),
                "projection_valid": int(valid.sum()), "static_depth_accepted": int(accepted.sum()),
                "lidar_accepted": int(lidar_accepted.sum()), "lidar_conflicts": int(lidar_conflicted.sum()),
                "lidar_timestamp_us": lidar_time,
                "lidar_within_tolerance": bool(lidar_valid),
            })
            print(
                f"Leakage evidence ref={reference_index} camera={camera_id} roi={int(roi.sum())} "
                f"static={int(accepted.sum())} lidar={int(lidar_accepted.sum())} conflict={int(lidar_conflicted.sum())}", flush=True,
            )

    selected, confidence = select_multiview_leakage_candidates(
        candidates, view_support, lidar_support, min_view_support=options.min_view_support,
        min_lidar_support=options.min_lidar_support,
    )
    details: dict[str, object] = {
        "method": "track_specific_multiview_semantic_static_depth_lidar_v1",
        "ncore_path": str(options.ncore_path), "actor_dataset_manifest": str(options.actor_dataset_manifest),
        "track_id": options.track_id, "camera_ids": list(options.camera_ids),
        "reference_frame_indices": list(options.reference_frame_indices), "resolution": [options.width, options.height],
        "cuboid_margin_m": options.cuboid_margin_m, "cuboid_margin_pixels": options.cuboid_margin_pixels,
        "static_depth_tolerance_m": options.static_depth_tolerance_m, "lidar_depth_tolerance_m": options.lidar_depth_tolerance_m,
        "min_view_support": options.min_view_support, "min_lidar_support": options.min_lidar_support,
        "opacity_scale": options.opacity_scale, "trajectory_volume_candidates": int(len(candidates)),
        "selected_gaussians": int(len(selected)), "observation_count": observation_count,
        "usable_observation_count": usable_observation_count,
        "view_support_percentiles": np.percentile(view_support, [0, 25, 50, 75, 100]).tolist(),
        "lidar_support_percentiles": np.percentile(lidar_support, [0, 25, 50, 75, 100]).tolist(),
        "lidar_conflict_percentiles": np.percentile(lidar_conflict, [0, 25, 50, 75, 100]).tolist(),
        "views": records,
    }
    if not len(selected):
        report = options.output.with_suffix(".report.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        details["status"] = "no_candidate_passed_gates"
        report.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            "no static Gaussian met multi-view/depth leakage support; do not relax gates blindly—inspect "
            f"{report}"
        )
    original = scene.opacities[selected].astype(np.float32)
    correction = GaussianOpacityCorrection(
        indices=selected.astype(np.int64), original_opacities=original,
        corrected_opacities=(original * options.opacity_scale).astype(np.float32), confidence=confidence,
        support_count=view_support[np.isin(candidates, selected)].astype(np.uint16), ply_sha256=file_sha256(options.ply_path),
        gaussian_count=scene.count,
        metadata=details,
    )
    correction.save(options.output)
    options.output.with_suffix(".report.json").write_text(json.dumps(correction.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote multi-view static leakage correction for {correction.count:,} Gaussians: {options.output}", flush=True)
    return correction
