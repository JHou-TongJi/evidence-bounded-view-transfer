"""Temporally regularised rigid-pose correction in native NCore FTheta space.

This deliberately freezes canonical Gaussian appearance and geometry.  It
only learns a bounded SE(3) correction curve over the NCore cuboid trajectory,
so a failed temporal experiment cannot silently turn into a per-frame 3-DGS
appearance fit.  The resulting trajectory is written back into a normal
``CanonicalActorAsset`` and is directly consumable by the production renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .canonical_actor import ActorTrajectory, CanonicalActorAsset
from .ftheta_actor_optimize import (
    DIRECT_DEFAULT_CAMERAS,
    DirectObservation,
    FThetaActorOptimizeOptions,
    _build_observations,
    _camera_kwargs,
    _load_tracks,
    _rasterize_rgb_alpha,
    _subset_asset,
    _world_actor_parameters,
)
from .ply_io import load_gaussian_ply
from .render import GsplatRenderer, RenderOptions


@dataclass(frozen=True)
class FThetaActorPoseOptimizeOptions(FThetaActorOptimizeOptions):
    """Pose-only extension of the direct FTheta actor objective."""

    pose_knot_frame_indices: tuple[int, ...] = ()
    pose_translation_limit_m: float = 0.25
    pose_rotation_limit_deg: float = 4.0
    temporal_rgb_weight: float = 1.0
    pose_prior_weight: float = 0.02
    pose_velocity_weight: float = 0.10
    pose_acceleration_weight: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        knots = self.pose_knot_frame_indices or tuple(sorted(set((*self.frame_indices, *self.validation_frame_indices))))
        if len(knots) < 2 or tuple(sorted(set(knots))) != knots or any(value < 0 for value in knots):
            raise ValueError("pose_knot_frame_indices must contain at least two unique increasing indices")
        if self.pose_translation_limit_m <= 0 or self.pose_rotation_limit_deg <= 0:
            raise ValueError("pose correction bounds must be positive")
        if any(value < 0 for value in (
            self.temporal_rgb_weight, self.pose_prior_weight,
            self.pose_velocity_weight, self.pose_acceleration_weight,
        )):
            raise ValueError("pose loss weights must be non-negative")

    @property
    def resolved_pose_knots(self) -> tuple[int, ...]:
        return self.pose_knot_frame_indices or tuple(sorted(set((*self.frame_indices, *self.validation_frame_indices))))


def _reference_midpoint_us(observation: DirectObservation) -> int:
    camera = observation.camera
    return int(camera.timestamp_start_us + (camera.timestamp_us - camera.timestamp_start_us) // 2) if camera.timestamp_start_us is not None else int(camera.timestamp_us)


def _interpolation_indices(knots: tuple[int, ...], frame_index: int) -> tuple[int, int, float]:
    """Clamp outside knots and linearly interpolate inside them."""
    values = np.asarray(knots, dtype=np.int64)
    right = int(np.searchsorted(values, frame_index, side="left"))
    if right <= 0:
        return 0, 0, 0.0
    if right >= len(values):
        last = len(values) - 1
        return last, last, 0.0
    left = right - 1
    fraction = float((frame_index - int(values[left])) / max(int(values[right] - values[left]), 1))
    return left, right, fraction


def _interpolated_correction(deltas: Any, knots: tuple[int, ...], frame_index: int) -> Any:
    left, right, fraction = _interpolation_indices(knots, frame_index)
    return (1.0 - fraction) * deltas[left] + fraction * deltas[right]


def _clip_pose_deltas(deltas: Any, translation_limit_m: float, rotation_limit_rad: float) -> None:
    """Project each SE(3) update into an isotropic, interpretable bound.

    Component-wise clamping lets a diagonal vector exceed the advertised
    translation/rotation bound by ``sqrt(3)``.  The optimizer's safety bounds
    are physical magnitudes, so clamp the two 3-vectors by their L2 norms.
    """
    import torch

    with torch.no_grad():
        for values, limit in ((deltas[..., :3], translation_limit_m), (deltas[..., 3:], rotation_limit_rad)):
            magnitude = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
            values.mul_((float(limit) / magnitude.clamp_min(float(limit))).clamp_max(1.0))


def _quaternion_multiply(left: Any, right: Any) -> Any:
    import torch

    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def _axis_angle_quaternion(axis_angle: Any) -> Any:
    import torch
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half = angle * 0.5
    scale = 0.5 * torch.sinc(angle / (2.0 * torch.pi))
    return torch.cat((torch.cos(half), axis_angle * scale), dim=-1)


def _quaternion_matrix(quaternion: Any) -> Any:
    import torch
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def _world_actor_pose_corrected(
    asset: CanonicalActorAsset, local_positions: Any, observation: DirectObservation, correction: Any
) -> tuple[Any, Any]:
    """Rigidly apply an actor-wise world-frame SE(3) correction with gradients."""
    import torch
    timestamp = _reference_midpoint_us(observation)
    base_rotations, base_quaternions = _world_actor_parameters(asset, observation.camera, device=local_positions.device)
    corrected_positions = torch.empty_like(local_positions)
    corrected_quaternions = torch.empty((asset.count, 4), device=local_positions.device, dtype=local_positions.dtype)
    correction_quaternions = _axis_angle_quaternion(correction[:, 3:])
    correction_rotations = _quaternion_matrix(correction_quaternions)
    base_world_quaternions = torch.from_numpy(base_quaternions).to(local_positions.device, dtype=local_positions.dtype)
    for actor_index, trajectory in enumerate(asset.trajectories):
        times = trajectory.timestamps_us
        right = min(max(int(np.searchsorted(times, timestamp, side="right")), 1), len(times) - 1)
        left = right - 1
        fraction = np.clip((timestamp - int(times[left])) / max(int(times[right] - times[left]), 1), 0.0, 1.0)
        translation = (1.0 - fraction) * trajectory.actor_to_world[left, :3, 3] + fraction * trajectory.actor_to_world[right, :3, 3]
        base_rotation = torch.from_numpy(base_rotations[actor_index]).to(local_positions.device, dtype=local_positions.dtype)
        rotation = correction_rotations[actor_index] @ base_rotation
        members = asset.actor_indices == actor_index
        translation_tensor = torch.from_numpy(translation).to(local_positions.device, dtype=local_positions.dtype)
        corrected_positions[members] = local_positions[members] @ rotation.T + translation_tensor + correction[actor_index, :3]
        corrected_quaternions[members] = _quaternion_multiply(
            correction_quaternions[actor_index].expand(int(members.sum()), -1), base_world_quaternions[members]
        )
    return corrected_positions, corrected_quaternions


def _render_joint(renderer: GsplatRenderer, asset: CanonicalActorAsset, observation: DirectObservation, correction: Any) -> tuple[Any, Any]:
    torch = renderer.torch
    local = torch.from_numpy(asset.positions).to(renderer.means.device)
    means, quaternions = _world_actor_pose_corrected(asset, local, observation, correction)
    colors = torch.from_numpy(((asset.rgb - 0.5) / 0.28209479177387814)[:, None, :]).to(renderer.means.device)
    opacities = torch.from_numpy(asset.opacities).to(renderer.means.device)
    actor_rgb, actor_alpha = _rasterize_rgb_alpha(
        renderer, observation.camera, means, quaternions,
        torch.from_numpy(asset.scales).to(renderer.means.device), opacities, colors,
    )
    static = torch.from_numpy(observation.static_rgb.astype(np.float32)).to(renderer.means.device)
    return actor_rgb + (1.0 - actor_alpha[..., None]) * static, actor_alpha


def _temporal_pairs(observations: list[DirectObservation]) -> list[tuple[DirectObservation, DirectObservation]]:
    groups: dict[str, list[DirectObservation]] = {}
    for observation in observations:
        groups.setdefault(observation.camera_id, []).append(observation)
    pairs: list[tuple[DirectObservation, DirectObservation]] = []
    for group in groups.values():
        group.sort(key=lambda item: item.reference_frame_index)
        pairs.extend((left, right) for left, right in zip(group, group[1:], strict=False)
                     if right.reference_frame_index == left.reference_frame_index + 1)
    return pairs


def _temporal_residual(renderer: GsplatRenderer, asset: CanonicalActorAsset, pair: tuple[DirectObservation, DirectObservation], deltas: Any, knots: tuple[int, ...]) -> Any:
    torch = renderer.torch
    left, right = pair
    left_rgb, _ = _render_joint(renderer, asset, left, _interpolated_correction(deltas, knots, left.reference_frame_index))
    right_rgb, _ = _render_joint(renderer, asset, right, _interpolated_correction(deltas, knots, right.reference_frame_index))
    left_target = torch.from_numpy(left.target_rgb.astype(np.float32)).to(renderer.means.device)
    right_target = torch.from_numpy(right.target_rgb.astype(np.float32)).to(renderer.means.device)
    mask = torch.from_numpy(left.vehicle_roi | right.vehicle_roi).to(renderer.means.device)
    return torch.sqrt(((right_rgb - left_rgb) - (right_target - left_target)).square() + 1e-6).mean(dim=-1)[mask].mean()


def _trajectory_pose(trajectory: ActorTrajectory, timestamp_us: int) -> tuple[np.ndarray, np.ndarray]:
    times = trajectory.timestamps_us
    right = min(max(int(np.searchsorted(times, timestamp_us, side="right")), 1), len(times) - 1)
    left = right - 1
    u = np.clip((timestamp_us - int(times[left])) / max(int(times[right] - times[left]), 1), 0.0, 1.0)
    rotation = (1.0 - u) * trajectory.actor_to_world[left, :3, :3] + u * trajectory.actor_to_world[right, :3, :3]
    left_u, _, left_v = np.linalg.svd(rotation)
    rotation = left_u @ left_v
    translation = (1.0 - u) * trajectory.actor_to_world[left, :3, 3] + u * trajectory.actor_to_world[right, :3, 3]
    return rotation.astype(np.float32), translation.astype(np.float32)


def _bake_corrected_trajectories(asset: CanonicalActorAsset, knots: tuple[int, ...], knot_timestamps: np.ndarray, deltas: np.ndarray) -> tuple[ActorTrajectory, ...]:
    import torch

    output: list[ActorTrajectory] = []
    for actor_index, trajectory in enumerate(asset.trajectories):
        times = np.unique(np.concatenate((trajectory.timestamps_us, knot_timestamps))).astype(np.int64)
        poses = np.tile(np.eye(4, dtype=np.float32), (len(times), 1, 1))
        for index, timestamp in enumerate(times):
            rotation, translation = _trajectory_pose(trajectory, int(timestamp))
            # The correction curve is parameterised by reference-frame index;
            # knot timestamps are only used to make it available to rendering.
            right = int(np.searchsorted(knot_timestamps, timestamp, side="left"))
            if right <= 0:
                correction = deltas[0, actor_index]
            elif right >= len(knot_timestamps):
                correction = deltas[-1, actor_index]
            else:
                left = right - 1
                u = (timestamp - knot_timestamps[left]) / max(int(knot_timestamps[right] - knot_timestamps[left]), 1)
                correction = (1.0 - u) * deltas[left, actor_index] + u * deltas[right, actor_index]
            delta_rotation = _quaternion_matrix(
                _axis_angle_quaternion(torch.from_numpy(correction[None, 3:]))
            ).detach().cpu().numpy()[0]
            poses[index, :3, :3] = delta_rotation @ rotation
            poses[index, :3, 3] = translation + correction[:3]
        output.append(ActorTrajectory(times, poses))
    return tuple(output)


def optimize_ftheta_actor_poses(options: FThetaActorPoseOptimizeOptions) -> CanonicalActorAsset:
    import torch

    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite actor output: {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(options.seed)
    static_scene = load_gaussian_ply(options.static_ply)
    initial = _subset_asset(CanonicalActorAsset.load(options.initial_actors), options.max_gaussians_per_actor)
    tracks = _load_tracks(options.actor_dataset_manifest, set(initial.actor_ids))
    renderer = GsplatRenderer(static_scene, RenderOptions(device=options.device, static_opacity_scale=1.0))
    initial_actor_renderer = GsplatRenderer(static_scene, RenderOptions(device=options.device, static_opacity_scale=0.0), canonical_actor_asset=initial)
    train, validation = _build_observations(options, renderer, initial_actor_renderer, initial, tracks)
    requested_knots = options.resolved_pose_knots
    if any(index not in set((*options.frame_indices, *options.validation_frame_indices)) for index in requested_knots):
        raise ValueError("pose knots must be included in the train or validation frame selection")
    observations_by_frame = {item.reference_frame_index: item for item in train + validation}
    skipped_knots = tuple(knot for knot in requested_knots if knot not in observations_by_frame)
    knots = tuple(knot for knot in requested_knots if knot in observations_by_frame)
    if len(knots) < 2:
        raise ValueError("fewer than two pose knots have a usable semantic actor observation")
    if skipped_knots:
        print(
            "Skipping pose knots without a semantic actor ROI; the correction curve is clamped at the nearest valid knot: "
            f"{list(skipped_knots)}",
            flush=True,
        )
    deltas = torch.zeros((len(knots), len(initial.actor_ids), 6), device=renderer.means.device, requires_grad=True)
    optimizer = torch.optim.Adam((deltas,), lr=options.learning_rate)
    train_pairs = _temporal_pairs(train)
    validation_pairs = _temporal_pairs(validation)
    history: list[dict[str, float]] = []
    best_score, best_step, best_deltas = float("inf"), None, None
    for step in range(options.iterations):
        observation = train[step % len(train)]
        correction = _interpolated_correction(deltas, knots, observation.reference_frame_index)
        prediction, alpha = _render_joint(renderer, initial, observation, correction)
        target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(renderer.means.device)
        roi = torch.from_numpy(observation.vehicle_roi).to(renderer.means.device)
        outside = torch.from_numpy(~observation.dilated_vehicle_roi).to(renderer.means.device)
        static_alpha = torch.from_numpy(observation.static_alpha).to(renderer.means.device)
        rgb = torch.sqrt((prediction - target).square() + 1e-6).mean(dim=-1)[roi].mean()
        outside_loss = alpha[outside].mean() if bool(outside.any()) else alpha.new_zeros(())
        occluded_mask = (
            (~observation.dilated_vehicle_roi) & observation.static_alpha
            & np.isfinite(observation.static_depth) & np.isfinite(observation.actor_initial_depth)
            & (observation.actor_initial_depth > observation.static_depth + options.static_occluder_depth_margin_m)
        )
        occluded = torch.from_numpy(occluded_mask).to(renderer.means.device)
        occluded_loss = alpha[occluded].mean() if bool(occluded.any()) else alpha.new_zeros(())
        static_leak = ((1.0 - alpha) * static_alpha.float())[roi].mean()
        temporal = _temporal_residual(renderer, initial, train_pairs[step % len(train_pairs)], deltas, knots) if train_pairs else alpha.new_zeros(())
        prior = deltas.square().mean()
        velocity = (deltas[1:] - deltas[:-1]).square().mean() if len(knots) > 1 else alpha.new_zeros(())
        acceleration = (deltas[2:] - 2.0 * deltas[1:-1] + deltas[:-2]).square().mean() if len(knots) > 2 else alpha.new_zeros(())
        loss = (options.rgb_weight * rgb + options.outside_alpha_weight * outside_loss
                + options.occluded_alpha_weight * occluded_loss + options.static_leak_weight * static_leak
                + options.temporal_rgb_weight * temporal + options.pose_prior_weight * prior
                + options.pose_velocity_weight * velocity + options.pose_acceleration_weight * acceleration)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _clip_pose_deltas(
            deltas, options.pose_translation_limit_m, np.deg2rad(options.pose_rotation_limit_deg)
        )
        record = {"step": float(step + 1), "loss": float(loss.detach()), "rgb": float(rgb.detach()), "temporal": float(temporal.detach()), "outside_alpha": float(outside_loss.detach()), "occluded_alpha": float(occluded_loss.detach()), "static_leak": float(static_leak.detach())}
        history.append(record)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == options.iterations:
            print(f"Pose FTheta optimize [{step + 1:05d}/{options.iterations}] frame={observation.reference_frame_index} loss={record['loss']:.6f} rgb={record['rgb']:.6f} temporal={record['temporal']:.6f} outside={record['outside_alpha']:.6f}", flush=True)
        if (step + 1) % options.checkpoint_every == 0 or step + 1 == options.iterations:
            with torch.no_grad():
                rgb_values = []
                for observation in validation:
                    pred, _ = _render_joint(renderer, initial, observation, _interpolated_correction(deltas, knots, observation.reference_frame_index))
                    target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(renderer.means.device)
                    roi = torch.from_numpy(observation.vehicle_roi).to(renderer.means.device)
                    rgb_values.append(float(torch.sqrt((pred-target).square()+1e-6).mean(dim=-1)[roi].mean()))
                temporal_values = [float(_temporal_residual(renderer, initial, pair, deltas, knots)) for pair in validation_pairs]
            val_rgb = float(np.mean(rgb_values)) if rgb_values else None
            val_temporal = float(np.mean(temporal_values)) if temporal_values else 0.0
            score = (val_rgb if val_rgb is not None else 0.0) + options.temporal_rgb_weight * val_temporal
            record.update(validation_rgb=val_rgb, validation_temporal=val_temporal, validation_score=score)
            print(f"  holdout_rgb={val_rgb!s} holdout_temporal={val_temporal:.6f} score={score:.6f}", flush=True)
            if validation and score < best_score:
                best_score, best_step, best_deltas = score, step + 1, deltas.detach().clone()
            torch.save({"step": step + 1, "pose_deltas": deltas.detach(), "best_score": best_score, "best_step": best_step, "best_deltas": best_deltas}, options.output.with_suffix(".checkpoint.pt"))
    selected = best_deltas if best_deltas is not None else deltas.detach()
    selected_np = selected.cpu().numpy().astype(np.float32)
    knot_times = np.asarray([_reference_midpoint_us(observations_by_frame[knot]) for knot in knots], dtype=np.int64)
    trajectories = _bake_corrected_trajectories(initial, knots, knot_times, selected_np)
    metadata = dict(initial.metadata)
    metadata.update({"optimizer_status": "pose_only_native_NCore_FTheta_rolling_shutter", "pose_only": True, "selected_step": best_step or options.iterations, "best_validation_score": best_score if best_deltas is not None else None, "requested_pose_knot_frame_indices": list(requested_knots), "pose_knot_frame_indices": list(knots), "skipped_pose_knot_frame_indices": list(skipped_knots), "pose_knot_timestamps_us": knot_times.tolist(), "pose_translation_limit_m": options.pose_translation_limit_m, "pose_rotation_limit_deg": options.pose_rotation_limit_deg, "temporal_rgb_weight": options.temporal_rgb_weight, "training_frame_indices": list(options.frame_indices), "validation_frame_indices": list(options.validation_frame_indices)})
    result = CanonicalActorAsset(initial.positions, initial.rotations_wxyz, initial.scales, initial.rgb, initial.opacities, initial.actor_indices, initial.visibility_counts, initial.source_counts, initial.actor_ids, trajectories, metadata)
    result.save(options.output)
    options.output.with_suffix(".report.json").write_text(json.dumps({"output": str(options.output), "history": history, "train_observations": [{"camera_id": x.camera_id, "frame": x.reference_frame_index} for x in train], "validation_observations": [{"camera_id": x.camera_id, "frame": x.reference_frame_index} for x in validation]}, indent=2), encoding="utf-8")
    print(f"Wrote pose-only FTheta corrected canonical actors: {options.output}", flush=True)
    return result
