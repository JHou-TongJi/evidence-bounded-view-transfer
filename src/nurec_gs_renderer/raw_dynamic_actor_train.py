"""Minimal raw-NCore rigid dynamic-layer trainer.

This deliberately starts a new actor layer from masks, RGB, LiDAR and cuboid
trajectories recorded in ``ncore-dynamic-layer-trusted-supervision``.  It does
not load a static PLY, a previous dynamic NPZ, or static alpha/depth.  The
first implementation is intentionally small: one canonical Gaussian shell per
rigid actor, exact source FTheta/rolling-shutter rendering, and grouped
train/validation/cross-view-holdout metrics.  It is a smoke-scale trainer,
not authorisation to produce a dynamic sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import interpolate_w2c
from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, ActorTrajectory, CanonicalActorAsset
from .ftheta_actor_optimize import _camera_kwargs, _rasterize_rgb_alpha, _world_actor_means
from .ply_io import GaussianScene
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .render import GsplatRenderer, RenderOptions, _write_json_atomic
from .sky_build import dilate_boolean_mask


SCHEMA = "ncore-raw-ftheta-rigid-actor-train"
VERSION = 1
TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_lidar_pixels(xy: np.ndarray, *, source_width: int, source_height: int, width: int, height: int) -> np.ndarray:
    """Map pixel centres with the same half-pixel convention as FTheta resize."""
    value = np.asarray(xy, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2 or min(source_width, source_height, width, height) <= 0:
        raise ValueError("xy must be [N,2] and image sizes positive")
    result = np.empty_like(value)
    result[:, 0] = np.rint((value[:, 0] + .5) * width / source_width - .5).astype(np.int64)
    result[:, 1] = np.rint((value[:, 1] + .5) * height / source_height - .5).astype(np.int64)
    return result


def box_surface_points(length_width_height: np.ndarray, spacing_m: float) -> np.ndarray:
    """Deterministic actor-local cuboid surface samples (no static geometry)."""
    dimensions = np.asarray(length_width_height, dtype=np.float32)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0) or spacing_m <= 0.0:
        raise ValueError("actor dimensions and spacing must be positive")
    axes = [np.arange(-value / 2, value / 2 + spacing_m * .5, spacing_m, dtype=np.float32) for value in dimensions]
    faces: list[np.ndarray] = []
    for fixed_axis in range(3):
        other = [axis for axis in range(3) if axis != fixed_axis]
        mesh = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            points = np.zeros((mesh[0].size, 3), dtype=np.float32)
            points[:, fixed_axis] = sign * dimensions[fixed_axis] / 2
            points[:, other[0]] = mesh[0].ravel()
            points[:, other[1]] = mesh[1].ravel()
            faces.append(points)
    merged = np.concatenate(faces, axis=0)
    rounded = np.round(merged / max(spacing_m * .1, 1e-5)).astype(np.int64)
    return merged[np.unique(rounded, axis=0, return_index=True)[1]]


@dataclass(frozen=True)
class RawDynamicActorTrainOptions:
    trusted_manifest: Path
    output: Path
    track_ids: tuple[int, ...]
    initial_actors: Path | None = None
    reference_start: int | None = None
    reference_end: int | None = None
    camera_ids: tuple[str, ...] | None = None
    temporal_even_train_odd_holdout: bool = False
    device: str = "cuda"
    width: int = 480
    height: int = 270
    iterations: int = 200
    learning_rate: float = 2.0e-3
    shell_spacing_m: float = .25
    max_gaussians_per_actor: int = 2_000
    mask_dilation_pixels: int = 4
    rgb_weight: float = 1.0
    alpha_weight: float = .20
    outside_alpha_weight: float = .10
    lidar_depth_weight: float = .05
    position_prior_weight: float = .02
    color_prior_weight: float = .01
    opacity_prior_weight: float = .002
    camera_bias_prior_weight: float = .02
    max_camera_bias: float = .08
    validation_every: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output.suffix != ".npz":
            raise ValueError("output must use .npz")
        if not self.track_ids or len(set(self.track_ids)) != len(self.track_ids) or any(value < 0 for value in self.track_ids):
            raise ValueError("track_ids must be unique non-negative values")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")
        if self.camera_ids is not None and (not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique when specified")
        if self.width <= 0 or self.height <= 0 or self.iterations < 0 or self.learning_rate <= 0:
            raise ValueError("resolution, iterations and learning rate are invalid")
        if self.shell_spacing_m <= 0 or self.max_gaussians_per_actor <= 0 or self.mask_dilation_pixels < 0:
            raise ValueError("shell budget/margins are invalid")
        if self.validation_every <= 0 or not 0 < self.max_camera_bias <= 1:
            raise ValueError("validation/camera-bias parameters are invalid")
        if any(value < 0 for value in (
            self.rgb_weight, self.alpha_weight, self.outside_alpha_weight, self.lidar_depth_weight,
            self.position_prior_weight, self.color_prior_weight, self.opacity_prior_weight, self.camera_bias_prior_weight,
        )):
            raise ValueError("loss weights must be non-negative")


@dataclass
class RawObservation:
    split: str
    camera_id: str
    reference_frame_index: int
    camera: Any
    target_rgb: np.ndarray
    mask: np.ndarray
    lidar_xy: np.ndarray
    lidar_depth_m: np.ndarray


def _dummy_renderer(device: str) -> GsplatRenderer:
    """Provide gsplat bindings/device without a static PLY or static layer."""
    scene = GaussianScene(
        means=np.asarray([[0.0, 0.0, -1_000.0]], np.float32), scales=np.asarray([[.01, .01, .01]], np.float32),
        quats_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), opacities=np.asarray([0.0], np.float32),
        sh_coeffs=np.zeros((1, 1, 3), np.float32), sh_degree=0,
    )
    return GsplatRenderer(scene, RenderOptions(device=device))


def _initial_asset(manifest: dict[str, Any], track_ids: tuple[int, ...], options: RawDynamicActorTrainOptions) -> CanonicalActorAsset:
    records = {int(value["track_id"]): value for value in manifest["tracks"]}
    selected = [records[value] for value in track_ids]
    positions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for actor_index, record in enumerate(selected):
        points = box_surface_points(np.asarray(record["length_width_height"], np.float32), options.shell_spacing_m)
        if len(points) > options.max_gaussians_per_actor:
            points = points[np.linspace(0, len(points) - 1, options.max_gaussians_per_actor, dtype=np.int64)]
        positions.append(points)
        indices.append(np.full(len(points), actor_index, np.int32))
    local = np.concatenate(positions, axis=0)
    trajectories = tuple(ActorTrajectory(
        np.asarray(record["timestamps_us"], np.int64), np.asarray(record["actor_to_world"], np.float32)
    ) for record in selected)
    count = len(local)
    asset = CanonicalActorAsset(
        positions=local, rotations_wxyz=np.tile(np.asarray([[1., 0., 0., 0.]], np.float32), (count, 1)),
        scales=np.full((count, 3), options.shell_spacing_m * .45, np.float32), rgb=np.full((count, 3), .5, np.float32),
        opacities=np.full(count, .20, np.float32), actor_indices=np.concatenate(indices),
        visibility_counts=np.zeros(count, np.uint16), source_counts=np.ones(count, np.uint16), actor_ids=track_ids,
        trajectories=trajectories,
        metadata={"schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION, "coordinate_frame": "actor_local",
                  "initialization": "raw_trusted_manifest_cuboid_surface_shell", "optimizer_status": "not_optimized"},
    )
    asset.validate()
    return asset


def temporal_split(reference_index: int, *, even_train_odd_holdout: bool, manifest_split: str) -> str:
    """Keep a strict temporal holdout independent of legacy manifest roles."""
    return ("train" if reference_index % 2 == 0 else "crossview_holdout") if even_train_odd_holdout else manifest_split


def _load_observations(manifest: dict[str, Any], manifest_path: Path, options: RawDynamicActorTrainOptions) -> tuple[list[RawObservation], Any]:
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    ncore_path = Path(manifest["ncore_path"])
    loader = _create_ncore_loader(ncore_path)
    selected = set(options.track_ids)
    observations: list[RawObservation] = []
    for frame in manifest["frames"]:
        reference_index = int(frame["reference_frame_index"])
        split = temporal_split(reference_index, even_train_odd_holdout=options.temporal_even_train_odd_holdout, manifest_split=str(frame["split"]))
        if options.reference_start is not None and reference_index < options.reference_start:
            continue
        if options.reference_end is not None and reference_index >= options.reference_end:
            continue
        for record in frame["cameras"]:
            if options.camera_ids is not None and str(record["camera_id"]) not in options.camera_ids:
                continue
            instances = [item for item in record["instances"] if int(item["track_id"]) in selected]
            if not instances:
                continue
            # A future multi-actor trainer may accept overlaps; the first smoke
            # requires one selected instance per view so mask/RGB ownership is explicit.
            if len(instances) != 1:
                raise ValueError(f"raw trainer requires one selected actor per observation: {record['camera_id']} ref={frame['reference_frame_index']}")
            instance = instances[0]
            role = str(instance["supervision_role"])
            if not options.temporal_even_train_odd_holdout and split == "crossview_holdout" and role not in {"crossview_holdout_source", "crossview_holdout_sam2"}:
                raise ValueError("crossview split contains an invalid supervision role")
            if split != "crossview_holdout" and instance.get("mask_origin") != "source":
                raise ValueError("only source masks may be train/validation supervision")
            camera_id = str(record["camera_id"])
            source_index = int(record["source_frame_index"])
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [source_index], "output": {"width": options.width, "height": options.height}},
                loader, camera_id,
            ).frames[0]
            if int(camera.timestamp_start_us) != int(record["timestamp_start_us"]) or int(camera.timestamp_us) != int(record["timestamp_end_us"]):
                raise ValueError(f"NCore exposure no longer matches trusted manifest: {camera_id} frame={source_index}")
            sensor = loader.get_camera_sensor(camera_id)
            raw = np.asarray(sensor.get_frame_image_array(source_index), np.uint8)
            rgb = np.asarray(Image.fromarray(raw, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR), np.float32) / 255.
            root = roots[str(instance["mask_root"])]
            mask_native = np.asarray(Image.open(root / str(instance["mask"])).convert("L"), np.uint8) > 0
            if mask_native.shape != raw.shape[:2]:
                raise ValueError(f"mask/image shape mismatch: {root / str(instance['mask'])}")
            mask = np.asarray(Image.fromarray(mask_native.astype(np.uint8) * 255).resize((options.width, options.height), Image.Resampling.NEAREST), np.uint8) > 0
            # The source dataset admits masks as small as 24 native pixels.
            # At a smoke resolution a thin distant instance can disappear
            # entirely under nearest-neighbour resize; an empty reduction
            # would create a NaN loss and must not masquerade as evidence.
            if not mask.any():
                continue
            lidar_xy, lidar_depth = np.empty((0, 2), np.int64), np.empty(0, np.float32)
            depth_rel = instance.get("lidar_depth")
            if isinstance(depth_rel, str):
                with np.load(roots[str(instance["lidar_depth_root"])] / depth_rel, allow_pickle=False) as data:
                    xy, depth = np.asarray(data["xy"], np.int64), np.asarray(data["depth_m"], np.float32)
                xy = resize_lidar_pixels(xy, source_width=raw.shape[1], source_height=raw.shape[0], width=options.width, height=options.height)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < options.width) & (xy[:, 1] >= 0) & (xy[:, 1] < options.height) & np.isfinite(depth) & (depth > .1)
                lidar_xy, lidar_depth = xy[inside], depth[inside]
            observations.append(RawObservation(
                split=split, camera_id=camera_id, reference_frame_index=reference_index, camera=camera,
                target_rgb=rgb, mask=mask, lidar_xy=lidar_xy, lidar_depth_m=lidar_depth,
            ))
    if not any(item.split == "train" for item in observations):
        raise RuntimeError("trusted manifest has no source training observations")
    if not any(item.split == "crossview_holdout" for item in observations):
        raise RuntimeError("trusted manifest has no grouped cross-view holdout observations")
    return observations, loader


def _world_depth(renderer: GsplatRenderer, camera: Any, means: Any, quats: Any, scales: Any, opacities: Any) -> tuple[Any, Any]:
    """Differentiable expected camera-z via the validated eval3d world pass."""
    torch = renderer.torch
    kwargs, packed = _camera_kwargs(renderer, camera)
    common = dict(means=means, quats=quats, scales=scales, opacities=opacities,
                  viewmats=torch.from_numpy(camera.world_to_camera).to(renderer.means.device)[None],
                  Ks=torch.from_numpy(camera.intrinsics.K).to(renderer.means.device)[None],
                  width=camera.intrinsics.width, height=camera.intrinsics.height, packed=packed,
                  near_plane=renderer.options.near_plane, far_plane=renderer.options.far_plane,
                  backgrounds=None, rasterize_mode=renderer.options.rasterize_mode)
    extra = {"with_eval3d": True} if camera.ftheta is not None and camera.ftheta.is_rolling else {}
    world, alpha, _ = renderer.rasterization(colors=means, sh_degree=None, render_mode="RGB", **extra, **common, **kwargs)
    alpha = alpha[0, ..., 0].clamp(1e-6, 1.)
    expected = world[0] / alpha[..., None]
    height, width = alpha.shape
    shutter = camera.ftheta.shutter_type if camera.ftheta is not None else "GLOBAL"
    vertical = shutter in {"GLOBAL", "ROLLING_TOP_TO_BOTTOM", "ROLLING_BOTTOM_TO_TOP"}
    count = height if vertical else width
    times = np.zeros(count, np.float64) if count == 1 else np.arange(count, dtype=np.float64) / (count - 1)
    if shutter in {"ROLLING_BOTTOM_TO_TOP", "ROLLING_RIGHT_TO_LEFT"}:
        times = 1. - times
    end_w2c = camera.world_to_camera if camera.world_to_camera_end is None else camera.world_to_camera_end
    rotations, translations = interpolate_w2c(camera.world_to_camera, end_w2c, times)
    if vertical:
        row = torch.from_numpy(rotations[:, 2, :].astype(np.float32)).to(expected.device)
        offset = torch.from_numpy(translations[:, 2].astype(np.float32)).to(expected.device)
        depth = (expected * row[:, None]).sum(dim=-1) + offset[:, None]
    else:
        col = torch.from_numpy(rotations[:, 2, :].astype(np.float32)).to(expected.device)
        offset = torch.from_numpy(translations[:, 2].astype(np.float32)).to(expected.device)
        depth = (expected * col[None]).sum(dim=-1) + offset[None]
    return depth, alpha


def _metrics(
    observations: list[RawObservation], asset: CanonicalActorAsset, local: Any, rgb: Any, opacity: Any,
    biases: Any, camera_index: dict[str, int], renderer: GsplatRenderer, *, temporal_reference_step: int = 1,
) -> dict[str, float] | None:
    if not observations:
        return None
    torch = renderer.torch
    if temporal_reference_step <= 0:
        raise ValueError("temporal_reference_step must be positive")
    values: list[tuple[float, float]] = []
    lidar_errors: list[float] = []
    lidar_sample_count = 0
    with torch.no_grad():
        for observation in observations:
            means, quats = _world_actor_means(asset, local, observation.camera, torch=torch)
            colors = ((rgb + biases[camera_index[observation.camera_id]].tanh() * .08).clamp(0., 1.) - .5) / .28209479177387814
            rendered, alpha = _rasterize_rgb_alpha(renderer, observation.camera, means, quats, torch.from_numpy(asset.scales).to(means.device), opacity, colors[:, None])
            target = torch.from_numpy(observation.target_rgb).to(means.device)
            mask = torch.from_numpy(observation.mask).to(means.device)
            rgb_value = torch.sqrt(((rendered / alpha[..., None].clamp_min(.05)).clamp(0., 1.) - target).square() + 1e-6).mean(dim=-1)[mask].mean()
            alpha_value = (alpha[mask] - 1.).abs().mean()
            values.append((float(rgb_value), float(alpha_value)))
            if len(observation.lidar_xy):
                depth, _ = _world_depth(
                    renderer, observation.camera, means, quats, torch.from_numpy(asset.scales).to(means.device), opacity
                )
                xy = torch.from_numpy(observation.lidar_xy).to(means.device)
                target_depth = torch.from_numpy(observation.lidar_depth_m).to(means.device)
                predicted_depth = depth[xy[:, 1], xy[:, 0]]
                finite = torch.isfinite(predicted_depth) & torch.isfinite(target_depth)
                if bool(finite.any()):
                    lidar_errors.extend((predicted_depth[finite] - target_depth[finite]).abs().detach().cpu().tolist())
                    lidar_sample_count += int(finite.sum())
    result = {"rgb_mae": float(np.mean([value[0] for value in values])), "alpha_error": float(np.mean([value[1] for value in values])), "observation_count": float(len(values)),
              "lidar_depth_mae_m": float(np.mean(lidar_errors)) if lidar_errors else float("nan"),
              "lidar_depth_sample_count": float(lidar_sample_count)}
    # A one-frame alpha score can conceal a source actor that flickers or
    # jumps.  Compare only consecutive reference frames from the same camera;
    # the target is the delta of trusted masks, not a static-composited RGB.
    temporal: list[float] = []
    by_camera: dict[str, list[RawObservation]] = {}
    for item in observations:
        by_camera.setdefault(item.camera_id, []).append(item)
    with torch.no_grad():
        for camera_items in by_camera.values():
            camera_items.sort(key=lambda item: item.reference_frame_index)
            for earlier, later in zip(camera_items, camera_items[1:]):
                if later.reference_frame_index != earlier.reference_frame_index + temporal_reference_step:
                    continue
                def alpha(item: RawObservation):
                    means, quats = _world_actor_means(asset, local, item.camera, torch=torch)
                    colors = ((rgb + biases[camera_index[item.camera_id]].tanh() * .08).clamp(0., 1.) - .5) / .28209479177387814
                    _, value = _rasterize_rgb_alpha(renderer, item.camera, means, quats, torch.from_numpy(asset.scales).to(means.device), opacity, colors[:, None])
                    return value
                predicted = alpha(later) - alpha(earlier)
                target_delta = torch.from_numpy(later.mask.astype(np.float32) - earlier.mask.astype(np.float32)).to(predicted.device)
                region = torch.from_numpy(dilate_boolean_mask(later.mask | earlier.mask, 3)).to(predicted.device)
                if bool(region.any()):
                    temporal.append(float((predicted[region] - target_delta[region]).abs().mean()))
    result["temporal_alpha_delta_mae"] = float(np.mean(temporal)) if temporal else float("nan")
    result["temporal_pair_count"] = float(len(temporal))
    result["temporal_reference_step"] = float(temporal_reference_step)
    return result


def train_raw_dynamic_actor(options: RawDynamicActorTrainOptions) -> CanonicalActorAsset:
    """Run a bounded exact-FTheta/rolling optimisation from trusted raw supervision."""
    import torch

    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite actor asset: {options.output}")
    manifest_path = options.trusted_manifest.resolve()
    manifest = _json(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted_manifest must be complete trusted raw supervision")
    selected_tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    if set(options.track_ids).difference(selected_tracks):
        raise ValueError("requested track is absent from trusted manifest")
    if any(selected_tracks[value].get("actor_family") != "rigid" for value in options.track_ids):
        raise ValueError("raw trainer currently supports rigid actors only")
    torch.manual_seed(options.seed); np.random.seed(options.seed)
    if options.initial_actors is None:
        asset = _initial_asset(manifest, options.track_ids, options)
    else:
        asset = CanonicalActorAsset.load(options.initial_actors)
        if asset.actor_ids != options.track_ids:
            raise ValueError("initial actor IDs must exactly match --track-ids")
        if asset.metadata.get("trusted_manifest_sha256") != _sha256(manifest_path):
            raise ValueError("initial actor asset was not built from this trusted manifest")
    observations, _ = _load_observations(manifest, manifest_path, options)
    train = [item for item in observations if item.split == "train"]
    validation = [item for item in observations if item.split == "validation"]
    holdout = [item for item in observations if item.split == "crossview_holdout"]
    temporal_reference_step = 2 if options.temporal_even_train_odd_holdout else 1
    renderer = _dummy_renderer(options.device)
    device = renderer.means.device
    base_local = torch.from_numpy(asset.positions).to(device)
    base_rgb = torch.from_numpy(asset.rgb).to(device)
    base_logit = torch.logit(torch.from_numpy(asset.opacities).to(device).clamp(1e-4, 1 - 1e-4))
    local_delta = torch.zeros_like(base_local, requires_grad=True)
    rgb_delta = torch.zeros_like(base_rgb, requires_grad=True)
    opacity_delta = torch.zeros_like(base_logit, requires_grad=True)
    camera_ids = tuple(sorted({item.camera_id for item in observations}))
    camera_index = {value: index for index, value in enumerate(camera_ids)}
    biases = torch.zeros((len(camera_ids), 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((local_delta, rgb_delta, opacity_delta, biases), lr=options.learning_rate)
    best = (float("inf"), None)
    history: list[dict[str, float]] = []
    for step in range(options.iterations):
        observation = train[step % len(train)]
        local = base_local + .20 * local_delta.tanh()
        rgb = base_rgb + .35 * rgb_delta.tanh()
        opacity = torch.sigmoid(base_logit + opacity_delta)
        means, quats = _world_actor_means(asset, local, observation.camera, torch=torch)
        colors = ((rgb + options.max_camera_bias * biases[camera_index[observation.camera_id]].tanh()).clamp(0., 1.) - .5) / .28209479177387814
        rendered, alpha = _rasterize_rgb_alpha(renderer, observation.camera, means, quats, torch.from_numpy(asset.scales).to(device), opacity, colors[:, None])
        target = torch.from_numpy(observation.target_rgb).to(device)
        mask = torch.from_numpy(observation.mask).to(device)
        dilated = torch.from_numpy(dilate_boolean_mask(observation.mask, options.mask_dilation_pixels)).to(device)
        rgb_loss = torch.sqrt(((rendered / alpha[..., None].clamp_min(.05)).clamp(0., 1.) - target).square() + 1e-6).mean(dim=-1)[mask].mean()
        alpha_loss = (alpha[mask] - 1.).square().mean()
        outside_loss = alpha[~dilated].mean()
        depth_loss = alpha.new_zeros(())
        if len(observation.lidar_xy):
            depth, _ = _world_depth(renderer, observation.camera, means, quats, torch.from_numpy(asset.scales).to(device), opacity)
            xy = torch.from_numpy(observation.lidar_xy).to(device)
            target_depth = torch.from_numpy(observation.lidar_depth_m).to(device)
            depth_loss = torch.sqrt((depth[xy[:, 1], xy[:, 0]] - target_depth).square() + .01).mean()
        loss = (options.rgb_weight * rgb_loss + options.alpha_weight * alpha_loss + options.outside_alpha_weight * outside_loss
                + options.lidar_depth_weight * depth_loss + options.position_prior_weight * local_delta.square().mean()
                + options.color_prior_weight * rgb_delta.square().mean() + options.opacity_prior_weight * opacity_delta.square().mean()
                + options.camera_bias_prior_weight * biases.square().mean())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 0 or (step + 1) % options.validation_every == 0 or step + 1 == options.iterations:
            local_eval = base_local + .20 * local_delta.tanh(); rgb_eval = base_rgb + .35 * rgb_delta.tanh(); opacity_eval = torch.sigmoid(base_logit + opacity_delta)
            validation_metrics = _metrics(validation, asset, local_eval, rgb_eval, opacity_eval, biases, camera_index, renderer, temporal_reference_step=temporal_reference_step)
            score = float(validation_metrics["rgb_mae"]) if validation_metrics is not None else float(loss.detach())
            history.append({"step": float(step + 1), "loss": float(loss.detach()), "rgb": float(rgb_loss.detach()), "alpha": float(alpha_loss.detach()), "lidar": float(depth_loss.detach()), "validation_rgb_mae": score})
            if score < best[0]:
                best = (score, (local_eval.detach().clone(), rgb_eval.detach().clone(), opacity_eval.detach().clone()))
            print(f"Raw FTheta actor [{step + 1:05d}/{options.iterations}] loss={float(loss):.6f} rgb={float(rgb_loss):.6f} alpha={float(alpha_loss):.6f} lidar={float(depth_loss):.6f} val={score:.6f}", flush=True)
    local, rgb, opacity = best[1] if best[1] is not None else (base_local, base_rgb, torch.sigmoid(base_logit))
    result = CanonicalActorAsset(
        positions=local.detach().cpu().numpy().astype(np.float32), rotations_wxyz=asset.rotations_wxyz, scales=asset.scales,
        rgb=rgb.clamp(0., 1.).detach().cpu().numpy().astype(np.float32), opacities=opacity.detach().cpu().numpy().astype(np.float32),
        actor_indices=asset.actor_indices, visibility_counts=asset.visibility_counts, source_counts=asset.source_counts,
        actor_ids=asset.actor_ids, trajectories=asset.trajectories,
        metadata={"schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION, "coordinate_frame": "actor_local",
                  "optimizer_status": "raw_ncore_ftheta_rolling_smoke", "trusted_manifest": str(manifest_path),
                  "trusted_manifest_sha256": _sha256(manifest_path), "static_ply_used": False,
                  "initial_actors": None if options.initial_actors is None else str(options.initial_actors.resolve()),
                  "train_observations": len(train), "validation_observations": len(validation), "crossview_holdout_observations": len(holdout),
                  "reference_range": [options.reference_start, options.reference_end],
                  "camera_ids": None if options.camera_ids is None else list(options.camera_ids),
                  "temporal_even_train_odd_holdout": options.temporal_even_train_odd_holdout,
                  "best_validation_rgb_mae": best[0]},
    )
    result.save(options.output)
    local_eval = torch.from_numpy(result.positions).to(device); rgb_eval = torch.from_numpy(result.rgb).to(device); opacity_eval = torch.from_numpy(result.opacities).to(device)
    report = {"schema": SCHEMA, "version": VERSION, "trusted_manifest": str(manifest_path), "output": str(options.output.resolve()),
              "parameters": {key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value) for key, value in vars(options).items()},
              "train": _metrics(train, result, local_eval, rgb_eval, opacity_eval, biases, camera_index, renderer, temporal_reference_step=temporal_reference_step),
              "validation": _metrics(validation, result, local_eval, rgb_eval, opacity_eval, biases, camera_index, renderer, temporal_reference_step=temporal_reference_step),
              "crossview_holdout": _metrics(holdout, result, local_eval, rgb_eval, opacity_eval, biases, camera_index, renderer, temporal_reference_step=temporal_reference_step),
              "history": history,
              "limitations": ["No static PLY, static RGB, static alpha or static depth was loaded.", "This is a rigid cuboid-surface smoke trainer, not a full view-dependent/deformable dynamic reconstruction method.", "Temporal holdout observations are never back-propagated; they are not used for early stopping when no validation split exists.", "Do not composite or render a sequence until holdout and temporal audits are implemented."],}
    _write_json_atomic(report, options.output.with_suffix(".report.json"))
    print(f"Wrote raw FTheta canonical actor: {options.output}", flush=True)
    return result
