from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from .color_correction import GaussianColorCorrection, SH_C0, file_sha256
from .ply_io import GaussianScene
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .render import GsplatRenderer
from .sky import SkyCubemap, pad_sky_edges, sample_sky_cubemap
from .sky_build import (
    DEFAULT_CAMERA_IDS,
    SegformerSkySegmenter,
    _load_ego_mask,
    _sequence_sampling_interval,
    dilate_boolean_mask,
    sample_chunk_frame_indices,
)


@dataclass(frozen=True)
class GaussianDecontaminationOptions:
    ply_path: Path
    ncore_path: Path
    chunk_index: int
    model_dir: Path
    sky_cubemap_path: Path
    output: Path
    camera_ids: tuple[str, ...] = ("camera_front_wide_120fov",)
    slots: tuple[int, ...] = (0,)
    validation_slots: tuple[int, ...] = ()
    device: str = "cuda"
    width: int = 480
    height: int = 270
    boundary_pixels: int = 5
    sky_probability_threshold: float = 0.50
    sky_margin_threshold: float = 0.0
    geometry_alpha_threshold: float = 0.05
    halo_min_luminance_excess: float = 0.03
    halo_max_sky_color_distance: float = 0.30
    candidate_fraction: float = 0.01
    per_view_candidate_fraction: float = 0.02
    min_view_support: int = 1
    iterations: int = 200
    learning_rate: float = 0.03
    max_color_delta: float = 0.25
    color_prior_weight: float = 0.05
    static_preservation_weight: float = 0.20
    sky_edge_padding_pixels: int = 8
    sky_edge_smoothing_iterations: int = 16
    cache_dir: Path | None = None
    preview_views: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output.suffix.lower() != ".npz":
            raise ValueError("decontamination output must use the .npz extension")
        if self.chunk_index < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("chunk index and training resolution are invalid")
        if not self.camera_ids or not self.slots:
            raise ValueError("at least one camera and slot are required")
        if any(slot < 0 or slot >= 18 for slot in (*self.slots, *self.validation_slots)):
            raise ValueError("decontamination slots must be in [0,17]")
        if set(self.slots).intersection(self.validation_slots):
            raise ValueError("training and validation slots must not overlap")
        if self.boundary_pixels <= 0:
            raise ValueError("boundary_pixels must be positive")
        if not 0.0 < self.sky_probability_threshold < 1.0:
            raise ValueError("sky probability threshold must be in (0,1)")
        if not -1.0 < self.sky_margin_threshold < 1.0:
            raise ValueError("sky margin threshold must be in (-1,1)")
        if not 0.0 < self.geometry_alpha_threshold <= 1.0:
            raise ValueError("geometry alpha threshold must be in (0,1]")
        if self.halo_min_luminance_excess < 0.0:
            raise ValueError("halo luminance excess must be non-negative")
        if not 0.0 < self.halo_max_sky_color_distance <= 1.0:
            raise ValueError("halo sky color distance must be in (0,1]")
        if not 0.0 < self.candidate_fraction <= 1.0:
            raise ValueError("candidate_fraction must be in (0,1]")
        if not 0.0 < self.per_view_candidate_fraction <= 1.0:
            raise ValueError("per_view_candidate_fraction must be in (0,1]")
        if self.min_view_support <= 0 or self.iterations < 0:
            raise ValueError("support and optimization iteration parameters are invalid")
        if self.learning_rate <= 0.0 or not 0.0 < self.max_color_delta <= 1.0:
            raise ValueError("optimization learning rate or color bound is invalid")
        if self.color_prior_weight < 0.0 or self.static_preservation_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.sky_edge_padding_pixels < 0 or self.sky_edge_smoothing_iterations < 0:
            raise ValueError("sky edge filling parameters must be non-negative")
        if self.preview_views < 0:
            raise ValueError("preview_views must be non-negative")


@dataclass
class SourceObservation:
    camera_id: str
    slot: int
    frame_index: int
    camera: Any
    target_rgb: np.ndarray
    boundary_mask: np.ndarray
    static_mask: np.ndarray
    sky_fill: np.ndarray
    baseline_rgb: np.ndarray
    exposure_gain: float


def _differentiable_rgb_alpha(
    renderer: GsplatRenderer,
    camera: Any,
    colors: Any,
) -> tuple[Any, Any]:
    """Render RGB/alpha without inference mode so DC colors remain differentiable."""
    torch = renderer.torch
    intrinsics = camera.intrinsics
    viewmat = torch.from_numpy(camera.world_to_camera).to(renderer.means.device)[None]
    K = torch.from_numpy(intrinsics.K).to(renderer.means.device)[None]
    camera_kwargs: dict[str, object] = {"camera_model": camera.camera_model}
    packed = renderer.options.packed
    if camera.ftheta is not None:
        ftheta = camera.ftheta
        camera_kwargs.update(
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
            camera_kwargs["viewmats_rs"] = torch.from_numpy(camera.world_to_camera_end).to(
                renderer.means.device
            )[None]
    common_kwargs = dict(
        means=renderer.means,
        quats=renderer.quats,
        scales=renderer.scales,
        opacities=renderer.opacities,
        viewmats=viewmat,
        Ks=K,
        width=intrinsics.width,
        height=intrinsics.height,
        near_plane=renderer.options.near_plane,
        far_plane=renderer.options.far_plane,
        packed=packed,
        backgrounds=None,
        rasterize_mode=renderer.options.rasterize_mode,
    )
    extra = {"with_eval3d": True} if camera.ftheta is not None and camera.ftheta.is_rolling else {}
    rendered, alpha, _ = renderer.rasterization(
        colors=colors,
        sh_degree=renderer.sh_degree,
        render_mode="RGB",
        **extra,
        **common_kwargs,
        **camera_kwargs,
    )
    return rendered[0, ..., :3], alpha[0, ..., 0].clamp(0.0, 1.0)


def _segmentation_scores(
    segmenter: SegformerSkySegmenter,
    image: Image.Image,
    cache_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["sky_probability"], dtype=np.float32),
                np.asarray(archive["sky_margin"], dtype=np.float32),
            )
    probability, margin = segmenter.sky_scores(image)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            sky_probability=probability.astype(np.float16),
            sky_margin=margin.astype(np.float16),
        )
    return probability, margin


def _fit_exposure_gain(baseline: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    baseline_luma = baseline @ luminance_weights
    target_luma = target @ luminance_weights
    valid = mask & (baseline_luma > 0.03) & (target_luma > 0.03)
    if int(valid.sum()) < 512:
        return 1.0
    ratio = baseline_luma[valid] / np.maximum(target_luma[valid], 1e-4)
    return float(np.clip(np.median(ratio), 0.80, 1.25))


def select_halo_pixels(
    semantic_boundary: np.ndarray,
    alpha: np.ndarray,
    premultiplied_rgb: np.ndarray,
    sky_fill: np.ndarray,
    baseline_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    geometry_alpha_threshold: float,
    min_luminance_excess: float,
    max_sky_color_distance: float,
) -> np.ndarray:
    """Select only bright, sky-colored geometry residuals inside a semantic trimap."""
    boundary = np.asarray(semantic_boundary, dtype=bool)
    alpha = np.asarray(alpha, dtype=np.float32)
    premultiplied = np.asarray(premultiplied_rgb, dtype=np.float32)
    sky = np.asarray(sky_fill, dtype=np.float32)
    baseline = np.asarray(baseline_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    if (
        boundary.shape != alpha.shape
        or premultiplied.shape != boundary.shape + (3,)
        or sky.shape != premultiplied.shape
        or baseline.shape != premultiplied.shape
        or target.shape != premultiplied.shape
    ):
        raise ValueError("halo selection inputs have inconsistent shapes")
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    foreground = premultiplied / np.maximum(alpha[..., None], 1e-5)
    foreground = np.clip(foreground, 0.0, 1.0)
    sky_distance = np.abs(foreground - sky).mean(axis=-1)
    luminance_excess = baseline @ weights - target @ weights
    return (
        boundary
        & (alpha >= geometry_alpha_threshold)
        & (sky_distance <= max_sky_color_distance)
        & (luminance_excess >= min_luminance_excess)
    )


def _resolve_source_views(
    options: GaussianDecontaminationOptions,
    renderer: GsplatRenderer,
    sky_cubemap: SkyCubemap,
) -> list[SourceObservation]:
    try:
        import ncore.data
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Gaussian decontamination requires nvidia-ncore and PyTorch") from exc

    loader = _create_ncore_loader(options.ncore_path)
    available = {str(camera_id) for camera_id in loader.camera_ids}
    missing = set(options.camera_ids).difference(available)
    if missing:
        raise ValueError(f"NCore sequence is missing requested cameras: {sorted(missing)}")
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in options.camera_ids}
    timestamps = {
        camera_id: np.asarray(
            sensor.get_frames_timestamps_us(ncore.data.FrameTimepoint.END), dtype=np.int64
        )
        for camera_id, sensor in sensors.items()
    }
    sequence_start, sequence_end = _sequence_sampling_interval(loader, sensors, timestamps)
    sampled, _, _ = sample_chunk_frame_indices(
        timestamps,
        sequence_start,
        sequence_end,
        options.chunk_index,
        n_frames_per_chunk=18,
        max_frame_gap_timestamp_us=750_000,
    )
    segmenter = SegformerSkySegmenter(options.model_dir, options.device)
    observations: list[SourceObservation] = []
    total = len(options.camera_ids) * len(options.slots)
    processed = 0
    for camera_id in options.camera_ids:
        sensor = sensors[camera_id]
        ego_mask = _load_ego_mask(sensor, (options.width, options.height))
        for slot in options.slots:
            started = time.perf_counter()
            frame_index = sampled[camera_id][slot]
            config = {
                "frame_indices": [frame_index],
                "output": {"width": options.width, "height": options.height},
            }
            camera = _load_ncore_source_camera_path(config, loader, camera_id).frames[0]
            source = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
            image = Image.fromarray(source, mode="RGB").resize(
                (options.width, options.height), resample=Image.Resampling.BILINEAR
            )
            target = np.asarray(image, dtype=np.float32) / 255.0
            cache_path = None
            if options.cache_dir is not None:
                cache_path = (
                    options.cache_dir
                    / "segformer"
                    / f"{options.width}x{options.height}"
                    / camera_id
                    / f"{frame_index:06d}.npz"
                )
            probability, margin = _segmentation_scores(segmenter, image, cache_path)
            sky = (
                (probability >= options.sky_probability_threshold)
                & (margin >= options.sky_margin_threshold)
                & ~ego_mask
            )
            boundary = (
                dilate_boolean_mask(sky, options.boundary_pixels)
                & dilate_boolean_mask(~sky & ~ego_mask, options.boundary_pixels)
                & ~ego_mask
            )
            sky_rgb, sky_valid = sample_sky_cubemap(
                sky_cubemap,
                camera,
                confidence_threshold=renderer.options.sky_confidence_threshold,
                min_sample_count=renderer.options.sky_min_sample_count,
            )
            sky_rgb, sky_valid = pad_sky_edges(
                sky_rgb,
                sky_valid,
                options.sky_edge_padding_pixels,
                valid_threshold=renderer.options.sky_valid_threshold,
                smoothing_iterations=options.sky_edge_smoothing_iterations,
            )
            background = np.asarray(renderer.options.background, dtype=np.float32)
            sky_fill = sky_valid[..., None] * sky_rgb + (1.0 - sky_valid[..., None]) * background
            boundary &= dilate_boolean_mask(sky_valid >= renderer.options.sky_valid_threshold, 1)
            with torch.no_grad():
                premultiplied, alpha = _differentiable_rgb_alpha(renderer, camera, renderer.colors)
                fill_tensor = torch.from_numpy(sky_fill).to(premultiplied.device)
                baseline = (premultiplied + (1.0 - alpha[..., None]) * fill_tensor).clamp(0.0, 1.0)
            premultiplied_np = premultiplied.float().cpu().numpy()
            baseline_np = baseline.float().cpu().numpy()
            alpha_np = alpha.float().cpu().numpy()
            static_mask = (
                (alpha_np >= 0.50)
                & ~dilate_boolean_mask(boundary, options.boundary_pixels * 2)
                & ~ego_mask
            )
            gain = _fit_exposure_gain(baseline_np, target, static_mask)
            target = np.clip(target * gain, 0.0, 1.0)
            halo_mask = select_halo_pixels(
                boundary,
                alpha_np,
                premultiplied_np,
                sky_fill,
                baseline_np,
                target,
                geometry_alpha_threshold=options.geometry_alpha_threshold,
                min_luminance_excess=options.halo_min_luminance_excess,
                max_sky_color_distance=options.halo_max_sky_color_distance,
            )
            if not halo_mask.any():
                print(
                    f"Skipping camera={camera_id} slot={slot}: no confident bright sky-colored halo",
                    flush=True,
                )
                continue
            observations.append(
                SourceObservation(
                    camera_id=camera_id,
                    slot=slot,
                    frame_index=frame_index,
                    camera=camera,
                    target_rgb=target.astype(np.float16),
                    boundary_mask=halo_mask,
                    static_mask=static_mask,
                    sky_fill=sky_fill.astype(np.float16),
                    baseline_rgb=baseline_np.astype(np.float16),
                    exposure_gain=gain,
                )
            )
            processed += 1
            print(
                f"[{processed:03d}/{total}] camera={camera_id} slot={slot:02d} frame={frame_index:03d} "
                f"halo={int(halo_mask.sum()):6d} exposure={gain:.3f} "
                f"time={time.perf_counter() - started:.2f}s",
                flush=True,
            )
    if not observations:
        raise RuntimeError("no usable source observations were constructed")
    return observations


def _load_semantic_exclusion_masks(path: Path, gaussian_count: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - base dependency
        raise RuntimeError("Gaussian decontamination requires plyfile") from exc
    vertex = PlyData.read(str(path), mmap=True)["vertex"].data
    if len(vertex) != gaussian_count:
        raise ValueError("PLY semantic attribute count does not match scene")
    names = set(vertex.dtype.names or ())
    road = np.asarray(vertex["road_mask"], dtype=np.uint8) > 0 if "road_mask" in names else np.zeros(gaussian_count, bool)
    sky = np.asarray(vertex["sky_mask"], dtype=np.float32) > 0.5 if "sky_mask" in names else np.zeros(gaussian_count, bool)
    return road, sky


def attribute_boundary_gaussians(
    renderer: GsplatRenderer,
    observations: list[SourceObservation],
    options: GaussianDecontaminationOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = renderer.torch
    gaussian_count = int(renderer.colors.shape[0])
    aggregate = torch.zeros(gaussian_count, device=renderer.colors.device, dtype=torch.float32)
    support = torch.zeros(gaussian_count, device=renderer.colors.device, dtype=torch.int32)
    top_per_view = max(1, int(round(gaussian_count * options.per_view_candidate_fraction)))
    for view_index, observation in enumerate(observations):
        colors = renderer.colors.detach().clone().requires_grad_(True)
        premultiplied, alpha = _differentiable_rgb_alpha(renderer, observation.camera, colors)
        fill = torch.from_numpy(observation.sky_fill.astype(np.float32)).to(premultiplied.device)
        target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(premultiplied.device)
        mask = torch.from_numpy(observation.boundary_mask).to(premultiplied.device)
        prediction = (premultiplied + (1.0 - alpha[..., None]) * fill).clamp(0.0, 1.0)
        residual = torch.sqrt((prediction - target).square() + 1e-6).mean(dim=-1)
        loss = residual[mask].mean()
        loss.backward()
        gradient = colors.grad[:, 0].norm(dim=-1)
        positive = int((gradient > 0).sum().item())
        count = min(top_per_view, positive)
        if count == 0:
            print(f"Attribution view {view_index}: no non-zero color gradients", flush=True)
            continue
        values, indices = torch.topk(gradient, count, sorted=False)
        scale = values.median().clamp_min(1e-12)
        aggregate.index_add_(0, indices, values / scale)
        support[indices] += 1
        print(
            f"Attribution [{view_index + 1:03d}/{len(observations)}] "
            f"loss={float(loss):.6f} positive={positive:,} selected={count:,}",
            flush=True,
        )
        del colors, premultiplied, alpha, prediction, residual, loss, gradient, values, indices
        torch.cuda.empty_cache() if renderer.colors.is_cuda else None

    score = aggregate.cpu().numpy()
    support_np = support.clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
    road, sky = _load_semantic_exclusion_masks(options.ply_path, gaussian_count)
    eligible = (score > 0.0) & (support_np >= options.min_view_support) & ~road & ~sky
    eligible_indices = np.flatnonzero(eligible)
    max_candidates = max(1, int(round(gaussian_count * options.candidate_fraction)))
    if len(eligible_indices) > max_candidates:
        order = np.argpartition(score[eligible_indices], -max_candidates)[-max_candidates:]
        selected = eligible_indices[order]
    else:
        selected = eligible_indices
    if len(selected) == 0:
        raise RuntimeError("gradient attribution selected no Gaussian color candidates")
    selected = selected[np.argsort(score[selected])[::-1]].astype(np.int64)
    confidence = score[selected] / max(float(score[selected].max()), 1e-12)
    print(
        f"Selected {len(selected):,}/{gaussian_count:,} Gaussian color candidates "
        f"(support >= {options.min_view_support})",
        flush=True,
    )
    return selected, confidence.astype(np.float32), support_np[selected]


def _render_with_candidate_f_dc(
    renderer: GsplatRenderer,
    observation: SourceObservation,
    base_colors: Any,
    candidate_indices: Any,
    candidate_f_dc: Any,
) -> tuple[Any, Any]:
    colors = base_colors.clone()
    colors.index_copy_(0, candidate_indices, candidate_f_dc[:, None, :])
    premultiplied, alpha = _differentiable_rgb_alpha(renderer, observation.camera, colors)
    fill = renderer.torch.from_numpy(observation.sky_fill.astype(np.float32)).to(premultiplied.device)
    return (premultiplied + (1.0 - alpha[..., None]) * fill).clamp(0.0, 1.0), alpha


def optimize_candidate_colors(
    renderer: GsplatRenderer,
    observations: list[SourceObservation],
    indices: np.ndarray,
    options: GaussianDecontaminationOptions,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch = renderer.torch
    torch.manual_seed(options.seed)
    base_colors = renderer.colors.detach()
    index_tensor = torch.from_numpy(indices).to(base_colors.device)
    initial_f_dc = base_colors[index_tensor, 0].detach()
    delta_parameter = torch.zeros_like(initial_f_dc, requires_grad=True)
    optimizer = torch.optim.Adam([delta_parameter], lr=options.learning_rate)
    history: list[dict[str, float]] = []
    if options.iterations == 0:
        return initial_f_dc.cpu().numpy().astype(np.float32), history

    order = np.arange(len(observations), dtype=np.int64)
    rng = np.random.default_rng(options.seed)
    for step in range(options.iterations):
        if step % len(observations) == 0:
            rng.shuffle(order)
        observation = observations[int(order[step % len(observations)])]
        rgb_delta = options.max_color_delta * torch.tanh(delta_parameter)
        candidate_f_dc = initial_f_dc + rgb_delta / SH_C0
        prediction, alpha = _render_with_candidate_f_dc(
            renderer, observation, base_colors, index_tensor, candidate_f_dc
        )
        target = torch.from_numpy(observation.target_rgb.astype(np.float32)).to(prediction.device)
        boundary = torch.from_numpy(observation.boundary_mask).to(prediction.device)
        static = torch.from_numpy(observation.static_mask).to(prediction.device)
        baseline = torch.from_numpy(observation.baseline_rgb.astype(np.float32)).to(prediction.device)
        difference = torch.sqrt((prediction - target).square() + 1e-6).mean(dim=-1)
        boundary_loss = difference[boundary].mean()
        static_loss = (
            torch.abs(prediction - baseline).mean(dim=-1)[static].mean()
            if bool(static.any())
            else prediction.new_zeros(())
        )
        prior_loss = rgb_delta.square().mean()
        loss = (
            boundary_loss
            + options.static_preservation_weight * static_loss
            + options.color_prior_weight * prior_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        record = {
            "step": float(step),
            "loss": float(loss.detach()),
            "boundary_loss": float(boundary_loss.detach()),
            "static_loss": float(static_loss.detach()),
            "prior_loss": float(prior_loss.detach()),
        }
        history.append(record)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == options.iterations:
            print(
                f"Optimize [{step + 1:04d}/{options.iterations}] "
                f"view={observation.camera_id}:{observation.slot:02d} "
                f"loss={record['loss']:.6f} boundary={record['boundary_loss']:.6f} "
                f"static={record['static_loss']:.6f}",
                flush=True,
            )
    final_f_dc = initial_f_dc + options.max_color_delta * torch.tanh(
        delta_parameter.detach()
    ) / SH_C0
    return final_f_dc.cpu().numpy().astype(np.float32), history


def _write_report(
    renderer: GsplatRenderer,
    observations: list[SourceObservation],
    indices: np.ndarray,
    corrected_f_dc: np.ndarray,
    history: list[dict[str, float]],
    output: Path,
    preview_views: int,
    training_keys: set[tuple[str, int, int]],
) -> dict[str, object]:
    torch = renderer.torch
    report_dir = output.with_suffix("").parent / f"{output.stem}_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_colors = renderer.colors.detach()
    index_tensor = torch.from_numpy(indices).to(base_colors.device)
    f_dc_tensor = torch.from_numpy(corrected_f_dc).to(base_colors.device)
    view_records: list[dict[str, object]] = []
    with torch.no_grad():
        for view_index, observation in enumerate(observations):
            prediction, _ = _render_with_candidate_f_dc(
                renderer, observation, base_colors, index_tensor, f_dc_tensor
            )
            after = prediction.float().cpu().numpy()
            before = observation.baseline_rgb.astype(np.float32)
            target = observation.target_rgb.astype(np.float32)
            mask = observation.boundary_mask
            before_mae = float(np.abs(before - target)[mask].mean())
            after_mae = float(np.abs(after - target)[mask].mean())
            record = {
                "split": "train"
                if (observation.camera_id, observation.slot, observation.frame_index) in training_keys
                else "validation",
                "camera_id": observation.camera_id,
                "slot": observation.slot,
                "frame_index": observation.frame_index,
                "boundary_pixels": int(mask.sum()),
                "before_mae": before_mae,
                "after_mae": after_mae,
                "exposure_gain": observation.exposure_gain,
            }
            view_records.append(record)
            if view_index < preview_views:
                stem = f"{view_index:03d}_{observation.camera_id}_slot{observation.slot:02d}"
                for suffix, image in (("target", target), ("before", before), ("after", after)):
                    Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(
                        report_dir / f"{stem}_{suffix}.png"
                    )
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                    report_dir / f"{stem}_boundary.png"
                )
                delta = np.clip(np.abs(after - before) * 4.0, 0.0, 1.0)
                Image.fromarray(np.round(delta * 255.0).astype(np.uint8), mode="RGB").save(
                    report_dir / f"{stem}_delta_x4.png"
                )
    train_records = [record for record in view_records if record["split"] == "train"]
    validation_records = [record for record in view_records if record["split"] == "validation"]

    def split_summary(records: list[dict[str, object]]) -> dict[str, object] | None:
        if not records:
            return None
        return {
            "views": len(records),
            "mean_before_boundary_mae": float(np.mean([record["before_mae"] for record in records])),
            "mean_after_boundary_mae": float(np.mean([record["after_mae"] for record in records])),
        }

    report: dict[str, object] = {
        "corrected_gaussians": int(len(indices)),
        "mean_before_boundary_mae": float(np.mean([record["before_mae"] for record in view_records])),
        "mean_after_boundary_mae": float(np.mean([record["after_mae"] for record in view_records])),
        "train": split_summary(train_records),
        "validation": split_summary(validation_records),
        "views": view_records,
        "history": history,
    }
    with (report_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def decontaminate_gaussian_colors(
    scene: GaussianScene,
    renderer: GsplatRenderer,
    options: GaussianDecontaminationOptions,
) -> GaussianColorCorrection:
    if scene.sh_degree != 0:
        raise ValueError("the first decontamination implementation requires SH degree 0")
    sky_cubemap = SkyCubemap.load(options.sky_cubemap_path)
    all_slots = tuple(dict.fromkeys((*options.slots, *options.validation_slots)))
    observations = _resolve_source_views(
        replace(options, slots=all_slots, validation_slots=()), renderer, sky_cubemap
    )
    training_slot_set = set(options.slots)
    training_observations = [observation for observation in observations if observation.slot in training_slot_set]
    if not training_observations:
        raise RuntimeError("no usable training observations were constructed")
    indices, confidence, support = attribute_boundary_gaussians(renderer, training_observations, options)
    corrected_f_dc, history = optimize_candidate_colors(
        renderer, training_observations, indices, options
    )
    original_f_dc = scene.sh_coeffs[indices, 0].astype(np.float32)
    report = _write_report(
        renderer,
        observations,
        indices,
        corrected_f_dc,
        history,
        options.output,
        options.preview_views,
        {
            (observation.camera_id, observation.slot, observation.frame_index)
            for observation in training_observations
        },
    )
    details: dict[str, object] = {
        "ncore_path": str(options.ncore_path),
        "chunk_index": options.chunk_index,
        "sky_cubemap": str(options.sky_cubemap_path),
        "camera_ids": list(options.camera_ids),
        "slots": list(options.slots),
        "validation_slots": list(options.validation_slots),
        "resolution": [options.width, options.height],
        "boundary_pixels": options.boundary_pixels,
        "halo_min_luminance_excess": options.halo_min_luminance_excess,
        "halo_max_sky_color_distance": options.halo_max_sky_color_distance,
        "candidate_fraction": options.candidate_fraction,
        "per_view_candidate_fraction": options.per_view_candidate_fraction,
        "min_view_support": options.min_view_support,
        "iterations": options.iterations,
        "learning_rate": options.learning_rate,
        "max_color_delta": options.max_color_delta,
        "color_prior_weight": options.color_prior_weight,
        "static_preservation_weight": options.static_preservation_weight,
        "report": report,
    }
    correction = GaussianColorCorrection(
        indices=indices.astype(np.int64),
        original_f_dc=original_f_dc,
        corrected_f_dc=corrected_f_dc,
        confidence=confidence.astype(np.float32),
        support_count=support.astype(np.uint16),
        ply_sha256=file_sha256(options.ply_path),
        gaussian_count=scene.count,
        metadata=details,
    )
    correction.save(options.output)
    return correction
