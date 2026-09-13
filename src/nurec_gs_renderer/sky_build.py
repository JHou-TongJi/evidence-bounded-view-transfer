from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any
import warnings

import numpy as np
from PIL import Image

from .camera import compose_ncore_camera_pose
from .chunking import sample_chunk_frame_indices, sequence_sampling_interval
from .pose_sources import _create_ncore_loader
from .sky import (
    ASSET_VERSION,
    FACE_NAMES,
    TEMPORAL_ASSET_VERSION,
    SkyCubemap,
    linear_to_srgb,
    srgb_to_linear,
)


SEGFORMER_MODEL_PAGE = "https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640"
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
class SkyBuildOptions:
    ncore_path: Path
    chunk_index: int
    model_dir: Path
    output: Path
    temporal_output: Path | None = None
    cache_dir: Path | None = None
    device: str = "cuda"
    face_size: int = 1024
    working_width: int = 960
    working_height: int = 540
    sky_threshold: float = 0.85
    sky_margin_threshold: float = 0.30
    erosion_pixels: int = 6
    edge_exclusion_pixels: int = 4
    edge_gradient_threshold: float = 0.12
    n_frames_per_chunk: int = 18
    max_frame_gap_timestamp_us: int = 750_000
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    reference_camera_id: str = "camera_front_wide_120fov"
    reference_slot: int | None = None
    min_observation_support: int = 2
    color_outlier_log_threshold: float = 0.35
    coherence_block_size: int = 16
    reference_camera_priority: float = 0.25
    normalize_exposure: bool = True
    exposure_gain_min: float = 0.80
    exposure_gain_max: float = 1.25
    exposure_min_overlap: int = 512
    exposure_compensation_ev: float = 0.0
    geometry_mask_dir: Path | None = None
    geometry_mask_threshold: float = 0.10

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.output.suffix.lower() != ".npz":
            raise ValueError("output must use the .npz extension")
        if self.temporal_output is not None and self.temporal_output.suffix.lower() != ".json":
            raise ValueError("temporal_output must use the .json extension")
        if self.face_size <= 0 or self.working_width <= 0 or self.working_height <= 0:
            raise ValueError("face and working image dimensions must be positive")
        if not 0.0 < self.sky_threshold < 1.0:
            raise ValueError("sky_threshold must be in (0, 1)")
        if not 0.0 <= self.sky_margin_threshold < 1.0:
            raise ValueError("sky_margin_threshold must be in [0, 1)")
        if self.erosion_pixels < 0:
            raise ValueError("erosion_pixels must be non-negative")
        if self.edge_exclusion_pixels < 0 or not 0.0 <= self.edge_gradient_threshold <= 1.0:
            raise ValueError("edge filtering parameters are invalid")
        if self.n_frames_per_chunk <= 0 or self.max_frame_gap_timestamp_us <= 0:
            raise ValueError("chunk sampling parameters must be positive")
        if not self.camera_ids:
            raise ValueError("at least one camera id is required")
        if self.reference_camera_id not in self.camera_ids:
            raise ValueError("reference_camera_id must be included in camera_ids")
        if self.reference_slot is not None and not 0 <= self.reference_slot < self.n_frames_per_chunk:
            raise ValueError("reference_slot is out of range")
        if self.min_observation_support <= 0:
            raise ValueError("min_observation_support must be positive")
        if (
            self.color_outlier_log_threshold <= 0.0
            or self.coherence_block_size <= 0
            or self.reference_camera_priority < 0.0
        ):
            raise ValueError("fusion thresholds must be positive")
        if not 0.0 < self.exposure_gain_min <= self.exposure_gain_max:
            raise ValueError("exposure gain bounds are invalid")
        if self.exposure_min_overlap < 0 or not np.isfinite(self.exposure_compensation_ev):
            raise ValueError("exposure parameters are invalid")
        if not 0.0 <= self.geometry_mask_threshold <= 1.0:
            raise ValueError("geometry_mask_threshold must be in [0, 1]")


def find_sky_label_id(id2label: dict[Any, Any]) -> int:
    matches = [int(index) for index, label in id2label.items() if str(label).strip().casefold() == "sky"]
    if len(matches) != 1:
        raise ValueError(f"SegFormer model must contain exactly one 'sky' label, found {matches}")
    return matches[0]


def erode_boolean_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary erosion implemented with torch so scipy/OpenCV are not required."""
    mask = np.asarray(mask, dtype=bool)
    if pixels == 0:
        return mask.copy()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - torch is a base dependency
        raise RuntimeError("mask erosion requires PyTorch") from exc
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    outside_input = functional.pad(1.0 - tensor, (pixels, pixels, pixels, pixels), value=1.0)
    outside = functional.max_pool2d(outside_input, kernel_size=2 * pixels + 1, stride=1)
    return (outside[0, 0].numpy() < 0.5)


def dilate_boolean_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if pixels == 0:
        return mask.copy()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - torch is a base dependency
        raise RuntimeError("mask dilation requires PyTorch") from exc
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    dilated = functional.max_pool2d(tensor, kernel_size=2 * pixels + 1, stride=1, padding=pixels)
    return dilated[0, 0].numpy() > 0.5


def refine_sky_mask(
    sky_probability: np.ndarray,
    sky_margin: np.ndarray,
    image_rgb: np.ndarray,
    ego_mask: np.ndarray,
    *,
    sky_threshold: float,
    margin_threshold: float,
    erosion_pixels: int,
    edge_exclusion_pixels: int,
    edge_gradient_threshold: float,
    geometry_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative sky mask and per-pixel observation quality."""
    probability = np.asarray(sky_probability, dtype=np.float32)
    margin = np.asarray(sky_margin, dtype=np.float32)
    image = np.asarray(image_rgb, dtype=np.float32)
    if probability.shape != margin.shape or probability.shape != image.shape[:2]:
        raise ValueError("sky probability, margin, and image shapes do not match")
    raw = (probability >= sky_threshold) & (margin >= margin_threshold) & ~np.asarray(ego_mask, dtype=bool)
    if geometry_mask is not None:
        raw &= ~np.asarray(geometry_mask, dtype=bool)

    mask = erode_boolean_mask(raw, erosion_pixels)
    if edge_exclusion_pixels > 0 and edge_gradient_threshold > 0.0:
        luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
        gx = np.zeros_like(luminance)
        gy = np.zeros_like(luminance)
        gx[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1])
        gy[1:] = np.abs(luminance[1:] - luminance[:-1])
        strong_edge = np.maximum(gx, gy) >= edge_gradient_threshold
        boundary_width = erosion_pixels + edge_exclusion_pixels
        boundary_band = raw & ~erode_boolean_mask(raw, boundary_width)
        mask &= ~(dilate_boolean_mask(strong_edge, edge_exclusion_pixels) & boundary_band)

    normalized_margin = np.clip(
        (margin - margin_threshold) / max(1.0 - margin_threshold, 1e-6), 0.0, 1.0
    )
    quality = probability * (0.5 + 0.5 * normalized_margin)
    quality[~mask] = 0.0
    return mask, quality.astype(np.float32)


class SegformerSkySegmenter:
    def __init__(self, model_dir: Path, device: str) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"SegFormer model directory does not exist: {model_dir}\nDownload it from {SEGFORMER_MODEL_PAGE}"
            )
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "sky building requires the optional dependencies: "
                "pip install 'instant-nurec-gs-renderer[sky]'"
            ) from exc
        self.torch = torch
        self.device = torch.device(device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(
                f"failed to load a complete local SegFormer snapshot from {model_dir}. "
                f"Download all model files from {SEGFORMER_MODEL_PAGE}"
            ) from exc
        self.model.to(self.device).eval()
        self.sky_label_id = find_sky_label_id(self.model.config.id2label)

    def sky_scores(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = logits.softmax(dim=1)[0]
            sky_probability = probabilities[self.sky_label_id].clone()
            probabilities[self.sky_label_id] = -1.0
            strongest_non_sky = probabilities.max(dim=0).values
            margin = sky_probability - strongest_non_sky
        return sky_probability.float().cpu().numpy(), margin.float().cpu().numpy()

    def sky_probability(self, image: Image.Image) -> np.ndarray:
        """Compatibility helper retained for callers that only need sky probability."""
        return self.sky_scores(image)[0]


_sequence_sampling_interval = sequence_sampling_interval


def _evaluate_exposure_poses(loader: object, start_us: int, end_us: int) -> np.ndarray:
    """Evaluate NCore rig poses with the ndarray input required by NCore 18.7."""
    timestamps = np.asarray([start_us, end_us], dtype=np.uint64)
    poses = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", timestamps))
    if poses.shape != (2, 4, 4):
        raise ValueError(f"NCore pose graph returned an unexpected exposure pose shape: {poses.shape}")
    return poses


def _load_ego_mask(sensor: object, size: tuple[int, int]) -> np.ndarray:
    image = sensor.get_mask_images().get("ego")
    if image is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    resized = image.convert("L").resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(resized) != 0


def _prepare_camera_model(sensor: object, size: tuple[int, int], device: str):
    try:
        from ncore.impl.sensors.camera import FThetaCameraModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sky building requires nvidia-ncore") from exc
    parameters = sensor.model_parameters
    required = ("resolution", "transform", "pixeldist_to_angle_poly", "shutter_type")
    if not all(hasattr(parameters, name) for name in required):
        raise ValueError(f"camera {sensor} is not an NCore FTheta camera")
    if np.all(np.asarray(parameters.linear_cde) == 0.0):
        parameters = dataclasses.replace(parameters, linear_cde=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    source_resolution = np.asarray(parameters.resolution, dtype=np.float64)
    scale = (size[0] / source_resolution[0], size[1] / source_resolution[1])
    parameters = parameters.transform(scale, new_resolution=size)
    return FThetaCameraModel(parameters, device=device)


def _splat_observations(
    directions: object,
    colors: object,
    confidence: object,
    rgb_sum: object,
    weight_sum: object,
    support_sum: object,
    sample_count: object,
    face_size: int,
) -> None:
    import torch

    x, y, z = directions.unbind(-1)
    absolute = directions.abs()
    dominant = absolute.argmax(dim=-1)
    scale = absolute.gather(-1, dominant[:, None]).squeeze(-1).clamp_min(1e-12)
    positive = directions.gather(-1, dominant[:, None]).squeeze(-1) > 0
    face = torch.where(
        dominant == 0,
        torch.where(positive, torch.zeros_like(dominant), torch.ones_like(dominant)),
        torch.where(
            dominant == 1,
            torch.where(positive, torch.full_like(dominant, 3), torch.full_like(dominant, 2)),
            torch.where(positive, torch.full_like(dominant, 4), torch.full_like(dominant, 5)),
        ),
    )
    inv = 1.0 / scale
    u_faces = torch.stack((-z * inv, z * inv, x * inv, x * inv, x * inv, -x * inv), dim=-1)
    v_faces = torch.stack((y * inv, y * inv, z * inv, -z * inv, y * inv, y * inv), dim=-1)
    u = u_faces.gather(-1, face[:, None]).squeeze(-1)
    v = v_faces.gather(-1, face[:, None]).squeeze(-1)
    px = ((u + 1.0) * face_size * 0.5 - 0.5).clamp(0.0, face_size - 1.0)
    py = ((v + 1.0) * face_size * 0.5 - 0.5).clamp(0.0, face_size - 1.0)
    x0, y0 = px.floor().long(), py.floor().long()
    x1, y1 = (x0 + 1).clamp_max(face_size - 1), (y0 + 1).clamp_max(face_size - 1)
    wx, wy = px - x0, py - y0
    base = face * face_size * face_size
    flat_rgb = rgb_sum.view(-1, 3)
    flat_weight = weight_sum.view(-1)
    flat_support = support_sum.view(-1)
    flat_count = sample_count.view(-1)
    for xi, yi, bilinear in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        weight = confidence * bilinear
        keep = weight > 1e-8
        if not bool(keep.any()):
            continue
        index = base[keep] + yi[keep] * face_size + xi[keep]
        flat_rgb.index_add_(0, index, colors[keep] * weight[keep, None])
        flat_weight.index_add_(0, index, weight[keep])
        flat_support.index_add_(0, index, bilinear[keep])
        flat_count.index_add_(0, index, torch.ones_like(index, dtype=flat_count.dtype))


def _load_geometry_mask(
    directory: Path | None,
    camera_id: str,
    frame_index: int,
    size: tuple[int, int],
    threshold: float,
) -> np.ndarray | None:
    """Load an optional source-view geometry alpha mask.

    Accepted layouts are ``<dir>/<camera>/<frame>.png`` and
    ``<dir>/<camera>_<frame>.png``. White/non-zero values mean finite geometry
    and are excluded from the sky observations.
    """
    if directory is None:
        return None
    candidates = (
        directory / camera_id / f"{frame_index:06d}.png",
        directory / f"{camera_id}_{frame_index:06d}.png",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"missing geometry mask for camera={camera_id} frame={frame_index}; "
            f"tried {[str(candidate) for candidate in candidates]}"
        )
    alpha = Image.open(path).convert("L").resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(alpha, dtype=np.float32) / 255.0 >= threshold


def _linear_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def estimate_exposure_gains(
    colors: np.ndarray,
    valid: np.ndarray,
    reference_index: int,
    *,
    min_overlap: int,
    gain_min: float,
    gain_max: float,
) -> np.ndarray:
    """Robustly align linear-light camera observations to one reference camera."""
    colors = np.asarray(colors, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if colors.ndim != 5 or valid.shape != colors.shape[:-1]:
        raise ValueError("camera candidate color/valid shapes do not match")
    if not 0 <= reference_index < len(colors):
        raise ValueError("reference camera index is out of range")
    epsilon = 1e-5
    luminance = _linear_luminance(colors)
    gains = np.ones(len(colors), dtype=np.float32)
    for index in range(len(colors)):
        if index == reference_index:
            continue
        overlap = valid[index] & valid[reference_index]
        if int(overlap.sum()) >= min_overlap:
            log_ratio = np.log(luminance[reference_index][overlap] + epsilon) - np.log(
                luminance[index][overlap] + epsilon
            )
            gain = float(np.exp(np.median(log_ratio)))
        else:
            # Directions without overlap do not provide a defensible exposure
            # correspondence. Keep their native exposure instead of matching
            # unrelated cloud/horizon regions by a global median.
            gain = 1.0
        gains[index] = np.clip(gain, gain_min, gain_max)
    return gains


def _block_coherent_choices(scores: np.ndarray, valid: np.ndarray, block_size: int) -> np.ndarray:
    """Prefer one source per spatial block while falling back at invalid pixels."""
    camera_count, face_count, height, width = scores.shape
    pixel_choices = np.argmax(np.where(valid, scores, -np.inf), axis=0)
    if block_size <= 1:
        return pixel_choices
    pad_h = (-height) % block_size
    pad_w = (-width) % block_size
    padded_score = np.pad(scores * valid, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    padded_valid = np.pad(valid, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    padded_height, padded_width = height + pad_h, width + pad_w
    shape = (
        camera_count,
        face_count,
        padded_height // block_size,
        block_size,
        padded_width // block_size,
        block_size,
    )
    sums = padded_score.reshape(shape).sum(axis=(3, 5))
    counts = padded_valid.reshape(shape).sum(axis=(3, 5))
    means = np.divide(sums, counts, out=np.full_like(sums, -np.inf), where=counts > 0)
    block_choices = np.argmax(means, axis=0)
    choices = np.repeat(np.repeat(block_choices, block_size, axis=1), block_size, axis=2)[:, :height, :width]
    block_choice_valid = np.take_along_axis(valid, choices[None], axis=0)[0]
    return np.where(block_choice_valid, choices, pixel_choices)


def fuse_camera_observations(
    colors: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    *,
    coherence_block_size: int,
    color_outlier_log_threshold: float,
    preferred_camera_index: int | None = None,
    preferred_camera_priority: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose one camera observation per direction instead of averaging views."""
    colors = np.asarray(colors, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if colors.ndim != 5 or confidence.shape != colors.shape[:-1] or valid.shape != confidence.shape:
        raise ValueError("camera observation shapes do not match")

    luminance = _linear_luminance(colors)
    log_luminance = np.where(valid, np.log(luminance + 1e-5), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(log_luminance, axis=0)
        deviation = np.abs(log_luminance - median[None])
        mad = np.nanmedian(deviation, axis=0)
    support_before_rejection = valid.sum(axis=0)
    tolerance = np.maximum(color_outlier_log_threshold, 3.0 * np.nan_to_num(mad, nan=0.0))
    consistent = deviation <= tolerance[None]
    # With fewer than three views there is not enough evidence to identify the
    # outlier, so retain both observations and defer to confidence.
    filtered_valid = valid & ((support_before_rejection < 3)[None] | consistent)
    selection_score = confidence.copy()
    if preferred_camera_index is not None:
        if not 0 <= preferred_camera_index < len(colors):
            raise ValueError("preferred_camera_index is out of range")
        selection_score[preferred_camera_index] += preferred_camera_priority
    choices = _block_coherent_choices(selection_score, filtered_valid, coherence_block_size)
    selected_rgb = np.take_along_axis(colors, choices[None, ..., None], axis=0)[0]
    selected_confidence = np.take_along_axis(confidence, choices[None], axis=0)[0]
    any_valid = filtered_valid.any(axis=0)
    selected_rgb[~any_valid] = 0.0
    selected_confidence[~any_valid] = 0.0
    return (
        selected_rgb.astype(np.float32),
        any_valid.astype(np.uint8),
        selected_confidence.astype(np.float32),
        filtered_valid.sum(axis=0).astype(np.uint16),
    )


def reference_slot_fill_select(
    colors: np.ndarray,
    confidence: np.ndarray,
    counts: np.ndarray,
    *,
    reference_slot: int,
    min_observation_support: int,
    color_outlier_log_threshold: float,
    row_block: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep a coherent reference time and use nearby slots only to fill holes."""
    slot_count, face_count, face_size, _, _ = colors.shape
    if not 0 <= reference_slot < slot_count:
        raise ValueError("reference_slot is out of range")
    output_rgb = np.zeros((face_count, face_size, face_size, 3), dtype=np.float32)
    output_confidence = np.zeros((face_count, face_size, face_size), dtype=np.float32)
    output_valid = np.zeros((face_count, face_size, face_size), dtype=np.uint8)
    output_count = np.zeros((face_count, face_size, face_size), dtype=np.uint16)
    order = sorted(range(slot_count), key=lambda slot: (abs(slot - reference_slot), slot))
    for face in range(face_count):
        for row_start in range(0, face_size, row_block):
            row_end = min(row_start + row_block, face_size)
            rgb = np.asarray(colors[:, face, row_start:row_end], dtype=np.float32)
            conf = np.asarray(confidence[:, face, row_start:row_end], dtype=np.float32)
            count = np.asarray(counts[:, face, row_start:row_end], dtype=np.uint32)
            valid = count > 0
            total_count = count.sum(axis=0)
            log_luminance = np.where(valid, np.log(_linear_luminance(rgb) + 1e-5), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(log_luminance, axis=0)
                deviation = np.abs(log_luminance - median[None])
                mad = np.nanmedian(deviation, axis=0)
            tolerance = np.maximum(color_outlier_log_threshold, 3.0 * np.nan_to_num(mad, nan=0.0))
            consistent = deviation <= tolerance[None]
            support_slots = valid.sum(axis=0)
            usable = valid & ((support_slots < 3)[None] | consistent)

            selected_rgb = np.zeros(rgb.shape[1:], dtype=np.float32)
            selected_conf = np.zeros(conf.shape[1:], dtype=np.float32)
            selected = np.zeros(conf.shape[1:], dtype=bool)
            for slot in order:
                take = usable[slot] & ~selected
                selected_rgb[take] = rgb[slot][take]
                selected_conf[take] = conf[slot][take]
                selected |= take
            final_valid = selected & (total_count >= min_observation_support)
            selected_rgb[~final_valid] = 0.0
            selected_conf[~final_valid] = 0.0
            output_rgb[face, row_start:row_end] = selected_rgb
            output_confidence[face, row_start:row_end] = selected_conf
            output_valid[face, row_start:row_end] = final_valid.astype(np.uint8)
            output_count[face, row_start:row_end] = np.clip(total_count, 0, 65535).astype(np.uint16)
    return output_rgb, output_valid, output_confidence, output_count


def _temporal_median_luminance_select(
    colors: np.ndarray,
    confidence: np.ndarray,
    counts: np.ndarray,
    *,
    row_block: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one temporally coherent RGB observation per texel."""
    _, face_count, face_size, _, _ = colors.shape
    output_rgb = np.zeros((face_count, face_size, face_size, 3), dtype=np.float32)
    output_confidence = np.zeros((face_count, face_size, face_size), dtype=np.float32)
    output_valid = np.zeros((face_count, face_size, face_size), dtype=np.uint8)
    output_count = np.zeros((face_count, face_size, face_size), dtype=np.uint16)
    for face in range(face_count):
        for row_start in range(0, face_size, row_block):
            row_end = min(row_start + row_block, face_size)
            rgb = np.asarray(colors[:, face, row_start:row_end], dtype=np.float32)
            conf = np.asarray(confidence[:, face, row_start:row_end], dtype=np.float32)
            count = np.asarray(counts[:, face, row_start:row_end], dtype=np.uint32)
            valid = count > 0
            luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
            luminance[~valid] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(luminance, axis=0)
            distance = np.abs(luminance - median[None])
            distance[~valid] = np.inf
            selected = np.argmin(distance, axis=0)
            selected_rgb = np.take_along_axis(rgb, selected[None, ..., None], axis=0)[0]
            selected_conf = np.take_along_axis(conf, selected[None], axis=0)[0]
            any_valid = valid.any(axis=0)
            selected_rgb[~any_valid] = 0.0
            selected_conf[~any_valid] = 0.0
            output_rgb[face, row_start:row_end] = selected_rgb
            output_confidence[face, row_start:row_end] = selected_conf
            output_valid[face, row_start:row_end] = any_valid.astype(np.uint8)
            output_count[face, row_start:row_end] = np.clip(count.sum(axis=0), 0, 65535).astype(np.uint16)
    return output_rgb, output_valid, output_confidence, output_count


def _write_previews(asset: SkyCubemap, output: Path) -> dict[str, Any]:
    preview_dir = output.with_suffix("").parent / f"{output.stem}_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_names = ("pos_x", "neg_x", "neg_y", "pos_y", "pos_z", "neg_z")
    face_coverage: dict[str, float] = {}
    for index, (face_name, safe_name) in enumerate(zip(FACE_NAMES, safe_names, strict=True)):
        rgb = np.asarray(asset.rgb[index], dtype=np.float32) * asset.valid[index, ..., None]
        Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(
            preview_dir / f"{index:02d}_{safe_name}.png"
        )
        Image.fromarray(asset.valid[index].astype(np.uint8) * 255, mode="L").save(
            preview_dir / f"{index:02d}_{safe_name}_valid.png"
        )
        face_coverage[face_name] = float(np.asarray(asset.valid[index], dtype=bool).mean())
    report = {
        "asset": str(output),
        "total_coverage": float(np.asarray(asset.valid, dtype=bool).mean()),
        "face_coverage": face_coverage,
        "preview_directory": str(preview_dir),
    }
    report_path = output.with_name(f"{output.stem}_coverage.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def build_sky_cubemap(options: SkyBuildOptions) -> SkyCubemap:
    try:
        import ncore.data
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sky building requires nvidia-ncore and PyTorch") from exc

    options.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading NCore sequence: {options.ncore_path}", flush=True)
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
    sampled, references, n_chunks = sample_chunk_frame_indices(
        timestamps,
        sequence_start,
        sequence_end,
        options.chunk_index,
        n_frames_per_chunk=options.n_frames_per_chunk,
        max_frame_gap_timestamp_us=options.max_frame_gap_timestamp_us,
    )
    print(
        f"Sequence {loader.sequence_id}: chunk {options.chunk_index}/{n_chunks - 1}, "
        f"{len(options.camera_ids)} cameras x {options.n_frames_per_chunk} slots",
        flush=True,
    )
    segmenter = SegformerSkySegmenter(options.model_dir, options.device)
    camera_models = {
        camera_id: _prepare_camera_model(sensor, (options.working_width, options.working_height), options.device)
        for camera_id, sensor in sensors.items()
    }
    ego_masks = {
        camera_id: _load_ego_mask(sensor, (options.working_width, options.working_height))
        for camera_id, sensor in sensors.items()
    }
    device = torch.device(options.device)
    cube_shape = (6, options.face_size, options.face_size)
    frame_records: dict[str, list[dict[str, Any]]] = {camera_id: [] for camera_id in options.camera_ids}
    reference_camera_index = options.camera_ids.index(options.reference_camera_id)
    reference_slot = options.n_frames_per_chunk // 2 if options.reference_slot is None else options.reference_slot
    exposure_records: list[dict[str, float]] = []

    temporary_parent = options.output.parent if options.cache_dir is None else options.cache_dir
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nurec-sky-", dir=temporary_parent) as temp_dir:
        temp = Path(temp_dir)
        temporal_rgb = np.memmap(
            temp / "rgb.dat",
            mode="w+",
            dtype=np.float16,
            shape=(options.n_frames_per_chunk, *cube_shape, 3),
        )
        temporal_confidence = np.memmap(
            temp / "confidence.dat",
            mode="w+",
            dtype=np.float16,
            shape=(options.n_frames_per_chunk, *cube_shape),
        )
        temporal_count = np.memmap(
            temp / "count.dat",
            mode="w+",
            dtype=np.uint16,
            shape=(options.n_frames_per_chunk, *cube_shape),
        )

        total_frames = options.n_frames_per_chunk * len(options.camera_ids)
        processed = 0
        for slot in range(options.n_frames_per_chunk):
            camera_colors: list[np.ndarray] = []
            camera_confidence: list[np.ndarray] = []
            camera_valid: list[np.ndarray] = []
            for camera_id in options.camera_ids:
                started = time.perf_counter()
                sensor = sensors[camera_id]
                frame_index = sampled[camera_id][slot]
                image_array = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
                image = Image.fromarray(image_array, mode="RGB").resize(
                    (options.working_width, options.working_height), resample=Image.Resampling.BILINEAR
                )
                image_float = np.asarray(image, dtype=np.float32) / 255.0
                sky_probability, sky_margin = segmenter.sky_scores(image)
                geometry_mask = _load_geometry_mask(
                    options.geometry_mask_dir,
                    camera_id,
                    frame_index,
                    (options.working_width, options.working_height),
                    options.geometry_mask_threshold,
                )
                sky_mask, observation_quality = refine_sky_mask(
                    sky_probability,
                    sky_margin,
                    image_float,
                    ego_masks[camera_id],
                    sky_threshold=options.sky_threshold,
                    margin_threshold=options.sky_margin_threshold,
                    erosion_pixels=options.erosion_pixels,
                    edge_exclusion_pixels=options.edge_exclusion_pixels,
                    edge_gradient_threshold=options.edge_gradient_threshold,
                    geometry_mask=geometry_mask,
                )
                yy, xx = np.nonzero(sky_mask)
                start_us = int(sensor.get_frame_timestamp_us(frame_index, ncore.data.FrameTimepoint.START))
                end_us = int(sensor.get_frame_timestamp_us(frame_index, ncore.data.FrameTimepoint.END))
                rgb_sum = torch.zeros((*cube_shape, 3), device=device, dtype=torch.float32)
                weight_sum = torch.zeros(cube_shape, device=device, dtype=torch.float32)
                support_sum = torch.zeros(cube_shape, device=device, dtype=torch.float32)
                pixel_count = torch.zeros(cube_shape, device=device, dtype=torch.int32)
                if len(xx) > 0:
                    poses = _evaluate_exposure_poses(loader, start_us, end_us)
                    T_camera_rig = np.asarray(sensor.T_sensor_rig, dtype=np.float64)
                    start_c2w = compose_ncore_camera_pose(poses[0], T_camera_rig)
                    end_c2w = compose_ncore_camera_pose(poses[1], T_camera_rig)
                    pixels = np.stack((xx, yy), axis=-1).astype(np.int32)
                    world_rays = camera_models[camera_id].pixels_to_world_rays_shutter_pose(
                        pixels,
                        start_c2w,
                        end_c2w,
                        start_us,
                        end_us,
                    ).world_rays
                    colors = torch.from_numpy(srgb_to_linear(image_float)[sky_mask]).to(device=device)
                    radial = np.sqrt(
                        ((xx - (options.working_width - 1) * 0.5) / max(options.working_width * 0.5, 1.0)) ** 2
                        + ((yy - (options.working_height - 1) * 0.5) / max(options.working_height * 0.5, 1.0)) ** 2
                    )
                    optical_axis_quality = np.clip(1.0 - 0.35 * radial, 0.35, 1.0).astype(np.float32)
                    confidence = torch.from_numpy(
                        observation_quality[sky_mask] * optical_axis_quality
                    ).to(device=device, dtype=torch.float32)
                    _splat_observations(
                        world_rays[:, 3:],
                        colors,
                        confidence,
                        rgb_sum,
                        weight_sum,
                        support_sum,
                        pixel_count,
                        options.face_size,
                    )
                candidate_valid = weight_sum > 0
                candidate_rgb = rgb_sum / weight_sum.clamp_min(1e-8)[..., None]
                candidate_confidence = weight_sum / support_sum.clamp_min(1e-8)
                candidate_rgb[~candidate_valid] = 0.0
                candidate_confidence[~candidate_valid] = 0.0
                camera_colors.append(candidate_rgb.clamp(0.0, 1.0).half().cpu().numpy())
                camera_confidence.append(candidate_confidence.clamp(0.0, 1.0).half().cpu().numpy())
                camera_valid.append(candidate_valid.cpu().numpy())
                frame_records[camera_id].append(
                    {
                        "slot": slot,
                        "frame_index": frame_index,
                        "start_timestamp_us": start_us,
                        "end_timestamp_us": end_us,
                        "sky_pixels": int(len(xx)),
                    }
                )
                processed += 1
                coverage = float(candidate_valid.float().mean().item()) * 100.0
                print(
                    f"[{processed:03d}/{total_frames}] slot={slot:02d} camera={camera_id} frame={frame_index:03d} "
                    f"sky_pixels={len(xx):7d} camera_coverage={coverage:5.1f}% "
                    f"time={time.perf_counter() - started:5.2f}s",
                    flush=True,
                )

            colors_np = np.asarray(camera_colors, dtype=np.float32)
            confidence_np = np.asarray(camera_confidence, dtype=np.float32)
            valid_np = np.asarray(camera_valid, dtype=bool)
            if options.normalize_exposure:
                gains = estimate_exposure_gains(
                    colors_np,
                    valid_np,
                    reference_camera_index,
                    min_overlap=options.exposure_min_overlap,
                    gain_min=options.exposure_gain_min,
                    gain_max=options.exposure_gain_max,
                )
                colors_np = np.clip(colors_np * gains[:, None, None, None, None], 0.0, 1.0)
            else:
                gains = np.ones(len(options.camera_ids), dtype=np.float32)
            for camera_id, gain in zip(options.camera_ids, gains, strict=True):
                frame_records[camera_id][-1]["exposure_gain"] = float(gain)
            exposure_records.append(
                {camera_id: float(gain) for camera_id, gain in zip(options.camera_ids, gains, strict=True)}
            )
            rgb, valid, confidence, view_count = fuse_camera_observations(
                colors_np,
                confidence_np,
                valid_np,
                coherence_block_size=options.coherence_block_size,
                color_outlier_log_threshold=options.color_outlier_log_threshold,
                preferred_camera_index=reference_camera_index,
                preferred_camera_priority=options.reference_camera_priority,
            )
            temporal_rgb[slot] = rgb.astype(np.float16)
            temporal_confidence[slot] = confidence.astype(np.float16)
            temporal_count[slot] = view_count
            temporal_rgb.flush()
            temporal_confidence.flush()
            temporal_count.flush()
            print(
                f"  fused slot={slot:02d} coverage={float(valid.mean()) * 100.0:5.1f}% "
                f"mean_view_support={float(view_count[valid.astype(bool)].mean()) if valid.any() else 0.0:.2f} "
                f"gains={','.join(f'{gain:.3f}' for gain in gains)}",
                flush=True,
            )

        temporal_manifest: dict[str, Any] | None = None
        if options.temporal_output is not None:
            temporal_output = options.temporal_output
            temporal_output.parent.mkdir(parents=True, exist_ok=True)
            slot_dir = temporal_output.parent / f"{temporal_output.stem}_slots"
            slot_dir.mkdir(parents=True, exist_ok=True)
            temporal_slots: list[dict[str, Any]] = []
            slot_coverage: list[float] = []
            print(
                f"Writing {options.n_frames_per_chunk} time-indexed cubemap slots...",
                flush=True,
            )
            for target_slot in range(options.n_frames_per_chunk):
                slot_rgb_linear, slot_valid, slot_confidence, slot_count = reference_slot_fill_select(
                    temporal_rgb,
                    temporal_confidence,
                    temporal_count,
                    reference_slot=target_slot,
                    min_observation_support=options.min_observation_support,
                    color_outlier_log_threshold=options.color_outlier_log_threshold,
                )
                slot_rgb_linear = np.clip(
                    slot_rgb_linear * (2.0 ** options.exposure_compensation_ev), 0.0, 1.0
                )
                timestamp_us = int(
                    frame_records[options.reference_camera_id][target_slot]["end_timestamp_us"]
                )
                slot_metadata: dict[str, Any] = {
                    "asset_version": ASSET_VERSION,
                    "face_order": list(FACE_NAMES),
                    "color_space": "sRGB (linear-light fusion)",
                    "sequence_id": str(loader.sequence_id),
                    "ncore_path": str(options.ncore_path.resolve()),
                    "chunk_index": options.chunk_index,
                    "temporal_slot": target_slot,
                    "timestamp_us": timestamp_us,
                    "reference_camera_id": options.reference_camera_id,
                    "reference_frame_index": frame_records[options.reference_camera_id][target_slot][
                        "frame_index"
                    ],
                    "face_size": options.face_size,
                    "min_observation_support": options.min_observation_support,
                    "temporal_fusion": "current slot first; nearest slots fill invalid directions only",
                }
                slot_asset = SkyCubemap(
                    rgb=linear_to_srgb(slot_rgb_linear).astype(np.float16),
                    valid=slot_valid,
                    confidence=slot_confidence.astype(np.float16),
                    sample_count=slot_count,
                    metadata=slot_metadata,
                )
                slot_path = slot_dir / f"slot{target_slot:02d}.npz"
                slot_asset.save(slot_path)
                coverage = float(np.asarray(slot_valid, dtype=bool).mean())
                slot_coverage.append(coverage)
                temporal_slots.append(
                    {
                        "slot": target_slot,
                        "timestamp_us": timestamp_us,
                        "asset": str(slot_path.relative_to(temporal_output.parent)),
                        "coverage": coverage,
                    }
                )
                print(
                    f"  temporal slot={target_slot:02d} timestamp={timestamp_us} "
                    f"coverage={coverage * 100.0:.1f}%",
                    flush=True,
                )
            temporal_manifest = {
                "type": "temporal_sky_cubemap",
                "asset_version": TEMPORAL_ASSET_VERSION,
                "sequence_id": str(loader.sequence_id),
                "ncore_path": str(options.ncore_path.resolve()),
                "chunk_index": options.chunk_index,
                "face_size": options.face_size,
                "reference_camera_id": options.reference_camera_id,
                "interpolation": "linear light where both slots are valid; otherwise use the valid slot",
                "slot_coverage": slot_coverage,
                "slots": temporal_slots,
            }
            with temporal_output.open("w", encoding="utf-8") as handle:
                json.dump(temporal_manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            print(f"Wrote temporal sky manifest: {temporal_output}", flush=True)

        print(f"Selecting reference slot {reference_slot} and filling invalid directions...", flush=True)
        rgb_linear, valid, confidence, sample_count = reference_slot_fill_select(
            temporal_rgb,
            temporal_confidence,
            temporal_count,
            reference_slot=reference_slot,
            min_observation_support=options.min_observation_support,
            color_outlier_log_threshold=options.color_outlier_log_threshold,
        )
        rgb_linear = np.clip(rgb_linear * (2.0 ** options.exposure_compensation_ev), 0.0, 1.0)

    metadata: dict[str, Any] = {
        "asset_version": ASSET_VERSION,
        "face_order": list(FACE_NAMES),
        "color_space": "sRGB (linear-light fusion)",
        "sequence_id": str(loader.sequence_id),
        "ncore_path": str(options.ncore_path.resolve()),
        "chunk_index": options.chunk_index,
        "n_chunks": n_chunks,
        "sequence_sampling_interval_us": [sequence_start, sequence_end],
        "reference_timestamps_us": references,
        "camera_ids": list(options.camera_ids),
        "frames": frame_records,
        "face_size": options.face_size,
        "working_resolution": [options.working_width, options.working_height],
        "sky_threshold": options.sky_threshold,
        "sky_margin_threshold": options.sky_margin_threshold,
        "erosion_pixels": options.erosion_pixels,
        "edge_exclusion_pixels": options.edge_exclusion_pixels,
        "edge_gradient_threshold": options.edge_gradient_threshold,
        "reference_camera_id": options.reference_camera_id,
        "reference_slot": reference_slot,
        "min_observation_support": options.min_observation_support,
        "color_outlier_log_threshold": options.color_outlier_log_threshold,
        "coherence_block_size": options.coherence_block_size,
        "reference_camera_priority": options.reference_camera_priority,
        "normalize_exposure": options.normalize_exposure,
        "exposure_gain_bounds": [options.exposure_gain_min, options.exposure_gain_max],
        "exposure_min_overlap": options.exposure_min_overlap,
        "exposure_compensation_ev": options.exposure_compensation_ev,
        "exposure_gains": exposure_records,
        "geometry_mask_directory": None
        if options.geometry_mask_dir is None
        else str(options.geometry_mask_dir.resolve()),
        "geometry_mask_threshold": options.geometry_mask_threshold,
        "temporary_cache_parent": str(temporary_parent.resolve()),
        "segmentation_model_directory": str(options.model_dir.resolve()),
        "segmentation_model_page": SEGFORMER_MODEL_PAGE,
        "camera_fusion": "block-coherent best single observation with robust color rejection",
        "temporal_fusion": "reference slot first; nearest slots fill invalid directions only",
        "sample_count_semantics": "independent accepted camera-slot observations per texel",
    }
    asset = SkyCubemap(
        rgb=linear_to_srgb(rgb_linear).astype(np.float16),
        valid=valid,
        confidence=confidence.astype(np.float16),
        sample_count=sample_count,
        metadata=metadata,
    )
    print(f"Writing cubemap asset: {options.output}", flush=True)
    asset.save(options.output)
    report = _write_previews(asset, options.output)
    print(f"Finished: {report['total_coverage'] * 100.0:.1f}% total direction coverage", flush=True)
    return asset
