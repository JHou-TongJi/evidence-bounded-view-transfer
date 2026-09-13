"""Audit local SAM2 feature correspondences before attempting dense dynamic 3D.

The raw dynamic route established that sparse LiDAR is insufficient to complete
the observed vehicle surface.  This module is deliberately an *audit*, not a
trainer: it uses the already-downloaded local SAM2.1 image encoder to test
whether a tracked instance has stable, mutual temporal feature matches.  It
never loads a static PLY, generates Gaussians, or turns a SAM2-only mask into
training supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader


SCHEMA = "ncore-sam2-temporal-feature-correspondence-audit"
VERSION = 1
TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"


def sample_mask_pixels(mask: np.ndarray, maximum: int) -> np.ndarray:
    """Return deterministic ``(x, y)`` samples from a non-empty binary mask."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or maximum <= 0:
        raise ValueError("mask must be two-dimensional and maximum positive")
    yy, xx = np.nonzero(value)
    if not len(xx):
        return np.empty((0, 2), dtype=np.int64)
    indices = np.linspace(0, len(xx) - 1, min(maximum, len(xx)), dtype=np.int64)
    return np.stack((xx[indices], yy[indices]), axis=1).astype(np.int64)


def _pixel_features(feature_map: np.ndarray, pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sample a CxHxW feature map at original-image pixel centres."""
    feature = np.asarray(feature_map, dtype=np.float32)
    xy = np.asarray(pixels, dtype=np.int64)
    if feature.ndim != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("feature map must be [C,H,W] and pixels [N,2]")
    height, width = shape
    if min(height, width) <= 0:
        raise ValueError("image shape must be positive")
    feature_height, feature_width = feature.shape[1:]
    x = np.clip(np.rint((xy[:, 0] + .5) * feature_width / width - .5), 0, feature_width - 1).astype(np.int64)
    y = np.clip(np.rint((xy[:, 1] + .5) * feature_height / height - .5), 0, feature_height - 1).astype(np.int64)
    vectors = feature[:, y, x].T
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def mutual_feature_matches(
    source_feature: np.ndarray,
    target_feature: np.ndarray,
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Match two masks with mutual nearest SAM2 features and robust motion stats."""
    source = _pixel_features(source_feature, source_pixels, image_shape)
    target = _pixel_features(target_feature, target_pixels, image_shape)
    if not len(source) or not len(target):
        return {"source_samples": int(len(source)), "target_samples": int(len(target)), "match_count": 0,
                "mutual_source_fraction": 0.0, "mutual_target_fraction": 0.0}
    similarity = source @ target.T
    forward = similarity.argmax(axis=1)
    reverse = similarity.argmax(axis=0)
    indices = np.arange(len(source), dtype=np.int64)
    mutual = reverse[forward] == indices
    matched_source = indices[mutual]
    matched_target = forward[mutual]
    result: dict[str, Any] = {
        "source_samples": int(len(source)), "target_samples": int(len(target)),
        "match_count": int(len(matched_source)),
        "mutual_source_fraction": float(mutual.mean()),
        "mutual_target_fraction": float(len(np.unique(matched_target)) / len(target)),
    }
    if not len(matched_source):
        return result
    displacement = target_pixels[matched_target].astype(np.float32) - source_pixels[matched_source].astype(np.float32)
    median = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - median[None], axis=1)
    values = similarity[matched_source, matched_target]
    result.update({
        "mean_cosine_similarity": float(values.mean()),
        "median_displacement_xy": [float(value) for value in median],
        "median_motion_residual_pixels": float(np.median(residual)),
        "p95_motion_residual_pixels": float(np.percentile(residual, 95)),
    })
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_sam2_feature_encoder(model_dir: Path, device: str) -> tuple[Any, Any, Any, Any]:
    """Load the local image encoder once; callers may process many frame pairs."""
    try:
        import torch
        from transformers import Sam2Processor, Sam2VideoModel
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("requires local transformers SAM2.1 support") from exc
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SAM2 model directory is missing: {model_dir}")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    processor = Sam2Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Sam2VideoModel.from_pretrained(str(model_dir), local_files_only=True).to(device, dtype=dtype).eval()
    return torch, model, processor, dtype


def _sam2_feature_maps(
    torch: Any, model: Any, processor: Any, dtype: Any, images: list[Image.Image], device: str, feature_level: str,
) -> tuple[list[np.ndarray], tuple[int, int, int]]:
    """Extract one local-SAM2 feature scale for a frame batch.

    SAM2 emits a shallow 32-channel 256² map plus deeper 64/256-channel
    maps.  The former has the best localization but is not necessarily a good
    descriptor; the audit must be able to test the latter explicitly.
    """
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_embeddings(inputs["pixel_values"].to(device, dtype=dtype))
    candidates = [value.detach().float().cpu().numpy() for value in features if getattr(value, "ndim", 0) == 4]
    if not candidates:
        raise RuntimeError("SAM2 returned no [B,C,H,W] image feature")
    ordered = sorted(candidates, key=lambda value: value.shape[-2] * value.shape[-1], reverse=True)
    index = {"high": 0, "mid": 1, "deep": -1}[feature_level]
    if len(ordered) < 3:
        raise RuntimeError("SAM2 did not return the required three feature scales")
    chosen = ordered[index]
    if chosen.shape[0] != len(images):
        raise RuntimeError("SAM2 feature batch does not match requested images")
    return [chosen[index] for index in range(len(images))], tuple(int(value) for value in chosen.shape[1:])


@dataclass(frozen=True)
class DynamicFeatureCorrespondenceOptions:
    trusted_manifest: Path
    model_dir: Path
    output: Path
    track_id: int
    camera_ids: tuple[str, ...] | None = None
    reference_start: int | None = None
    reference_end: int | None = None
    max_samples_per_mask: int = 512
    max_pairs: int | None = None
    feature_level: str = "deep"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.max_samples_per_mask <= 0:
            raise ValueError("track_id and max_samples_per_mask are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if self.feature_level not in {"high", "mid", "deep"}:
            raise ValueError("feature_level must be high, mid or deep")
        if self.output.suffix != ".json":
            raise ValueError("output must have a .json suffix")


def audit_dynamic_feature_correspondence(options: DynamicFeatureCorrespondenceOptions) -> Path:
    """Audit adjacent trusted source masks; no correspondence is used as a label."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {options.output}")
    manifest_path = options.trusted_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted manifest must be complete raw dynamic supervision")
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    allowed_cameras = None if options.camera_ids is None else set(options.camera_ids)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if options.reference_start is not None and reference < options.reference_start:
            continue
        if options.reference_end is not None and reference >= options.reference_end:
            continue
        for camera in frame["cameras"]:
            camera_id = str(camera["camera_id"])
            if allowed_cameras is not None and camera_id not in allowed_cameras:
                continue
            instance = next((item for item in camera["instances"] if int(item["track_id"]) == options.track_id and item.get("mask_origin") == "source"), None)
            if instance is None:
                continue
            grouped.setdefault(camera_id, []).append({"reference": reference, "source": int(camera["source_frame_index"]), "instance": instance})
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for camera_id, records in grouped.items():
        records.sort(key=lambda value: value["reference"])
        for earlier, later in zip(records, records[1:]):
            if later["reference"] == earlier["reference"] + 1:
                pairs.append((camera_id, earlier, later))
    if options.max_pairs is not None:
        pairs = pairs[:options.max_pairs]
    if not pairs:
        raise RuntimeError("no consecutive source-mask pairs in the requested window")
    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    torch, model, processor, dtype = _load_sam2_feature_encoder(options.model_dir, options.device)
    results: list[dict[str, Any]] = []
    for index, (camera_id, earlier, later) in enumerate(pairs, start=1):
        sensor = loader.get_camera_sensor(camera_id)
        image_a = np.asarray(sensor.get_frame_image_array(earlier["source"]), np.uint8)
        image_b = np.asarray(sensor.get_frame_image_array(later["source"]), np.uint8)
        if image_a.shape != image_b.shape:
            raise ValueError(f"camera image shapes changed: {camera_id}")
        mask_a = np.asarray(Image.open(roots[str(earlier["instance"]["mask_root"])] / str(earlier["instance"]["mask"])).convert("L"), np.uint8) > 0
        mask_b = np.asarray(Image.open(roots[str(later["instance"]["mask_root"])] / str(later["instance"]["mask"])).convert("L"), np.uint8) > 0
        if mask_a.shape != image_a.shape[:2] or mask_b.shape != image_a.shape[:2]:
            raise ValueError(f"source mask/image shape mismatch: {camera_id}")
        pixels_a = sample_mask_pixels(mask_a, options.max_samples_per_mask)
        pixels_b = sample_mask_pixels(mask_b, options.max_samples_per_mask)
        if not len(pixels_a) or not len(pixels_b):
            continue
        (feature_a, feature_b), feature_shape = _sam2_feature_maps(
            torch, model, processor, dtype, [Image.fromarray(image_a), Image.fromarray(image_b)], options.device, options.feature_level,
        )
        metrics = mutual_feature_matches(feature_a, feature_b, pixels_a, pixels_b, image_a.shape[:2])
        results.append({"camera_id": camera_id, "reference_frames": [earlier["reference"], later["reference"]],
                        "source_frames": [earlier["source"], later["source"]], "feature_shape": list(feature_shape), **metrics})
        print(f"SAM2 correspondence [{index}/{len(pairs)}] {camera_id} ref={earlier['reference']}->{later['reference']} mutual={metrics['mutual_source_fraction']:.3f} matches={metrics['match_count']}", flush=True)
    if not results:
        raise RuntimeError("all requested source masks were empty")
    summary = {
        "pair_count": len(results),
        "mean_mutual_source_fraction": float(np.mean([item["mutual_source_fraction"] for item in results])),
        "mean_cosine_similarity": float(np.mean([item.get("mean_cosine_similarity", float("nan")) for item in results])),
        "median_p95_motion_residual_pixels": float(np.nanmedian([item.get("p95_motion_residual_pixels", float("nan")) for item in results])),
    }
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "trusted_manifest": str(manifest_path),
              "model_dir": str(options.model_dir.resolve()), "model_loader": "transformers Sam2VideoModel local_files_only",
              "track_id": options.track_id, "parameters": {"camera_ids": None if options.camera_ids is None else list(options.camera_ids), "reference_start": options.reference_start, "reference_end": options.reference_end, "max_samples_per_mask": options.max_samples_per_mask, "max_pairs": options.max_pairs, "feature_level": options.feature_level},
              "summary": summary, "pairs": results,
              "limitations": ["Mutual temporal feature matches are an evidence audit, not pseudo-labels or 3D points.", "No static PLY, dynamic NPZ, SAM2-only mask, actor asset, or renderer output was loaded or changed.", "Only after adequate temporal correspondence is shown should a separate FTheta ray-triangulation/visibility experiment be considered."]}
    _atomic_json(options.output, report)
    print(f"Wrote SAM2 temporal correspondence audit: {options.output}", flush=True)
    return options.output
