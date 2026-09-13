"""Build a LiDAR-anchored Depth Anything V2 prior for a static 2DGS proxy.

The prior deliberately has a narrower role than a dense depth reconstruction:
it is predicted on the already rectified virtual-pinhole source views, calibrated
only from *training* sparse road LiDAR returns, and supervised only in a soft
neighbourhood of those returns.  Dynamic/invalid source pixels remain excluded
through the proxy image alpha channel.  This avoids treating a monocular model
as geometry truth in unobserved or moving-object regions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "ncore-2dgs-depth-anything-prior"


@dataclass(frozen=True)
class TwoDgsDepthAnythingOptions:
    source_path: Path
    model_dir: Path
    output: Path
    device: str = "cuda"
    batch_size: int = 4
    max_records: int | None = None
    min_lidar_pixels: int = 128
    support_radius_px: float = 16.0
    max_log_gradient: float = 0.22
    min_depth_m: float = 2.0
    max_depth_m: float = 60.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.min_lidar_pixels <= 0:
            raise ValueError("batch_size and min_lidar_pixels must be positive")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive when supplied")
        if self.support_radius_px <= 0.0 or self.max_log_gradient <= 0.0:
            raise ValueError("support_radius_px and max_log_gradient must be positive")
        if self.min_depth_m <= 0.0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("invalid depth range")


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ncore-2dgs-pinhole-proxy" or not isinstance(payload.get("records"), list):
        raise ValueError(f"unsupported NCore 2DGS proxy manifest: {path}")
    return payload


def _key(record: dict[str, Any]) -> str:
    return "{}|{:.5f}|{:.5f}".format(
        record["source_camera"], float(record["tile_yaw_deg"]), float(record["tile_pitch_deg"]),
    )


def _read_rgba_static_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        if "A" not in opened.getbands():
            return np.ones((opened.height, opened.width), dtype=bool)
        return np.asarray(opened.getchannel("A"), dtype=np.uint8) > 0


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def _read_road_target(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"depth", "confidence", "split"}.issubset(payload.files):
            raise ValueError(f"unsupported road LiDAR target: {path}")
        depth = np.asarray(payload["depth"], dtype=np.float32)
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        split = str(payload["split"].item())
    if depth.ndim != 2 or confidence.shape != depth.shape or split not in {"train", "holdout"}:
        raise ValueError(f"invalid road LiDAR target: {path}")
    return depth, confidence, split


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = float(weights.sum()) * .5
    return float(values[min(int(np.searchsorted(np.cumsum(weights), cutoff, side="left")), len(values) - 1)])


def _depth_confidence(depth: np.ndarray, static_mask: np.ndarray, anchor: np.ndarray, radius: float,
                      max_log_gradient: float, min_depth: float, max_depth: float) -> np.ndarray:
    """Restrict a monocular prior to smooth, static road neighbourhoods."""
    from scipy.ndimage import distance_transform_edt

    if depth.shape != static_mask.shape or anchor.shape != depth.shape:
        raise ValueError("depth prior shapes do not match")
    log_depth = np.log(np.clip(depth, min_depth, max_depth))
    dy = np.zeros_like(log_depth); dx = np.zeros_like(log_depth)
    dy[1:] = np.abs(log_depth[1:] - log_depth[:-1])
    dx[:, 1:] = np.abs(log_depth[:, 1:] - log_depth[:, :-1])
    gradient = np.maximum.reduce((dy, np.roll(dy, -1, axis=0), dx, np.roll(dx, -1, axis=1)))
    distance = distance_transform_edt(~anchor)
    support = np.exp(-0.5 * (distance / radius) ** 2).astype(np.float32)
    valid = static_mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth) & (gradient <= max_log_gradient)
    return support * valid.astype(np.float32)


def _symlink_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source, destination, target_is_directory=True)


def build_two_dgs_depth_anything_prior(options: TwoDgsDepthAnythingOptions) -> Path:
    """Predict, LiDAR-calibrate and materialise a non-destructive depth-prior dataset."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    source = options.source_path.resolve()
    model_dir = options.model_dir.resolve()
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite depth-prior output: {output}")
    manifest = _read_manifest(source / "ncore_2dgs_manifest.json")
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"not a local Transformers Depth Anything model: {model_dir}")
    records = [record for record in manifest["records"] if record.get("road_depth")]
    if options.max_records is not None:
        records = records[:options.max_records]
    if not records:
        raise ValueError("source proxy has no sparse road LiDAR targets to anchor DAV2")

    output.mkdir(parents=True)
    _symlink_directory(source / "images", output / "images")
    _symlink_directory(source / "sparse", output / "sparse")
    if (source / "road_depth").is_dir():
        _symlink_directory(source / "road_depth", output / "road_depth")
    prior_dir = output / "depth_anything_prior"; prior_dir.mkdir()

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).to(options.device).eval()
    print(f"Loaded local DAV2 model: {model_dir}", flush=True)

    calibration_values: dict[str, list[np.ndarray]] = {}
    calibration_weights: dict[str, list[np.ndarray]] = {}
    raw_index: list[tuple[dict[str, Any], Path, np.ndarray, np.ndarray, str]] = []
    with torch.inference_mode():
        for start in range(0, len(records), options.batch_size):
            batch = records[start:start + options.batch_size]
            images = [_read_rgb(source / str(record["image"])) for record in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(options.device, non_blocking=True) for name, value in inputs.items()}
            prediction = model(**inputs).predicted_depth[:, None]
            for offset, (record, image) in enumerate(zip(batch, images)):
                target_h, target_w = image.shape[:2]
                depth = functional.interpolate(prediction[offset:offset + 1], size=(target_h, target_w), mode="bicubic", align_corners=False)[0, 0]
                depth_np = depth.float().cpu().numpy().astype(np.float32)
                road = record["road_depth"]
                target, anchor_weight, split = _read_road_target(source / str(road["file"]))
                if target.shape != depth_np.shape:
                    raise ValueError(f"DAV2/road target shape mismatch: {record['image']}")
                static_mask = _read_rgba_static_mask(source / str(record["image"]))
                anchor = (anchor_weight > 0.0) & (target >= options.min_depth_m) & (target <= options.max_depth_m) & static_mask
                prior_path = prior_dir / f"{Path(str(record['image'])).stem}.npz"
                np.savez_compressed(prior_path, raw_depth=depth_np)
                raw_index.append((record, prior_path, target, anchor_weight, split))
                if split == "train" and int(anchor.sum()) >= options.min_lidar_pixels:
                    log_ratio = np.log(target[anchor]) - np.log(np.clip(depth_np[anchor], options.min_depth_m, options.max_depth_m))
                    stride = max(1, int(np.ceil(log_ratio.size / 1024)))
                    key = _key(record)
                    calibration_values.setdefault(key, []).append(log_ratio[::stride])
                    calibration_weights.setdefault(key, []).append(anchor_weight[anchor][::stride].astype(np.float32))
            done = min(start + len(batch), len(records))
            if done == len(records) or done == 1 or done % 100 == 0:
                print(f"DAV2 prior [{done}/{len(records)}]", flush=True)

    all_values = [value for values in calibration_values.values() for value in values]
    all_weights = [value for values in calibration_weights.values() for value in values]
    if not all_values:
        raise ValueError("no training LiDAR pixels passed DAV2 calibration gate")
    global_log_scale = _weighted_median(np.concatenate(all_values), np.concatenate(all_weights))
    scales: dict[str, float] = {}
    for key, values in calibration_values.items():
        weights = calibration_weights[key]
        if int(sum(value.size for value in values)) < options.min_lidar_pixels:
            scales[key] = float(np.exp(global_log_scale))
        else:
            scales[key] = float(np.exp(_weighted_median(np.concatenate(values), np.concatenate(weights))))

    metrics: dict[str, list[np.ndarray]] = {"train_raw": [], "train_calibrated": [], "holdout_raw": [], "holdout_calibrated": []}
    updated_records: list[dict[str, Any]] = []
    by_image: dict[str, dict[str, Any]] = {}
    for record, prior_path, target, anchor_weight, split in raw_index:
        with np.load(prior_path, allow_pickle=False) as payload:
            raw_depth = np.asarray(payload["raw_depth"], dtype=np.float32)
        scale = scales.get(_key(record), float(np.exp(global_log_scale)))
        # DAV2 may emit arbitrary signed values outside its reliable range.
        # Those pixels have zero support below, but the official 2DGS loader
        # validates the complete stored tensor; keep its inactive portion
        # finite and physically bounded as well.
        calibrated = np.clip(raw_depth * scale, options.min_depth_m, options.max_depth_m).astype(np.float32)
        static_mask = _read_rgba_static_mask(source / str(record["image"]))
        anchor = (anchor_weight > 0.0) & (target >= options.min_depth_m) & (target <= options.max_depth_m) & static_mask
        if int(anchor.sum()) >= options.min_lidar_pixels:
            sample = anchor_weight > 0.0
            raw_error = np.abs(raw_depth[sample] - target[sample])
            calibrated_error = np.abs(calibrated[sample] - target[sample])
            metrics[f"{split}_raw"].append(raw_error[::max(1, int(np.ceil(raw_error.size / 4096)))])
            metrics[f"{split}_calibrated"].append(calibrated_error[::max(1, int(np.ceil(calibrated_error.size / 4096)))])
        confidence = _depth_confidence(calibrated, static_mask, anchor, options.support_radius_px,
                                       options.max_log_gradient, options.min_depth_m, options.max_depth_m)
        np.savez_compressed(prior_path, depth=calibrated.astype(np.float32), confidence=confidence.astype(np.float32),
                            split=np.asarray(split), calibration_scale=np.asarray(scale, dtype=np.float32),
                            raw_depth=raw_depth.astype(np.float32))
        copied = dict(record)
        copied["depth_prior"] = {"file": f"depth_anything_prior/{prior_path.name}", "split": split,
                                 "valid_pixels": int(np.count_nonzero(confidence > 0.0)), "calibration_key": _key(record)}
        by_image[str(record["image"])] = copied

    for record in manifest["records"]:
        updated_records.append(by_image.get(str(record["image"]), dict(record)))
    report_metrics: dict[str, dict[str, float | int]] = {}
    for name, groups in metrics.items():
        values = np.concatenate(groups) if groups else np.empty(0, dtype=np.float32)
        report_metrics[name] = {"pixels": int(values.size), "mae_m": float(values.mean()) if values.size else float("nan"),
                                "median_m": float(np.median(values)) if values.size else float("nan"),
                                "p90_m": float(np.quantile(values, .9)) if values.size else float("nan")}
    updated_manifest = dict(manifest)
    updated_manifest["records"] = updated_records
    updated_manifest["depth_anything_prior"] = {
        "schema": SCHEMA, "model_dir": str(model_dir), "model_config": str(getattr(model.config, "_name_or_path", "local")),
        "calibration": "per virtual tile weighted median log-scale from sparse train road LiDAR only",
        "global_scale": float(np.exp(global_log_scale)), "per_tile_scales": scales,
        "support": {"road_lidar_neighbourhood_radius_px": options.support_radius_px,
                    "max_log_depth_gradient": options.max_log_gradient, "static_alpha_required": True},
        "records_with_prior": len(raw_index), "metrics": report_metrics,
        "limitations": ["DAV2 is a monocular pseudo-depth prior, not a new observation.",
                        "Calibration uses train split LiDAR only; holdout is reported but never optimised.",
                        "Prior supervision is restricted to static smooth neighbourhoods of raw road LiDAR, and cannot supervise vehicles, people, sky or unobserved geometry."],
    }
    (output / "ncore_2dgs_manifest.json").write_text(json.dumps(updated_manifest, indent=2), encoding="utf-8")
    report = output / "depth_anything_report.json"
    report.write_text(json.dumps(updated_manifest["depth_anything_prior"], indent=2), encoding="utf-8")
    print(f"Wrote LiDAR-anchored DAV2 proxy dataset: {output}", flush=True)
    return report
