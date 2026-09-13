"""Conservative RGB-only presentation refinement for completed render runs.

This module deliberately does *not* alter a Gaussian scene, camera model,
alpha, or expected depth.  It is useful for short visual demonstrations where
the projected road surface contains screen-space splat/moire texture.  The
result is explicitly a display derivative, not a replacement camera-supervision
image or a new reconstruction asset.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


SCHEMA = "nurec-gs-presentation-refinement"
VERSION = 1


@dataclass(frozen=True)
class PresentationRefineOptions:
    input_root: Path
    output_root: Path
    lower_start_ratio: float = 0.43
    feather_ratio: float = 0.12
    alpha_threshold: float = 0.5
    strength: float = 0.78
    diameter: int = 9
    sigma_color: float = 35.0
    sigma_space: float = 7.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_start_ratio < 1.0:
            raise ValueError("lower_start_ratio must be in [0, 1)")
        if self.feather_ratio <= 0.0:
            raise ValueError("feather_ratio must be positive")
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1]")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if self.diameter <= 0 or self.diameter % 2 == 0:
            raise ValueError("diameter must be a positive odd integer")
        if self.sigma_color <= 0.0 or self.sigma_space <= 0.0:
            raise ValueError("bilateral sigmas must be positive")


def _read_manifest(directory: Path) -> dict[str, object]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"render manifest is not complete: {path}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"render manifest has no frames: {path}")
    return manifest


def _render_units(root: Path) -> dict[str, Path]:
    root = Path(root)
    if (root / "manifest.json").is_file():
        return {"": root}
    if not root.is_dir():
        raise NotADirectoryError(root)
    units = {item.name: item for item in root.iterdir() if item.is_dir() and (item / "manifest.json").is_file()}
    if not units:
        raise ValueError(f"no complete render manifests found below {root}")
    return units


def _atomic_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(image, "RGB").save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _bilateral_filter_bgr(image: np.ndarray, options: PresentationRefineOptions) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError(
            "presentation refinement requires OpenCV; install the optional 'presentation' dependency"
        ) from exc
    return cv2.bilateralFilter(image, options.diameter, options.sigma_color, options.sigma_space)


def refine_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    options: PresentationRefineOptions,
) -> np.ndarray:
    """Apply a weak lower-image bilateral filter only to valid geometry."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape [H,W,3] and uint8 dtype")
    if alpha.shape != rgb.shape[:2] or depth.shape != rgb.shape[:2]:
        raise ValueError("alpha/depth must match RGB height and width")
    # OpenCV is BGR while renderer PNG output is RGB.
    filtered = _bilateral_filter_bgr(rgb[..., ::-1], options)[..., ::-1]
    height = rgb.shape[0]
    rows = np.arange(height, dtype=np.float32)[:, None] / max(float(height - 1), 1.0)
    vertical_weight = np.clip(
        (rows - options.lower_start_ratio) / options.feather_ratio,
        0.0,
        1.0,
    )
    geometry = (np.asarray(alpha, dtype=np.float32) >= options.alpha_threshold) & np.isfinite(depth)
    weight = vertical_weight * geometry.astype(np.float32) * options.strength
    result = rgb.astype(np.float32) * (1.0 - weight[..., None]) + filtered.astype(np.float32) * weight[..., None]
    return np.round(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def _frame_paths(input_dir: Path, frame: dict[str, object]) -> dict[str, Path]:
    outputs = frame.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"frame has no output table in {input_dir / 'manifest.json'}")
    paths = {str(key): input_dir / str(value) for key, value in outputs.items()}
    for required in ("rgb", "alpha", "depth_npy"):
        if required not in paths or not paths[required].is_file():
            raise ValueError(f"missing {required} for a frame in {input_dir / 'manifest.json'}")
    return paths


def _refine_unit(input_dir: Path, output_dir: Path, options: PresentationRefineOptions) -> dict[str, object]:
    manifest = _read_manifest(input_dir)
    frames: list[dict[str, object]] = []
    changed_fractions: list[float] = []
    for frame in manifest["frames"]:
        if not isinstance(frame, dict):
            raise ValueError(f"invalid frame record in {input_dir / 'manifest.json'}")
        paths = _frame_paths(input_dir, frame)
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(Image.open(paths["alpha"]).convert("L"), dtype=np.float32) / 255.0
        depth = np.asarray(np.load(paths["depth_npy"], allow_pickle=False), dtype=np.float32)
        refined = refine_rgb(rgb, alpha, depth, options)
        outputs = frame["outputs"]
        assert isinstance(outputs, dict)
        for key, relative in outputs.items():
            destination = output_dir / str(relative)
            if key == "rgb":
                _atomic_png(refined, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(paths[str(key)], destination)
        frames.append(dict(frame))
        changed_fractions.append(float(np.any(refined != rgb, axis=-1).mean()))
    result = dict(manifest)
    result.update({
        "status": "complete",
        "frames": frames,
        "frame_count": len(frames),
        "completed_frame_count": len(frames),
        "presentation_refinement": {
            "schema": SCHEMA,
            "version": VERSION,
            "input_render_directory": str(input_dir),
            "mode": "lower-geometry-bilateral-rgb-only",
            "lower_start_ratio": options.lower_start_ratio,
            "feather_ratio": options.feather_ratio,
            "alpha_threshold": options.alpha_threshold,
            "strength": options.strength,
            "diameter": options.diameter,
            "sigma_color": options.sigma_color,
            "sigma_space": options.sigma_space,
            "mean_changed_pixel_fraction": float(np.mean(changed_fractions)),
            "alpha_depth_policy": "copied byte-for-byte from input; RGB only is presentation-filtered",
            "usage_warning": "Display derivative only; do not use the refined RGB as sensor supervision or ground truth.",
        },
    })
    _atomic_json(result, output_dir / "manifest.json")
    return result["presentation_refinement"]


def refine_render_presentation(options: PresentationRefineOptions) -> dict[str, object]:
    """Refine every complete unit in a single camera or target-rig render root."""
    input_units = _render_units(options.input_root)
    output_root = Path(options.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    multiple = len(input_units) > 1
    for name in sorted(input_units):
        target = output_root / name if multiple else output_root
        report[name or "camera"] = _refine_unit(input_units[name], target, options)
    overall = {
        "schema": SCHEMA,
        "version": VERSION,
        "input_root": str(options.input_root),
        "output_root": str(output_root),
        "units": report,
        "usage_warning": "RGB-only display derivative; alpha/depth retain the original renderer semantics.",
    }
    _atomic_json(overall, output_root / "presentation_refinement_report.json")
    return overall
