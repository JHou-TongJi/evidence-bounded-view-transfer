"""Reversible quality-priority compositing of two static renderer outputs.

This is deliberately an *image-space* diagnostic/production adapter, not a
PLY merger: each source PLY is rendered with the identical target path first,
then the higher-fidelity ``primary`` geometry owns any pixel for which it has
valid alpha and expected depth.  The coverage-oriented ``fallback`` only
fills missing primary geometry.  This prevents a soft blend from turning two
incompatible reconstructions into additional road/tree ghosting.
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


SCHEMA = "nurec-gs-static-layer-fusion"
VERSION = 1


@dataclass(frozen=True)
class StaticLayerFusionOptions:
    primary_root: Path
    fallback_root: Path
    output_root: Path
    primary_alpha_threshold: float = 0.01
    maximum_relative_depth_difference: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_alpha_threshold <= 1.0:
            raise ValueError("primary_alpha_threshold must be in [0, 1]")
        if self.maximum_relative_depth_difference is not None and self.maximum_relative_depth_difference < 0.0:
            raise ValueError("maximum_relative_depth_difference must be non-negative or None")


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
    result = {path.name: path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()}
    if not result:
        raise ValueError(f"no render manifests found below {root}")
    return result


def _frame_outputs(directory: Path, frame: dict[str, object]) -> dict[str, Path]:
    outputs = frame.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"frame has no output table in {directory / 'manifest.json'}")
    result: dict[str, Path] = {}
    for key, value in outputs.items():
        path = directory / str(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(key)] = path
    for required in ("rgb", "alpha", "depth_npy"):
        if required not in result:
            raise ValueError(f"missing required {required} in {directory / 'manifest.json'}")
    return result


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _atomic_png(array: np.ndarray, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(array, mode=mode).save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, array.astype(np.float32, copy=False), allow_pickle=False)
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


def _validate_frames(primary: dict[str, object], fallback: dict[str, object], unit: str) -> list[tuple[dict[str, object], dict[str, object]]]:
    a = [frame for frame in primary["frames"] if isinstance(frame, dict)]
    b = [frame for frame in fallback["frames"] if isinstance(frame, dict)]
    if len(a) != len(b):
        raise ValueError(f"frame count mismatch for unit {unit!r}: {len(a)} != {len(b)}")
    for index, (left, right) in enumerate(zip(a, b, strict=True)):
        if left.get("frame_id") != right.get("frame_id") or left.get("timestamp_us") != right.get("timestamp_us"):
            raise ValueError(f"frame mismatch for unit {unit!r} at index {index}")
    return list(zip(a, b, strict=True))


def _fuse_unit(primary_dir: Path, fallback_dir: Path, output_dir: Path, options: StaticLayerFusionOptions) -> dict[str, object]:
    primary_manifest = _read_manifest(primary_dir)
    fallback_manifest = _read_manifest(fallback_dir)
    frames: list[dict[str, object]] = []
    ownership: list[float] = []
    deferred: list[float] = []
    for index, (primary_frame, fallback_frame) in enumerate(_validate_frames(primary_manifest, fallback_manifest, output_dir.name)):
        primary_paths = _frame_outputs(primary_dir, primary_frame)
        fallback_paths = _frame_outputs(fallback_dir, fallback_frame)
        primary_rgb, fallback_rgb = _read_rgb(primary_paths["rgb"]), _read_rgb(fallback_paths["rgb"])
        primary_alpha, fallback_alpha = _read_alpha(primary_paths["alpha"]), _read_alpha(fallback_paths["alpha"])
        primary_depth = np.asarray(np.load(primary_paths["depth_npy"], allow_pickle=False), dtype=np.float32)
        fallback_depth = np.asarray(np.load(fallback_paths["depth_npy"], allow_pickle=False), dtype=np.float32)
        shape = primary_alpha.shape
        if (primary_rgb.shape[:2] != shape or fallback_rgb.shape != primary_rgb.shape or fallback_alpha.shape != shape
                or primary_depth.shape != shape or fallback_depth.shape != shape):
            raise ValueError(f"RGB/alpha/depth shape mismatch for {output_dir.name} frame {index}")
        primary_geometry = (primary_alpha >= options.primary_alpha_threshold) & np.isfinite(primary_depth)
        fallback_geometry = (fallback_alpha >= options.primary_alpha_threshold) & np.isfinite(fallback_depth)
        both_geometry = primary_geometry & fallback_geometry
        depth_conflict = np.zeros(shape, dtype=bool)
        if options.maximum_relative_depth_difference is not None:
            relative_depth_difference = np.zeros(shape, dtype=np.float32)
            relative_depth_difference[both_geometry] = (
                np.abs(primary_depth[both_geometry] - fallback_depth[both_geometry])
                / np.maximum(np.minimum(primary_depth[both_geometry], fallback_depth[both_geometry]), 1e-3)
            )
            depth_conflict = both_geometry & (relative_depth_difference > options.maximum_relative_depth_difference)
        # A primary hole always falls back.  With the optional consistency gate,
        # a disagreeing pair also falls back as one coherent coverage layer
        # instead of splicing two different road surfaces along their boundary.
        choose_primary = primary_geometry & ~depth_conflict
        rgb = np.where(choose_primary[..., None], primary_rgb, fallback_rgb)
        alpha = np.where(choose_primary, primary_alpha, fallback_alpha)
        depth = np.where(choose_primary, primary_depth, fallback_depth)
        stem = f"{index:06d}_{primary_frame.get('frame_id', index)}"
        _atomic_png(rgb, output_dir / "rgb" / f"{stem}.png", "RGB")
        _atomic_png(np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), output_dir / "alpha" / f"{stem}.png", "L")
        _atomic_npy(depth, output_dir / "depth" / f"{stem}.npy")
        outputs: dict[str, str] = {
            "rgb": f"rgb/{stem}.png", "alpha": f"alpha/{stem}.png", "depth_npy": f"depth/{stem}.npy",
        }
        # Sky carries no geometry.  Copy fallback debug layers so coverage audits keep
        # their original geometry/sky/unknown semantics.
        for key in ("sky_valid", "sky_rgb"):
            source = fallback_paths.get(key)
            if source is not None:
                suffix = source.suffix.lower()
                target = output_dir / key / f"{stem}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                outputs[key] = str(target.relative_to(output_dir))
        frame = dict(primary_frame)
        frame["outputs"] = outputs
        frame["static_layer_fusion"] = {
            "primary_geometry_fraction": float(primary_geometry.mean()),
            "fallback_geometry_fraction": float(fallback_geometry.mean()),
            "primary_deferred_depth_conflict_fraction": float(depth_conflict.mean()),
        }
        frames.append(frame)
        ownership.append(float(primary_geometry.mean()))
        deferred.append(float(depth_conflict.mean()))
    manifest = dict(primary_manifest)
    manifest.update({
        "schema_version": 2,
        "status": "complete",
        "frame_count": len(frames),
        "completed_frame_count": len(frames),
        "frames": frames,
        "output_components": ["rgb", "alpha", "depth"],
        "static_layer_fusion": {
            "schema": SCHEMA,
            "version": VERSION,
            "primary_render_directory": str(primary_dir),
            "fallback_render_directory": str(fallback_dir),
            "primary_alpha_threshold": options.primary_alpha_threshold,
            "maximum_relative_depth_difference": options.maximum_relative_depth_difference,
            "primary_geometry_fraction_mean": float(np.mean(ownership)),
            "primary_deferred_depth_conflict_fraction_mean": float(np.mean(deferred)),
            "policy": "hard quality-priority: primary owns valid geometry except optional depth conflicts; fallback fills all remaining pixels; no RGB blending",
        },
    })
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest["static_layer_fusion"]


def fuse_static_render_layers(options: StaticLayerFusionOptions) -> dict[str, object]:
    """Fuse matching single-camera or named-rig renderer directories."""
    primary_units = _render_units(options.primary_root)
    fallback_units = _render_units(options.fallback_root)
    if set(primary_units) != set(fallback_units):
        raise ValueError(f"primary/fallback camera units differ: {sorted(primary_units)} != {sorted(fallback_units)}")
    output_root = Path(options.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    unit_reports: dict[str, object] = {}
    multiple = len(primary_units) > 1
    for name in sorted(primary_units):
        output_dir = output_root / name if multiple else output_root
        unit_reports[name or "camera"] = _fuse_unit(primary_units[name], fallback_units[name], output_dir, options)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "primary_root": str(options.primary_root),
        "fallback_root": str(options.fallback_root),
        "output_root": str(output_root),
        "primary_alpha_threshold": options.primary_alpha_threshold,
        "maximum_relative_depth_difference": options.maximum_relative_depth_difference,
        "units": unit_reports,
    }
    _atomic_json(report, output_root / "fusion_report.json")
    return report
