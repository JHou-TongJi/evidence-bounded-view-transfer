"""Read-only coverage audit for a rendered target-camera rig.

The renderer deliberately keeps geometry alpha, sky validity, and expected
depth separate.  This module turns those existing debug outputs into a
three-way per-pixel contract for target-rig planning:

* ``geometry``: supported by the Gaussian scene and has finite expected depth;
* ``sky``: no geometry, but a valid direction-domain sky observation exists;
* ``unknown``: neither of the above, so it must not be treated as usable road
  imagery or silently filled with a black fallback.

It never renders, trains, or changes a PLY.  It can therefore be used to
compare candidate PLY renders before producing a long L4 sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


SCHEMA = "nurec-gs-target-rig-coverage-audit"
VERSION = 1


@dataclass(frozen=True)
class TargetCoverageAuditOptions:
    render_root: Path
    output: Path
    alpha_threshold: float = 0.01
    sky_valid_threshold: float = 0.5
    minimum_geometric_fraction: float = 0.20
    minimum_lower_half_geometric_fraction: float = 0.35
    maximum_unknown_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in ("alpha_threshold", "sky_valid_threshold", "minimum_geometric_fraction",
                     "minimum_lower_half_geometric_fraction", "maximum_unknown_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def _load_manifest(directory: Path) -> dict[str, object]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"sequence manifest is not complete: {path}")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"manifest has no completed frames: {path}")
    return value


def _output_path(directory: Path, frame: dict[str, object], key: str, *, required: bool) -> Path | None:
    outputs = frame.get("outputs")
    value = outputs.get(key) if isinstance(outputs, dict) else None
    if value is None:
        if required:
            raise ValueError(f"missing required {key} output in {directory / 'manifest.json'}")
        return None
    path = directory / str(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_coverage_map(geometry: np.ndarray, sky: np.ndarray, unknown: np.ndarray, path: Path) -> None:
    # Green=depth-supported geometry, blue=valid cubemap sky, red=unknown.
    image = np.zeros(geometry.shape + (3,), dtype=np.uint8)
    image[geometry] = (38, 166, 91)
    image[sky] = (78, 144, 255)
    image[unknown] = (220, 62, 62)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(image, mode="RGB").save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mean(records: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def _frame_coverage(
    camera_directory: Path,
    frame: dict[str, object],
    options: TargetCoverageAuditOptions,
    coverage_directory: Path,
    index: int,
) -> dict[str, object]:
    alpha_path = _output_path(camera_directory, frame, "alpha", required=True)
    alpha = _read_mask(alpha_path)
    depth_path = _output_path(camera_directory, frame, "depth_npy", required=False)
    if depth_path is None:
        finite_depth = np.zeros(alpha.shape, dtype=bool)
        depth_available = False
    else:
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
        if depth.shape != alpha.shape:
            raise ValueError(f"alpha/depth shape mismatch in {camera_directory}: {alpha.shape} != {depth.shape}")
        finite_depth = np.isfinite(depth)
        depth_available = True

    alpha_geometry = alpha >= options.alpha_threshold
    geometry = alpha_geometry & finite_depth
    sky_path = _output_path(camera_directory, frame, "sky_valid", required=False)
    if sky_path is None:
        sky_valid = np.zeros(alpha.shape, dtype=bool)
        sky_available = False
    else:
        sky_value = _read_mask(sky_path)
        if sky_value.shape != alpha.shape:
            raise ValueError(f"alpha/sky shape mismatch in {camera_directory}: {alpha.shape} != {sky_value.shape}")
        sky_valid = sky_value >= options.sky_valid_threshold
        sky_available = True
    sky = ~geometry & sky_valid
    unknown = ~(geometry | sky)
    lower = slice(alpha.shape[0] // 2, None)
    name = f"{index:06d}_{str(frame.get('frame_id', index))}.png"
    map_path = coverage_directory / name
    _write_coverage_map(geometry, sky, unknown, map_path)
    return {
        "index": index,
        "frame_id": frame.get("frame_id"),
        "timestamp_us": frame.get("timestamp_us"),
        "geometry_fraction": float(geometry.mean()),
        "sky_fraction": float(sky.mean()),
        "unknown_fraction": float(unknown.mean()),
        "lower_half_geometry_fraction": float(geometry[lower].mean()),
        "lower_half_unknown_fraction": float(unknown[lower].mean()),
        "alpha_geometry_fraction": float(alpha_geometry.mean()),
        "geometry_depth_consistency": float(geometry.sum() / max(int(alpha_geometry.sum()), 1)),
        "depth_available": depth_available,
        "sky_available": sky_available,
        "coverage_map": str(map_path),
    }


def _camera_summary(records: list[dict[str, object]], options: TargetCoverageAuditOptions) -> dict[str, object]:
    summary = {
        "frame_count": len(records),
        "geometry_fraction_mean": _mean(records, "geometry_fraction"),
        "geometry_fraction_min": float(min(float(record["geometry_fraction"]) for record in records)),
        "sky_fraction_mean": _mean(records, "sky_fraction"),
        "unknown_fraction_mean": _mean(records, "unknown_fraction"),
        "unknown_fraction_max": float(max(float(record["unknown_fraction"]) for record in records)),
        "lower_half_geometry_fraction_mean": _mean(records, "lower_half_geometry_fraction"),
        "lower_half_unknown_fraction_mean": _mean(records, "lower_half_unknown_fraction"),
        "geometry_depth_consistency_mean": _mean(records, "geometry_depth_consistency"),
    }
    geometry_ok = (
        summary["geometry_fraction_mean"] >= options.minimum_geometric_fraction
        and summary["lower_half_geometry_fraction_mean"] >= options.minimum_lower_half_geometric_fraction
        and summary["geometry_depth_consistency_mean"] >= 0.995
    )
    rgb_ok = summary["unknown_fraction_mean"] <= options.maximum_unknown_fraction
    summary["geometric_road_ready"] = bool(geometry_ok)
    summary["rgb_coverage_ready"] = bool(rgb_ok)
    summary["status"] = "ready" if geometry_ok and rgb_ok else "insufficient_coverage"
    return summary


def audit_target_rig_coverage(options: TargetCoverageAuditOptions) -> dict[str, object]:
    """Audit every completed named-camera renderer directory below ``render_root``."""
    root = Path(options.render_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    camera_directories = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()
    )
    if not camera_directories:
        raise ValueError(f"no named camera render directories found below {root}")
    output = Path(options.output)
    maps_root = output.parent / f"{output.stem}_maps"
    cameras: dict[str, object] = {}
    all_records: list[dict[str, object]] = []
    for directory in camera_directories:
        manifest = _load_manifest(directory)
        records = [
            _frame_coverage(directory, frame, options, maps_root / directory.name, index)
            for index, frame in enumerate(manifest["frames"])
            if isinstance(frame, dict)
        ]
        if not records:
            raise ValueError(f"manifest has no usable frame records: {directory / 'manifest.json'}")
        cameras[directory.name] = {
            "render_directory": str(directory),
            "path_metadata": manifest.get("path_metadata"),
            "summary": _camera_summary(records, options),
            "frames": records,
        }
        all_records.extend(records)
    all_ready = all(bool(value["summary"]["geometric_road_ready"]) for value in cameras.values())
    report: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": "Read-only target-L4 rig static coverage audit; red pixels are explicitly unknown and must not be used as geometry-supported road imagery.",
        "render_root": str(root),
        "thresholds": {
            "alpha": options.alpha_threshold,
            "sky_valid": options.sky_valid_threshold,
            "minimum_geometric_fraction": options.minimum_geometric_fraction,
            "minimum_lower_half_geometric_fraction": options.minimum_lower_half_geometric_fraction,
            "maximum_unknown_fraction": options.maximum_unknown_fraction,
        },
        "camera_count": len(cameras),
        "all_cameras_geometric_road_ready": all_ready,
        "overall": {
            "geometry_fraction_mean": _mean(all_records, "geometry_fraction"),
            "sky_fraction_mean": _mean(all_records, "sky_fraction"),
            "unknown_fraction_mean": _mean(all_records, "unknown_fraction"),
            "lower_half_geometry_fraction_mean": _mean(all_records, "lower_half_geometry_fraction"),
            "lower_half_unknown_fraction_mean": _mean(all_records, "lower_half_unknown_fraction"),
        },
        "cameras": cameras,
        "limitations": [
            "This audit measures rendered support, not photometric fidelity or dynamic-object correctness.",
            "Valid sky is RGB support only and never counts as road geometry or depth support.",
            "Unknown pixels must remain invalid for geometric labels; this tool does not fill them.",
        ],
    }
    _write_json_atomic(report, output)
    return report
