"""Prepare and audit a strictly display-only ProPainter target-rig experiment.

This bridge deliberately removes dynamic-object remnants from an already
rendered target-camera video.  It does not claim to reconstruct, preserve, or
generate a vehicle/person, and never changes the renderer's RGB/alpha/depth
assets.  The prepared masks are projected *target-view* cuboid rectangles;
they are conservative removal regions, not instance ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _cube_corners
from .dynamic_reconstruction_dataset import SCHEMA as DATASET_SCHEMA


SCHEMA = "nurec-gs-propainter-display-demo"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _atomic_png(value: np.ndarray, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(value, mode=mode).save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def pinhole_cuboid_rectangle(
    corners_world: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    margin_pixels: int,
) -> np.ndarray:
    """Return a conservative target-pinhole rectangle for one 3-D cuboid."""
    if margin_pixels < 0:
        raise ValueError("margin_pixels must be non-negative")
    points = np.asarray(corners_world, dtype=np.float32)
    transform = np.asarray(world_to_camera, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("invalid cuboid/world-to-camera shapes")
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    depth = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth > .10)
    height, width = int(intrinsics["height"]), int(intrinsics["width"])
    mask = np.zeros((height, width), dtype=bool)
    if valid.sum() < 2:
        return mask
    pixels = np.empty((int(valid.sum()), 2), dtype=np.float32)
    pixels[:, 0] = float(intrinsics["fx"]) * camera[valid, 0] / depth[valid] + float(intrinsics["cx"]) - .5
    pixels[:, 1] = float(intrinsics["fy"]) * camera[valid, 1] / depth[valid] + float(intrinsics["cy"]) - .5
    xmin, ymin = np.floor(pixels.min(axis=0)).astype(np.int64) - margin_pixels
    xmax, ymax = np.ceil(pixels.max(axis=0)).astype(np.int64) + margin_pixels
    xmin, xmax = max(int(xmin), 0), min(int(xmax), width - 1)
    ymin, ymax = max(int(ymin), 0), min(int(ymax), height - 1)
    if xmin <= xmax and ymin <= ymax:
        mask[ymin : ymax + 1, xmin : xmax + 1] = True
    return mask


def background_unknown_mask(alpha: np.ndarray, sky_valid: np.ndarray, *, alpha_threshold: float, sky_valid_threshold: float) -> np.ndarray:
    """Select only pixels with neither geometric nor cubemap-background support."""
    geometry = np.asarray(alpha, np.float32)
    sky = np.asarray(sky_valid, np.float32)
    if geometry.shape != sky.shape or geometry.ndim != 2:
        raise ValueError("alpha and sky_valid must be matching 2-D arrays")
    if not 0.0 <= alpha_threshold <= 1.0 or not 0.0 <= sky_valid_threshold <= 1.0:
        raise ValueError("background thresholds must be in [0, 1]")
    return (geometry < alpha_threshold) & (sky < sky_valid_threshold)


def _reference_index(frame: dict[str, Any]) -> int:
    """Use the stable source reference encoded in a renderer frame ID.

    A sliced render restarts its ordinal ``index`` from zero, while the first
    numeric component of ``frame_id`` remains the source/L4 reference index.
    """
    try:
        return int(str(frame["frame_id"]).split("_", 1)[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("render frame_id must begin with a numeric reference index") from exc


@dataclass(frozen=True)
class ProPainterPrepareOptions:
    render_dir: Path
    dynamic_dataset_manifest: Path
    output: Path
    track_ids: tuple[int, ...]
    frame_start: int
    frame_end: int
    actor_mask_render_dir: Path | None = None
    actor_mask_allow_missing: bool = False
    actor_alpha_threshold: float = .02
    actor_mask_dilation_pixels: int = 2
    margin_pixels: int = 4
    max_mask_fraction: float = .20

    def __post_init__(self) -> None:
        if not self.track_ids or len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("track_ids must be unique and non-empty")
        if self.frame_start < 0 or self.frame_end <= self.frame_start or self.margin_pixels < 0:
            raise ValueError("frame/margin parameters are invalid")
        if not 0.0 <= self.actor_alpha_threshold <= 1.0 or self.actor_mask_dilation_pixels < 0:
            raise ValueError("actor alpha/mask dilation parameters are invalid")
        if not 0.0 < self.max_mask_fraction <= 1.0:
            raise ValueError("max_mask_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ProPainterBackgroundPrepareOptions:
    render_dir: Path
    output: Path
    frame_start: int
    frame_end: int
    alpha_threshold: float = .02
    sky_valid_threshold: float = .50
    dilation_pixels: int = 1
    max_mask_fraction: float = .08

    def __post_init__(self) -> None:
        if self.frame_start < 0 or self.frame_end <= self.frame_start or self.dilation_pixels < 0:
            raise ValueError("frame/dilation parameters are invalid")
        if not 0.0 <= self.alpha_threshold <= 1.0 or not 0.0 <= self.sky_valid_threshold <= 1.0:
            raise ValueError("background thresholds must be in [0, 1]")
        if not 0.0 < self.max_mask_fraction <= 1.0:
            raise ValueError("max_mask_fraction must be in (0, 1]")


def prepare_propainter_display_demo(options: ProPainterPrepareOptions) -> Path:
    """Copy a short L4 RGB window and write conservative dynamic-removal masks."""
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    render_root = options.render_dir.resolve()
    render = _read_json(render_root / "manifest.json")
    if render.get("status") != "complete" or not isinstance(render.get("frames"), list):
        raise ValueError("render-dir must contain a completed render manifest")
    dataset_path = options.dynamic_dataset_manifest.resolve()
    dataset = _read_json(dataset_path)
    if dataset.get("schema") != DATASET_SCHEMA or dataset.get("status") != "complete":
        raise ValueError("dynamic-dataset-manifest must be a completed NCore dynamic dataset")
    tracks = {int(item["track_id"]): item for item in dataset.get("tracks", []) if isinstance(item, dict) and "track_id" in item}
    missing = set(options.track_ids) - tracks.keys()
    if missing:
        raise ValueError(f"requested track IDs are absent from dataset: {sorted(missing)}")
    selected = [item for item in render["frames"] if item.get("complete") and options.frame_start <= int(item["index"]) < options.frame_end]
    selected.sort(key=lambda item: int(item["index"]))
    if not selected:
        raise ValueError("requested render frame window is empty")
    actor_frames: dict[int, dict[str, Any]] | None = None
    actor_root: Path | None = None
    if options.actor_mask_render_dir is not None:
        actor_root = options.actor_mask_render_dir.resolve()
        actor_manifest = _read_json(actor_root / "manifest.json")
        if actor_manifest.get("status") != "complete" or not isinstance(actor_manifest.get("frames"), list):
            raise ValueError("actor-mask-render-dir must contain a completed render manifest")
        actor_frames = {_reference_index(item): item for item in actor_manifest["frames"] if item.get("complete")}
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(selected):
        outputs = frame.get("outputs")
        if not isinstance(outputs, dict) or not isinstance(outputs.get("rgb"), str):
            raise ValueError("render frame lacks RGB output")
        rgb = np.asarray(Image.open(render_root / outputs["rgb"]).convert("RGB"), dtype=np.uint8)
        intrinsics = frame.get("intrinsics")
        if not isinstance(intrinsics, dict) or frame.get("camera_model") != "pinhole":
            raise ValueError("ProPainter display preparation currently requires target pinhole render frames")
        if rgb.shape[:2] != (int(intrinsics["height"]), int(intrinsics["width"])):
            raise ValueError("render RGB and manifest intrinsics dimensions disagree")
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        visible_tracks: list[int] = []
        mask_mode = "target_pinhole_cuboid_rectangle"
        if actor_frames is not None and actor_root is not None:
            actor = actor_frames.get(_reference_index(frame))
            outputs_actor = None if actor is None else actor.get("outputs")
            alpha_relative = outputs_actor.get("alpha") if isinstance(outputs_actor, dict) else None
            if not isinstance(alpha_relative, str):
                if not options.actor_mask_allow_missing:
                    raise ValueError(f"actor mask render lacks alpha for target frame index {frame['index']}")
                mask_mode = "actor_alpha_unavailable_transparent_context"
            else:
                alpha = np.asarray(Image.open(actor_root / alpha_relative).convert("L"), np.float32) / 255.0
                if alpha.shape != mask.shape:
                    raise ValueError("actor alpha and target RGB dimensions disagree")
                mask = alpha >= options.actor_alpha_threshold
                if options.actor_mask_dilation_pixels:
                    try:
                        from scipy.ndimage import binary_dilation
                    except ImportError as exc:  # pragma: no cover - environment-specific optional dependency
                        raise RuntimeError("actor-alpha ProPainter mask dilation requires scipy") from exc
                    mask = binary_dilation(mask, iterations=options.actor_mask_dilation_pixels)
                visible_tracks = list(options.track_ids) if mask.any() else []
                mask_mode = "actor_alpha_display_mask"
        else:
            for track_id in options.track_ids:
                corners = _cube_corners(tracks[track_id], int(frame["timestamp_us"]))
                if corners is None:
                    continue
                projected = pinhole_cuboid_rectangle(corners, np.asarray(frame["world_to_camera_start"]), intrinsics, margin_pixels=options.margin_pixels)
                if projected.any():
                    mask |= projected
                    visible_tracks.append(track_id)
        fraction = float(mask.mean())
        if fraction > options.max_mask_fraction:
            raise RuntimeError(f"frame index {frame['index']} projected removal mask {fraction:.2%} exceeds safety limit {options.max_mask_fraction:.2%}")
        name = f"{ordinal:04d}.png"
        _atomic_png(rgb, output / "frames" / name, "RGB")
        _atomic_png((mask.astype(np.uint8) * 255), output / "masks" / name, "L")
        preview = rgb.astype(np.float32)
        preview[mask] = preview[mask] * .35 + np.asarray((235., 52., 52.), np.float32) * .65
        _atomic_png(np.rint(preview).astype(np.uint8), output / "previews" / name, "RGB")
        records.append({"ordinal": ordinal, "input_frame_index": int(frame["index"]), "input_frame_id": str(frame["frame_id"]), "timestamp_us": int(frame["timestamp_us"]), "frame": f"frames/{name}", "mask": f"masks/{name}", "preview": f"previews/{name}", "visible_track_ids": visible_tracks, "mask_mode": mask_mode, "mask_fraction": fraction})
    manifest = {"schema": SCHEMA, "version": VERSION, "status": "complete", "kind": "dynamic_removal_background_inpainting_input", "render_dir": str(render_root), "dynamic_dataset_manifest": str(dataset_path), "track_ids": list(options.track_ids), "frame_start": options.frame_start, "frame_end": options.frame_end, "margin_pixels": options.margin_pixels, "actor_mask_render_dir": None if actor_root is None else str(actor_root), "actor_alpha_threshold": options.actor_alpha_threshold if actor_root is not None else None, "actor_mask_dilation_pixels": options.actor_mask_dilation_pixels if actor_root is not None else None, "actor_mask_allow_missing": options.actor_mask_allow_missing, "max_mask_fraction": options.max_mask_fraction, "frames": records, "limitations": ["Cuboid masks are conservative target-view rectangles; actor-alpha masks are display diagnostics and do not establish target-view geometry or visibility.", "Transparent actor-alpha context frames are not cleaned and may contain static-PLY leakage; they are supplied only to test temporal propagation.", "This asset is display-only ProPainter input. It must not be used as sensor truth, reconstruction supervision, PLY data, alpha, or expected depth.", "ProPainter fills background; it does not reconstruct the removed vehicle/person."]}
    _atomic_json(manifest, output / "manifest.json")
    print(f"Prepared ProPainter display-only input: {output} ({len(records)} frame(s), mean mask={np.mean([item['mask_fraction'] for item in records]):.2%})", flush=True)
    return output


def prepare_propainter_background_demo(options: ProPainterBackgroundPrepareOptions) -> Path:
    """Prepare a small, background-only unknown-pixel inpainting window."""
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    render_root = options.render_dir.resolve()
    render = _read_json(render_root / "manifest.json")
    if render.get("status") != "complete" or not isinstance(render.get("frames"), list):
        raise ValueError("render-dir must contain a completed render manifest")
    selected = [item for item in render["frames"] if item.get("complete") and options.frame_start <= int(item["index"]) < options.frame_end]
    selected.sort(key=lambda item: int(item["index"]))
    if not selected:
        raise ValueError("requested render frame window is empty")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(selected):
        outputs = frame.get("outputs")
        if not isinstance(outputs, dict) or not all(isinstance(outputs.get(name), str) for name in ("rgb", "alpha", "sky_valid")):
            raise ValueError("background ProPainter preparation requires RGB, alpha and sky_valid render outputs")
        rgb = np.asarray(Image.open(render_root / outputs["rgb"]).convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(Image.open(render_root / outputs["alpha"]).convert("L"), np.float32) / 255.0
        sky_valid = np.asarray(Image.open(render_root / outputs["sky_valid"]).convert("L"), np.float32) / 255.0
        mask = background_unknown_mask(alpha, sky_valid, alpha_threshold=options.alpha_threshold, sky_valid_threshold=options.sky_valid_threshold)
        if options.dilation_pixels:
            try:
                from scipy.ndimage import binary_dilation
            except ImportError as exc:  # pragma: no cover - optional environment dependency
                raise RuntimeError("background mask dilation requires scipy") from exc
            mask = binary_dilation(mask, iterations=options.dilation_pixels)
        fraction = float(mask.mean())
        if fraction > options.max_mask_fraction:
            raise RuntimeError(f"frame index {frame['index']} background mask {fraction:.2%} exceeds safety limit {options.max_mask_fraction:.2%}")
        name = f"{ordinal:04d}.png"
        _atomic_png(rgb, output / "frames" / name, "RGB")
        _atomic_png((mask.astype(np.uint8) * 255), output / "masks" / name, "L")
        preview = rgb.astype(np.float32)
        preview[mask] = preview[mask] * .30 + np.asarray((52., 112., 235.), np.float32) * .70
        _atomic_png(np.rint(preview).astype(np.uint8), output / "previews" / name, "RGB")
        records.append({"ordinal": ordinal, "input_frame_index": int(frame["index"]), "input_frame_id": str(frame["frame_id"]), "timestamp_us": int(frame["timestamp_us"]), "frame": f"frames/{name}", "mask": f"masks/{name}", "preview": f"previews/{name}", "mask_mode": "alpha_and_sky_invalid_background_only", "mask_fraction": fraction})
    manifest = {"schema": SCHEMA, "version": VERSION, "status": "complete", "kind": "background_unknown_inpainting_input", "render_dir": str(render_root), "frame_start": options.frame_start, "frame_end": options.frame_end, "alpha_threshold": options.alpha_threshold, "sky_valid_threshold": options.sky_valid_threshold, "dilation_pixels": options.dilation_pixels, "max_mask_fraction": options.max_mask_fraction, "frames": records, "limitations": ["Only alpha-low and sky-invalid pixels are eligible; high-alpha static geometry, dynamic actors, and all normal sky pixels are excluded.", "This is a display-only RGB fill for unobserved background holes, not a correction of 3DGS geometry or a sensor image."]}
    _atomic_json(manifest, output / "manifest.json")
    print(f"Prepared ProPainter background-only input: {output} ({len(records)} frame(s), mean mask={np.mean([item['mask_fraction'] for item in records]):.2%})", flush=True)
    return output


def audit_propainter_display_demo(prepared: Path, result_frames: Path, output: Path) -> dict[str, Any]:
    """Verify known pixels are untouched and quantify the display-only edit area."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest = _read_json(prepared.resolve() / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("prepared input is not a completed ProPainter display manifest")
    result_root = result_frames.resolve()
    records: list[dict[str, Any]] = []
    outside_mae: list[float] = []
    inside_change: list[float] = []
    for item in manifest["frames"]:
        ordinal = int(item["ordinal"])
        source = np.asarray(Image.open(prepared / str(item["frame"])).convert("RGB"), np.float32) / 255.0
        mask = np.asarray(Image.open(prepared / str(item["mask"])).convert("L"), np.uint8) > 0
        result_path = result_root / f"{ordinal:04d}.png"
        result = np.asarray(Image.open(result_path).convert("RGB"), np.float32) / 255.0
        if result.shape != source.shape:
            raise ValueError(f"ProPainter result shape differs for ordinal {ordinal}: {result.shape} vs {source.shape}")
        delta = np.abs(result - source).mean(axis=-1)
        outside_value = float(delta[~mask].mean()) if (~mask).any() else 0.0
        inside_value = float(delta[mask].mean()) if mask.any() else 0.0
        outside_mae.append(outside_value)
        inside_change.append(inside_value)
        records.append({"ordinal": ordinal, "input_frame_index": int(item["input_frame_index"]), "mask_fraction": float(mask.mean()), "outside_mask_rgb_mae": outside_value, "inside_mask_rgb_change": inside_value})
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "kind": "display_only_proPainter_audit", "prepared": str(prepared.resolve()), "result_frames": str(result_root), "frames": records, "summary": {"frame_count": len(records), "mean_outside_mask_rgb_mae": float(np.mean(outside_mae)), "mean_inside_mask_rgb_change": float(np.mean(inside_change)), "outside_mask_unchanged": bool(max(outside_mae, default=0.0) <= 1.0 / 255.0)}, "limitations": ["This verifies only display-space preservation outside the removal mask; it cannot prove background correctness behind a removed dynamic object.", "A passing result is not dynamic-object reconstruction and must not be treated as sensor ground truth."]}
    _atomic_json(report, output)
    print(f"ProPainter display audit frames={len(records)} outside_mae={report['summary']['mean_outside_mask_rgb_mae']:.6f} inside_change={report['summary']['mean_inside_mask_rgb_change']:.6f}", flush=True)
    return report
