"""Source-image guided static retexturing for target-view *display* renders.

This is deliberately not another 3D reconstruction or dynamic-actor model.
It keeps the target render's PLY-derived alpha/depth as the geometry authority,
unprojects only high-confidence target pinhole surface pixels, then uses NCore's
native FTheta rolling-shutter projection to borrow a conservative colour/detail
sample from the simultaneous source cameras.  It can therefore expose detail
lost when multi-view Gaussian DC colours were averaged, while never inventing
new depth, alpha, or dynamic-object geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .camera import invert_pose
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path


SCHEMA = "ncore-source-guided-static-retexture"
VERSION = 1
DEFAULT_SOURCE_CAMERAS = (
    "camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov",
    "camera_rear_left_70fov", "camera_rear_right_70fov",
)


@dataclass(frozen=True)
class SourceRetextureOptions:
    static_render_root: Path
    ncore_path: Path
    output_root: Path
    source_camera_ids: tuple[str, ...] = DEFAULT_SOURCE_CAMERAS
    dynamic_mask_manifest: Path | None = None
    device: str = "cuda"
    minimum_alpha: float = .90
    maximum_depth_m: float = 150.0
    minimum_view_cosine: float = .80
    color_blend: float = .35
    detail_strength: float = .65
    detail_blur_radius: float = 1.2
    frame_start: int | None = None
    frame_end: int | None = None

    def __post_init__(self) -> None:
        if not self.source_camera_ids or len(set(self.source_camera_ids)) != len(self.source_camera_ids):
            raise ValueError("source_camera_ids must be non-empty and unique")
        if not 0.0 <= self.minimum_alpha <= 1.0 or self.maximum_depth_m <= .1:
            raise ValueError("alpha/depth thresholds are invalid")
        if not -1.0 <= self.minimum_view_cosine <= 1.0:
            raise ValueError("minimum_view_cosine must be in [-1, 1]")
        if not 0.0 <= self.color_blend <= 1.0 or self.detail_strength < 0.0 or self.detail_blur_radius < 0.0:
            raise ValueError("retexture blending parameters are invalid")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_end < 0:
            raise ValueError("frame_end must be non-negative")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")


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


def _render_units(root: Path) -> dict[str, Path]:
    if (root / "manifest.json").is_file():
        return {"": root}
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = {item.name: item for item in root.iterdir() if item.is_dir() and (item / "manifest.json").is_file()}
    if not result:
        raise ValueError(f"no render manifests below {root}")
    return result


def target_depth_to_world(depth_z: np.ndarray, world_to_camera: np.ndarray, intrinsics: dict[str, Any]) -> np.ndarray:
    """Unproject target pinhole camera-z depth into world coordinates."""
    depth = np.asarray(depth_z, np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_z must have shape [height,width]")
    height, width = depth.shape
    if int(intrinsics["width"]) != width or int(intrinsics["height"]) != height:
        raise ValueError("target depth and intrinsics resolution disagree")
    yy, xx = np.indices((height, width), dtype=np.float32)
    x = ((xx + .5 - float(intrinsics["cx"])) / float(intrinsics["fx"])) * depth
    y = ((yy + .5 - float(intrinsics["cy"])) / float(intrinsics["fy"])) * depth
    camera = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
    c2w = invert_pose(np.asarray(world_to_camera, np.float64))
    return (camera @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)


def bilinear_rgb(image: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample raw RGB at floating pixel coordinates without extrapolation."""
    value = np.asarray(image, np.float32)
    xy = np.asarray(pixels, np.float32)
    if value.ndim != 3 or value.shape[2] != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("expected RGB [H,W,3] and pixels [N,2]")
    height, width = value.shape[:2]
    valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0.0) & (xy[:, 0] <= width - 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= height - 1.0)
    result = np.zeros((len(xy), 3), np.float32)
    if not valid.any():
        return result, valid
    x, y = xy[valid, 0], xy[valid, 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    tx, ty = (x - x0)[:, None], (y - y0)[:, None]
    result[valid] = ((1 - tx) * (1 - ty) * value[y0, x0] + tx * (1 - ty) * value[y0, x1]
                     + (1 - tx) * ty * value[y1, x0] + tx * ty * value[y1, x1])
    return result, valid


def source_detail(image: np.ndarray, radius: float) -> np.ndarray:
    raw = np.asarray(image, np.float32)
    if radius == 0.0:
        return np.zeros_like(raw)
    blurred = np.asarray(Image.fromarray(np.asarray(image, np.uint8), "RGB").filter(ImageFilter.GaussianBlur(radius)), np.float32)
    return raw - blurred


def compose_source_retexture(base_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_detail: np.ndarray,
                             candidate_score: np.ndarray, candidate_flat_index: np.ndarray, *, color_blend: float,
                             detail_strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the best view-consistent source sample per target pixel."""
    base = np.asarray(base_rgb, np.uint8)
    rgb, detail = np.asarray(candidate_rgb, np.float32), np.asarray(candidate_detail, np.float32)
    score, flat = np.asarray(candidate_score, np.float32), np.asarray(candidate_flat_index, np.int64)
    if base.ndim != 3 or base.shape[2] != 3 or rgb.shape != detail.shape or rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("invalid retexture tensor shapes")
    if len(rgb) != len(score) or len(rgb) != len(flat):
        raise ValueError("candidate arrays must have equal length")
    result = base.astype(np.float32)
    pixel_count = int(np.prod(base.shape[:2]))
    best_score = np.full(pixel_count, -np.inf, np.float32)
    owner = np.full(pixel_count, -1, np.int32)
    for i in np.argsort(score, kind="stable").tolist():
        index = int(flat[i])
        if 0 <= index < len(best_score) and score[i] > best_score[index]:
            best_score[index] = score[i]
            owner[index] = i
    chosen = np.flatnonzero(owner >= 0)
    if len(chosen):
        source = owner[chosen]
        result.reshape(-1, 3)[chosen] = np.clip(
            (1.0 - color_blend) * result.reshape(-1, 3)[chosen] + color_blend * rgb[source] + detail_strength * detail[source], 0.0, 255.0,
        )
    coverage = (owner >= 0).reshape(base.shape[:2])
    return np.rint(result).astype(np.uint8), coverage, owner.reshape(base.shape[:2])


def _source_masks(path: Path | None, camera_ids: tuple[str, ...]) -> dict[tuple[str, int], list[tuple[int, Path]]]:
    if path is None:
        return {}
    manifest = _read_json(path)
    root = path.parent
    result: dict[tuple[str, int], list[tuple[int, Path]]] = {}
    for camera in manifest.get("cameras", []):
        if not isinstance(camera, dict) or str(camera.get("camera_id")) not in camera_ids:
            continue
        for observation in camera.get("observations", []):
            if not isinstance(observation, dict) or observation.get("mask_origin") != "source":
                continue
            relative = observation.get("refined_mask")
            if isinstance(relative, str):
                candidate = root / relative
                if candidate.is_file():
                    result.setdefault((str(camera["camera_id"]), int(observation["reference_frame_index"])), []).append((int(observation["source_frame_index"]), candidate))
    return result


def _camera_source(loader: Any, camera_id: str, timestamp_us: int) -> tuple[Any, int, np.ndarray]:
    sensor = loader.get_camera_sensor(camera_id)
    index = int(sensor.get_closest_frame_index(timestamp_us))
    camera = _load_ncore_source_camera_path({"frame_indices": [index]}, loader, camera_id).frames[0]
    return camera, index, np.asarray(sensor.get_frame_image_array(index), np.uint8)


def _native_project(model: Any, points_world: np.ndarray, camera: Any) -> tuple[np.ndarray, np.ndarray]:
    """Exact NCore FTheta plus rolling-shutter forward projection."""
    projected = model.world_points_to_pixels_shutter_pose(
        np.asarray(points_world, np.float32), np.asarray(camera.world_to_camera, np.float64),
        np.asarray(camera.world_to_camera_end, np.float64), int(camera.timestamp_start_us), int(camera.timestamp_us),
        return_T_world_sensors=True, return_valid_indices=True,
    )
    indices = projected.valid_indices.detach().cpu().numpy().astype(np.int64)
    pixels = projected.pixels.detach().float().cpu().numpy().astype(np.float32)
    return indices, pixels


def _unit(options: SourceRetextureOptions, input_dir: Path, output_dir: Path, loader: Any, models: dict[str, Any],
          masks: dict[tuple[str, int], list[tuple[int, Path]]]) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete render manifest: {input_dir}")
    records: list[dict[str, Any]] = []
    for frame in manifest.get("frames", []):
        index = int(frame["index"])
        outputs = frame.get("outputs", {})
        if not isinstance(outputs, dict) or not {"rgb", "alpha", "depth_npy"}.issubset(outputs):
            raise ValueError("source retexture requires RGB, alpha and depth_npy")
        rgb = np.asarray(Image.open(input_dir / str(outputs["rgb"])).convert("RGB"), np.uint8)
        alpha = np.asarray(Image.open(input_dir / str(outputs["alpha"])).convert("L"), np.uint8).astype(np.float32) / 255.0
        depth = np.asarray(np.load(input_dir / str(outputs["depth_npy"]), allow_pickle=False), np.float32)
        process = ((options.frame_start is None or index >= options.frame_start)
                   and (options.frame_end is None or index <= options.frame_end))
        eligible = process & (alpha.reshape(-1) >= options.minimum_alpha) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > .1) & (depth.reshape(-1) <= options.maximum_depth_m)
        points = target_depth_to_world(depth, np.asarray(frame["world_to_camera_start"], np.float64), dict(frame["intrinsics"]))
        target_origin = -(np.asarray(frame["world_to_camera_start"], np.float32)[:3, :3].T @ np.asarray(frame["world_to_camera_start"], np.float32)[:3, 3])
        target_dirs = points - target_origin[None]
        target_dirs /= np.maximum(np.linalg.norm(target_dirs, axis=1, keepdims=True), 1e-8)
        candidate_rgb: list[np.ndarray] = []; candidate_detail: list[np.ndarray] = []; candidate_score: list[np.ndarray] = []; candidate_flat: list[np.ndarray] = []
        source_meta: list[dict[str, Any]] = []
        for camera_id in options.source_camera_ids:
            camera, source_index, image = _camera_source(loader, camera_id, int(frame["timestamp_us"]))
            source_indices, pixels = _native_project(models[camera_id], points[eligible], camera)
            if not len(source_indices):
                source_meta.append({"camera_id": camera_id, "source_frame_index": source_index, "accepted": 0})
                continue
            flat_candidates = np.flatnonzero(eligible)[source_indices]
            colors, inside = bilinear_rgb(image, pixels)
            source_origin = -(np.asarray(camera.world_to_camera, np.float32)[:3, :3].T @ np.asarray(camera.world_to_camera, np.float32)[:3, 3])
            source_dirs = points[flat_candidates] - source_origin[None]
            source_dirs /= np.maximum(np.linalg.norm(source_dirs, axis=1, keepdims=True), 1e-8)
            score = np.sum(target_dirs[flat_candidates] * source_dirs, axis=1).astype(np.float32)
            accepted = inside & (score >= options.minimum_view_cosine)
            mask_records = masks.get((camera_id, index), [])
            # A source mask is tied to an exact sensor frame.  If the closest
            # timestamp differs, rejecting this view is safer than applying a
            # mask at the wrong rolling-shutter exposure and leaking an actor.
            if mask_records and any(mask_source_index != source_index for mask_source_index, _ in mask_records):
                accepted[:] = False
            for _, mask_path in mask_records:
                mask = np.asarray(Image.open(mask_path).convert("L"), np.uint8) > 0
                if mask.shape != image.shape[:2]:
                    raise ValueError(f"dynamic mask/image mismatch: {mask_path}")
                xx = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
                yy = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
                accepted &= ~mask[yy, xx]
            if accepted.any():
                detail, _ = bilinear_rgb(source_detail(image, options.detail_blur_radius), pixels[accepted])
                candidate_rgb.append(colors[accepted]); candidate_detail.append(detail); candidate_score.append(score[accepted]); candidate_flat.append(flat_candidates[accepted])
            source_meta.append({"camera_id": camera_id, "source_frame_index": source_index, "projected": int(len(source_indices)), "accepted": int(accepted.sum())})
        if candidate_rgb:
            refined, coverage, owner = compose_source_retexture(
                rgb, np.concatenate(candidate_rgb), np.concatenate(candidate_detail), np.concatenate(candidate_score), np.concatenate(candidate_flat),
                color_blend=options.color_blend, detail_strength=options.detail_strength,
            )
        else:
            refined, coverage, owner = rgb, np.zeros(depth.shape, bool), np.full(depth.shape, -1, np.int32)
        for key, relative in outputs.items():
            destination = output_dir / str(relative)
            if key == "rgb":
                _atomic_png(refined, destination, "RGB")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(input_dir / str(relative), destination)
        stem = Path(str(outputs["rgb"])).stem
        _atomic_png((coverage.astype(np.uint8) * 255), output_dir / "source_retexture_coverage" / f"{stem}.png", "L")
        copied = dict(frame)
        copied["outputs"] = dict(outputs) | {"source_retexture_coverage": f"source_retexture_coverage/{stem}.png"}
        copied["source_retexture"] = {"processed": bool(process), "eligible_fraction": float(eligible.mean()), "coverage_fraction": float(coverage.mean()), "source_views": source_meta}
        records.append(copied)
        if process:
            print(f"Source retexture [{index:04d}] coverage={coverage.mean():.3%} eligible={eligible.mean():.3%}", flush=True)
    result = dict(manifest)
    result.update({"status": "complete", "frames": records, "frame_count": len(records), "completed_frame_count": len(records), "source_retexture": {
        "schema": SCHEMA, "version": VERSION, "source_cameras": list(options.source_camera_ids), "minimum_alpha": options.minimum_alpha,
        "maximum_depth_m": options.maximum_depth_m, "minimum_view_cosine": options.minimum_view_cosine, "color_blend": options.color_blend,
        "detail_strength": options.detail_strength, "detail_blur_radius": options.detail_blur_radius,
        "usage_warning": "RGB-only source-image retexture. Alpha/depth are copied unchanged; do not use as sensor ground truth or reconstruction supervision.",
    }})
    _atomic_json(result, output_dir / "manifest.json")
    return result["source_retexture"]


def retexture_static_render(options: SourceRetextureOptions) -> dict[str, Any]:
    from ncore.impl.sensors.camera import FThetaCameraModel

    if options.output_root.exists() and any(options.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {options.output_root}")
    loader = _create_ncore_loader(options.ncore_path)
    models = {camera_id: FThetaCameraModel(loader.get_camera_sensor(camera_id).model_parameters, device=options.device) for camera_id in options.source_camera_ids}
    masks = _source_masks(options.dynamic_mask_manifest, options.source_camera_ids)
    units = _render_units(options.static_render_root)
    options.output_root.mkdir(parents=True, exist_ok=True)
    multi = len(units) > 1
    reports = {name or "camera": _unit(options, source, options.output_root / name if multi else options.output_root, loader, models, masks) for name, source in sorted(units.items())}
    report = {"schema": SCHEMA, "version": VERSION, "static_render_root": str(options.static_render_root), "output_root": str(options.output_root), "units": reports,
              "usage_warning": "Source-guided RGB display derivative only; source dynamic masks are rejected and geometry outputs remain untouched."}
    _atomic_json(report, options.output_root / "source_retexture_report.json")
    return report
