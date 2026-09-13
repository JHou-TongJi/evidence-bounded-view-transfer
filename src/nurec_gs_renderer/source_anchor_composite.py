"""Source-image anchored dynamic RGB overlays for target-camera demonstrations.

Unlike a dynamic Gaussian layer, this module never invents an unobserved actor
surface.  It reprojects only original NCore RGB samples that are simultaneously
inside a conservative dynamic instance mask and supported by raw LiDAR depth.
The result is therefore a sparse, display-only dynamic overlay; the supplied
static render remains the geometry/depth authority.
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
from PIL import Image

from .pose_sources import _create_ncore_loader


SCHEMA = "nurec-gs-source-anchor-composite"
VERSION = 1


@dataclass(frozen=True)
class SourceAnchorCompositeOptions:
    static_render_root: Path
    dynamic_dataset_manifest: Path
    ncore_path: Path
    output_root: Path
    device: str = "cuda"
    maximum_timestamp_delta_us: int = 40_000
    static_occlusion_tolerance_m: float = 0.75
    splat_radius_pixels: int = 2
    min_lidar_samples_per_instance: int = 12
    maximum_depth_propagation_pixels: float = 0.0
    max_samples_per_instance: int = 30_000

    def __post_init__(self) -> None:
        if self.maximum_timestamp_delta_us < 0:
            raise ValueError("maximum_timestamp_delta_us must be non-negative")
        if self.static_occlusion_tolerance_m < 0.0:
            raise ValueError("static_occlusion_tolerance_m must be non-negative")
        if self.splat_radius_pixels < 0:
            raise ValueError("splat_radius_pixels must be non-negative")
        if self.min_lidar_samples_per_instance <= 0:
            raise ValueError("min_lidar_samples_per_instance must be positive")
        if self.maximum_depth_propagation_pixels < 0.0:
            raise ValueError("maximum_depth_propagation_pixels must be non-negative")
        if self.max_samples_per_instance <= 0:
            raise ValueError("max_samples_per_instance must be positive")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _render_units(root: Path) -> dict[str, Path]:
    if (root / "manifest.json").is_file():
        return {"": root}
    if not root.is_dir():
        raise NotADirectoryError(root)
    units = {item.name: item for item in root.iterdir() if item.is_dir() and (item / "manifest.json").is_file()}
    if not units:
        raise ValueError(f"no completed render manifests found below {root}")
    return units


def _atomic_png(image: np.ndarray, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(image, mode=mode).save(temporary)
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


def project_pinhole(points_world: np.ndarray, world_to_camera: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into an OpenCV pinhole target image at pixel centres."""
    points = np.asarray(points_world, dtype=np.float32)
    w2c = np.asarray(world_to_camera, dtype=np.float32)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[:, 2].astype(np.float32)
    valid = np.isfinite(camera).all(axis=1) & (depth > 0.1)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float32)
    if valid.any():
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
        pixels[valid, 0] = fx * camera[valid, 0] / depth[valid] + cx - 0.5
        pixels[valid, 1] = fy * camera[valid, 1] / depth[valid] + cy - 0.5
    width, height = int(intrinsics["width"]), int(intrinsics["height"])
    valid &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] <= width - 1.0) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= height - 1.0)
    return pixels, depth, valid


def _target_origin(world_to_camera: np.ndarray) -> np.ndarray:
    w2c = np.asarray(world_to_camera, dtype=np.float32)
    return -(w2c[:3, :3].T @ w2c[:3, 3]).astype(np.float32)


def splat_anchored_samples(
    base_rgb: np.ndarray,
    static_depth: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    alignment: np.ndarray,
    *,
    static_occlusion_tolerance_m: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-order sparse source samples over a static RGB image.

    ``coverage`` denotes only source-supported pixels, not a reconstructed
    actor alpha.  Equal-depth samples prefer the source view closest to the
    target view direction.
    """
    result = np.asarray(base_rgb, dtype=np.uint8).copy()
    depth_image = np.asarray(static_depth, dtype=np.float32)
    height, width = result.shape[:2]
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    view_score = np.full((height, width), -np.inf, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    owner = np.full((height, width), -1, dtype=np.int32)
    order = np.argsort(np.asarray(depths, dtype=np.float32), kind="stable")
    for index in order.tolist():
        x = int(np.rint(pixels[index, 0]))
        y = int(np.rint(pixels[index, 1]))
        z = float(depths[index])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                xx, yy = x + dx, y + dy
                if xx < 0 or xx >= width or yy < 0 or yy >= height:
                    continue
                reference_depth = depth_image[yy, xx]
                if np.isfinite(reference_depth) and z > float(reference_depth) + static_occlusion_tolerance_m:
                    continue
                closer = z < zbuffer[yy, xx] - 1e-3
                tie_better_view = abs(z - zbuffer[yy, xx]) <= 1e-3 and alignment[index] > view_score[yy, xx]
                if closer or tie_better_view:
                    result[yy, xx] = colors[index]
                    zbuffer[yy, xx] = z
                    view_score[yy, xx] = alignment[index]
                    coverage[yy, xx] = True
                    owner[yy, xx] = index
    return result, coverage, owner


def _nearest_dynamic_frame(frames: list[dict[str, Any]], timestamp_us: int, maximum_delta_us: int) -> dict[str, Any] | None:
    if not frames:
        return None
    candidate = min(frames, key=lambda item: abs(int(item["reference_timestamp_end_us"]) - timestamp_us))
    return candidate if abs(int(candidate["reference_timestamp_end_us"]) - timestamp_us) <= maximum_delta_us else None


def _instance_depth_samples(
    dataset_root: Path,
    instance: dict[str, Any],
    xy: np.ndarray,
    depth_z: np.ndarray,
    image_shape: tuple[int, int],
    options: SourceAnchorCompositeOptions,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Optionally propagate only to nearby pixels inside the original mask.

    This is a 2.5D *demonstration* approximation.  A propagation radius of
    zero is the default and returns raw LiDAR samples exactly.  It is never a
    substitute for a learned surface or for dynamic training supervision.
    """
    if options.maximum_depth_propagation_pixels == 0.0:
        return xy, depth_z, "raw_lidar"
    mask_relative = instance.get("mask")
    if not isinstance(mask_relative, str):
        return xy, depth_z, "raw_lidar_mask_missing"
    mask = np.asarray(Image.open(dataset_root / mask_relative).convert("L"), dtype=np.uint8) > 0
    if mask.shape != image_shape:
        raise ValueError("dynamic instance mask/image shape mismatch")
    height, width = mask.shape
    seeds = np.zeros(mask.shape, dtype=bool)
    xx = np.rint(xy[:, 0]).astype(np.int64)
    yy = np.rint(xy[:, 1]).astype(np.int64)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height) & np.isfinite(depth_z) & (depth_z > 0.1)
    seeds[yy[valid], xx[valid]] = True
    if not seeds.any():
        return xy, depth_z, "raw_lidar_no_seed"
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("dense source-anchor propagation requires scipy") from exc
    distance, nearest = distance_transform_edt(~seeds, return_indices=True)
    seed_depth = np.full(mask.shape, np.nan, dtype=np.float32)
    # The sparse source depth was already nearest-z-buffered by dataset construction.
    seed_depth[yy[valid], xx[valid]] = depth_z[valid]
    accepted = mask & (distance <= options.maximum_depth_propagation_pixels)
    sample_y, sample_x = np.nonzero(accepted)
    propagated_depth = seed_depth[nearest[0, sample_y, sample_x], nearest[1, sample_y, sample_x]]
    finite = np.isfinite(propagated_depth) & (propagated_depth > 0.1)
    sample_xy = np.stack((sample_x[finite], sample_y[finite]), axis=1).astype(np.float32)
    sample_depth = propagated_depth[finite].astype(np.float32)
    if len(sample_xy) > options.max_samples_per_instance:
        selected = np.linspace(0, len(sample_xy) - 1, options.max_samples_per_instance, dtype=np.int64)
        sample_xy, sample_depth = sample_xy[selected], sample_depth[selected]
    return sample_xy, sample_depth, "mask_nearest_lidar_depth"


def _source_points(
    loader: Any,
    models: dict[str, Any],
    dataset_root: Path,
    dynamic_frame: dict[str, Any],
    target_origin: np.ndarray,
    options: SourceAnchorCompositeOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    world_points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    alignment: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for record in dynamic_frame.get("cameras", []):
        camera_id = str(record["camera_id"])
        sensor = loader.get_camera_sensor(camera_id)
        image = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), dtype=np.uint8)
        for instance in record.get("instances", []):
            relative = instance.get("lidar_depth")
            if not isinstance(relative, str):
                continue
            with np.load(dataset_root / relative, allow_pickle=False) as archive:
                pixels = np.asarray(archive["xy"], dtype=np.float32)
                depth_z = np.asarray(archive["depth_m"], dtype=np.float32)
            if len(pixels) < options.min_lidar_samples_per_instance:
                continue
            pixels, depth_z, depth_mode = _instance_depth_samples(
                dataset_root, instance, pixels, depth_z, image.shape[:2], options,
            )
            returned = models[camera_id].image_points_to_world_rays_shutter_pose(
                pixels,
                np.asarray(record["camera_to_world_start"], dtype=np.float64),
                np.asarray(record["camera_to_world_end"], dtype=np.float64),
                int(record["timestamp_start_us"]), int(record["timestamp_end_us"]),
            )
            rays = returned.world_rays.detach().float().cpu().numpy()
            origins = rays[:, :3].astype(np.float32)
            directions = rays[:, 3:].astype(np.float32)
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
            c2w = np.asarray(record["camera_to_world_start"], dtype=np.float32)
            forward = c2w[:3, 2]
            ray_forward = np.maximum(directions @ forward, 1e-4)
            points = origins + directions * (depth_z / ray_forward)[:, None]
            xx = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
            yy = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
            source_view = points - origins
            source_view /= np.maximum(np.linalg.norm(source_view, axis=1, keepdims=True), 1e-8)
            target_view = points - target_origin[None]
            target_view /= np.maximum(np.linalg.norm(target_view, axis=1, keepdims=True), 1e-8)
            world_points.append(points)
            colors.append(image[yy, xx])
            alignment.append(np.sum(source_view * target_view, axis=1).astype(np.float32))
            sources.append({
                "camera_id": camera_id,
                "track_id": int(instance["track_id"]),
                "actor_family": str(instance["actor_family"]),
                "lidar_samples": int(len(points)),
                "depth_mode": depth_mode,
            })
    if not world_points:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty((0,), np.float32), sources)
    return (np.concatenate(world_points), np.concatenate(colors), np.concatenate(alignment), sources)


def _composite_unit(
    input_dir: Path,
    output_dir: Path,
    loader: Any,
    models: dict[str, Any],
    dataset_root: Path,
    dynamic_frames: list[dict[str, Any]],
    options: SourceAnchorCompositeOptions,
) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"render manifest is not complete: {input_dir}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"render manifest has no frames: {input_dir}")
    records: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("invalid render frame record")
        outputs = frame.get("outputs")
        if not isinstance(outputs, dict) or not {"rgb", "alpha", "depth_npy"}.issubset(outputs):
            raise ValueError("source-anchor compositing requires rgb, alpha and depth_npy")
        rgb_path = input_dir / str(outputs["rgb"])
        depth_path = input_dir / str(outputs["depth_npy"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
        dynamic = _nearest_dynamic_frame(dynamic_frames, int(frame["timestamp_us"]), options.maximum_timestamp_delta_us)
        if dynamic is None:
            refined, coverage = rgb, np.zeros(depth.shape, dtype=bool)
            source_summary: list[dict[str, Any]] = []
            point_count = 0
        else:
            w2c = np.asarray(frame["world_to_camera_start"], dtype=np.float32)
            points, colors, alignment, source_summary = _source_points(
                loader, models, dataset_root, dynamic, _target_origin(w2c), options,
            )
            pixels, point_depth, valid = project_pinhole(points, w2c, dict(frame["intrinsics"]))
            refined, coverage, _ = splat_anchored_samples(
                rgb, depth, pixels[valid], point_depth[valid], colors[valid], alignment[valid],
                static_occlusion_tolerance_m=options.static_occlusion_tolerance_m,
                radius=options.splat_radius_pixels,
            )
            point_count = int(valid.sum())
        for key, relative in outputs.items():
            destination = output_dir / str(relative)
            if key == "rgb":
                _atomic_png(refined, destination, "RGB")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_dir / str(relative), destination)
        stem = Path(str(outputs["rgb"])).stem
        _atomic_png((coverage.astype(np.uint8) * 255), output_dir / "source_anchor_coverage" / f"{stem}.png", "L")
        copied = dict(frame)
        copied["outputs"] = dict(outputs) | {"source_anchor_coverage": f"source_anchor_coverage/{stem}.png"}
        copied["source_anchor_composite"] = {
            "reference_frame_index": None if dynamic is None else int(dynamic["reference_frame_index"]),
            "source_projected_points": point_count,
            "coverage_fraction": float(coverage.mean()),
            "sources": source_summary,
        }
        records.append(copied)
    result = dict(manifest)
    result.update({
        "status": "complete", "frames": records, "frame_count": len(records), "completed_frame_count": len(records),
        "source_anchor_composite": {
            "schema": SCHEMA, "version": VERSION,
            "dynamic_dataset_manifest": str(options.dynamic_dataset_manifest.resolve()),
            "maximum_timestamp_delta_us": options.maximum_timestamp_delta_us,
            "static_occlusion_tolerance_m": options.static_occlusion_tolerance_m,
            "splat_radius_pixels": options.splat_radius_pixels,
            "maximum_depth_propagation_pixels": options.maximum_depth_propagation_pixels,
            "max_samples_per_instance": options.max_samples_per_instance,
            "usage_warning": "RGB-only sparse source/LiDAR overlay. Alpha and expected depth remain static-render outputs; not sensor ground truth.",
        },
    })
    _atomic_json(result, output_dir / "manifest.json")
    return result["source_anchor_composite"]


def composite_source_anchored_dynamic(options: SourceAnchorCompositeOptions) -> dict[str, Any]:
    """Create a target-view hybrid demo without a learned dynamic actor asset."""
    from ncore.impl.sensors.camera import FThetaCameraModel

    output = options.output_root
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output root is not empty: {output}")
    dataset = _read_json(options.dynamic_dataset_manifest)
    if dataset.get("status") != "complete":
        raise ValueError("dynamic dataset manifest is not complete")
    if Path(str(dataset.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic dataset ncore_path does not match --ncore-path")
    frames = [item for item in dataset.get("frames", []) if isinstance(item, dict)]
    loader = _create_ncore_loader(options.ncore_path)
    camera_ids = sorted({str(record["camera_id"]) for frame in frames for record in frame.get("cameras", [])})
    models = {camera_id: FThetaCameraModel(loader.get_camera_sensor(camera_id).model_parameters, device=options.device) for camera_id in camera_ids}
    units = _render_units(options.static_render_root)
    output.mkdir(parents=True, exist_ok=True)
    multiple = len(units) > 1
    reports: dict[str, Any] = {}
    dataset_root = options.dynamic_dataset_manifest.parent
    for name in sorted(units):
        reports[name or "camera"] = _composite_unit(
            units[name], output / name if multiple else output, loader, models, dataset_root, frames, options,
        )
    report = {"schema": SCHEMA, "version": VERSION, "static_render_root": str(options.static_render_root),
              "output_root": str(output), "units": reports,
              "usage_warning": "Sparse RGB-only hybrid demo; no learned actor geometry was produced."}
    _atomic_json(report, output / "source_anchor_composite_report.json")
    return report
