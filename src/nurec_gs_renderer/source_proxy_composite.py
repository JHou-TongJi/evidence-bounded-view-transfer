"""Single-source 2.5-D dynamic actor proxy for *display-only* target renders.

This is deliberately a weaker fallback than object-level 4-D reconstruction.
It turns one original NCore instance mask into a source-facing, actor-centred
texture plane and reprojects that plane into a nearby target pinhole camera.
The target image therefore receives dense original pixels rather than a sparse
and incomplete Gaussian cloud.  It never changes PLY assets, renderer RGB,
alpha, or expected depth; its output is separately named ``rgb_display``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader
from .source_anchor_composite import _render_units, _target_origin, project_pinhole
from .street_dataset import _interpolate_track_pose


SCHEMA = "nurec-gs-source-proxy-composite"
VERSION = 2


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
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


def _atomic_npy(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, value, allow_pickle=False)
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


def _normalise(vectors: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def source_facing_plane_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    plane_center: np.ndarray,
    source_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect source rays with an actor-centred plane facing its source view.

    The resulting plane preserves the observed source silhouette and scale.  It
    intentionally has no guessed back/side surface; target views outside the
    configured source-view angle are rejected by the caller.
    """
    ray_origins = np.asarray(origins, dtype=np.float32)
    ray_directions = _normalise(directions)
    centre = np.asarray(plane_center, dtype=np.float32).reshape(3)
    normal = _normalise(np.asarray(source_origin, dtype=np.float32).reshape(3) - centre)
    numerator = float(np.dot(centre - ray_origins[0], normal))
    denominator = ray_directions @ normal
    t = numerator / np.where(np.abs(denominator) > 1e-6, denominator, np.nan)
    valid = np.isfinite(t) & (t > .1) & (np.abs(denominator) > .10)
    points = ray_origins + ray_directions * np.where(valid, t, 0.0)[:, None]
    return points.astype(np.float32), valid


def cuboid_face_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    actor_to_world: np.ndarray,
    length_width_height: np.ndarray,
    *,
    margin_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect world rays with the observed entrance face of one tracked cuboid.

    This is deliberately a conservative 2.5-D upgrade of the old source plane:
    every copied source pixel receives the depth of the *measured track cuboid*
    face it hits.  Rays that miss the cuboid remain transparent rather than
    inventing a back/side surface.  ``face_axis`` is 0/1/2 for local
    length/width/height and -1 for a miss; it is written only as a diagnostic.
    """
    ray_origins = np.asarray(origins, dtype=np.float32)
    ray_directions = _normalise(directions)
    pose = np.asarray(actor_to_world, dtype=np.float32)
    dimensions = np.asarray(length_width_height, dtype=np.float32).reshape(3)
    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3 or ray_directions.shape != ray_origins.shape:
        raise ValueError("origins/directions must be [N,3]")
    if pose.shape != (4, 4) or np.any(dimensions <= 0.0) or margin_m < 0.0:
        raise ValueError("invalid cuboid pose, dimensions, or margin")
    # For row vectors, p_local = (p_world - t) @ R where R contains the local
    # axes in world coordinates.  This agrees with the actor_to_world fields
    # supplied by the NCore track parser.
    rotation = pose[:3, :3]
    local_origins = (ray_origins - pose[:3, 3]) @ rotation
    local_directions = ray_directions @ rotation
    half = dimensions * .5 + float(margin_m)
    parallel = np.abs(local_directions) <= 1e-7
    outside_parallel = parallel & (np.abs(local_origins) > half[None, :])
    inverse = np.where(parallel, 1.0, 1.0 / np.where(parallel, 1.0, local_directions))
    lower = (-half[None, :] - local_origins) * inverse
    upper = (half[None, :] - local_origins) * inverse
    near = np.minimum(lower, upper)
    far = np.maximum(lower, upper)
    near = np.where(parallel, -np.inf, near)
    far = np.where(parallel, np.inf, far)
    entry_axis = np.argmax(near, axis=1)
    entry = np.max(near, axis=1)
    exit_ = np.min(far, axis=1)
    # Cameras should be outside an actor box.  The fallback to exit makes the
    # routine well-defined for a rare rig/box overlap without extrapolation.
    distance = np.where(entry > .1, entry, exit_)
    valid = (~outside_parallel.any(axis=1)) & np.isfinite(distance) & (distance > .1) & (exit_ > np.maximum(entry, .1) + 1e-5)
    points = ray_origins + ray_directions * np.where(valid, distance, 0.0)[:, None]
    face_axis = np.where(valid, entry_axis, -1).astype(np.int8)
    return points.astype(np.float32), valid, face_axis


def view_angle_degrees(actor_center: np.ndarray, source_origin: np.ndarray, target_origin: np.ndarray) -> float:
    """Return the source/target observation angle at the actor centre."""
    source = _normalise(np.asarray(source_origin, np.float32) - np.asarray(actor_center, np.float32))
    target = _normalise(np.asarray(target_origin, np.float32) - np.asarray(actor_center, np.float32))
    return float(np.degrees(np.arccos(np.clip(float(np.dot(source, target)), -1.0, 1.0))))


def rasterize_proxy_samples(
    base_rgb: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
    alpha_dilation_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-buffer projected source samples and composite them over ``base_rgb``.

    Dynamic proxy depth is only an auxiliary display diagnostic.  In
    particular, static expected depth is *not* used for occlusion because the
    current PLY is known to contain dynamic leakage in these regions.
    """
    if radius < 0 or alpha_dilation_pixels < 0:
        raise ValueError("raster radius and alpha dilation must be non-negative")
    base = np.asarray(base_rgb, dtype=np.uint8)
    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError("base_rgb must be HxWx3")
    xy, z, rgb = np.asarray(pixels, np.float32), np.asarray(depths, np.float32), np.asarray(colors, np.uint8)
    if xy.ndim != 2 or xy.shape[1] != 2 or z.shape != (len(xy),) or rgb.shape != (len(xy), 3):
        raise ValueError("invalid proxy sample shapes")
    height, width = base.shape[:2]
    proxy = np.zeros_like(base)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    support = np.zeros((height, width), dtype=bool)
    for index in np.argsort(z, kind="stable").tolist():
        x, y = int(np.rint(xy[index, 0])), int(np.rint(xy[index, 1]))
        if not np.isfinite(z[index]) or z[index] <= .1:
            continue
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                xx, yy = x + dx, y + dy
                if xx < 0 or xx >= width or yy < 0 or yy >= height:
                    continue
                if not support[yy, xx] or z[index] < depth[yy, xx]:
                    proxy[yy, xx] = rgb[index]
                    depth[yy, xx] = z[index]
                    support[yy, xx] = True
    alpha = support.copy()
    if alpha_dilation_pixels and support.any():
        try:
            from scipy.ndimage import binary_dilation, distance_transform_edt
        except ImportError as exc:  # pragma: no cover - environment optional dependency
            raise RuntimeError("proxy alpha dilation requires scipy") from exc
        alpha = binary_dilation(support, iterations=alpha_dilation_pixels)
        _, nearest = distance_transform_edt(~support, return_indices=True)
        yy, xx = np.nonzero(alpha & ~support)
        proxy[yy, xx] = proxy[nearest[0, yy, xx], nearest[1, yy, xx]]
        depth[yy, xx] = depth[nearest[0, yy, xx], nearest[1, yy, xx]]
    result = base.copy()
    result[alpha] = proxy[alpha]
    return result, alpha, depth


def temporal_fade_weights(frame_indices: list[int], eligible: list[bool], fade_frames: int) -> np.ndarray:
    """Return fail-closed segment-local fade weights for display proxy frames."""
    if len(frame_indices) != len(eligible) or fade_frames < 0:
        raise ValueError("frame indices/eligibility/fade parameters are invalid")
    weights = np.zeros(len(frame_indices), dtype=np.float32)
    start = 0
    while start < len(frame_indices):
        if not eligible[start]:
            start += 1
            continue
        end = start + 1
        while end < len(frame_indices) and eligible[end] and frame_indices[end] == frame_indices[end - 1] + 1:
            end += 1
        if fade_frames == 0:
            weights[start:end] = 1.0
        else:
            length = end - start
            position = np.arange(length, dtype=np.float32)
            weights[start:end] = np.minimum(1.0, np.minimum((position + 1.0) / fade_frames, (length - position) / fade_frames))
        start = end
    return weights


def _nearest_dynamic_frame(frames: list[dict[str, Any]], timestamp_us: int, maximum_delta_us: int) -> dict[str, Any] | None:
    candidates = [item for item in frames if isinstance(item, dict) and "reference_timestamp_end_us" in item]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(int(item["reference_timestamp_end_us"]) - timestamp_us))
    return nearest if abs(int(nearest["reference_timestamp_end_us"]) - timestamp_us) <= maximum_delta_us else None


def _source_record(frame: dict[str, Any], source_camera_id: str, track_id: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for record in frame.get("cameras", []):
        if str(record.get("camera_id")) != source_camera_id:
            continue
        for instance in record.get("instances", []):
            if int(instance.get("track_id", -1)) == track_id:
                return record, instance
    return None


def source_frame_index(frame: dict[str, Any]) -> int:
    """Return the original sequence index, including fused subset manifests.

    Renderer subset manifests enumerate their local records from zero, while
    their ``frame_id`` retains the original six-digit NCore frame index.  A
    source proxy window is an original-clip window, so it must use the latter
    when available.
    """
    frame_id = frame.get("frame_id")
    if isinstance(frame_id, str):
        prefix = frame_id.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return int(frame["index"])


@dataclass(frozen=True)
class SourceProxyCompositeOptions:
    static_render_root: Path
    dynamic_dataset_manifest: Path
    ncore_path: Path
    output_root: Path
    track_id: int
    source_camera_id: str
    device: str = "cuda"
    frame_start: int | None = None
    frame_end: int | None = None
    maximum_timestamp_delta_us: int = 40_000
    maximum_view_angle_deg: float = 25.0
    max_source_pixels: int = 100_000
    mask_component_mode: str = "largest"
    minimum_source_mask_fraction: float = .0
    temporal_fade_frames: int = 0
    splat_radius_pixels: int = 1
    alpha_dilation_pixels: int = 1
    max_proxy_fraction: float = .25
    geometry_mode: str = "plane"
    cuboid_margin_m: float = .10

    def __post_init__(self) -> None:
        if self.track_id < 0 or not self.source_camera_id:
            raise ValueError("track_id and source_camera_id are required")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_end < 0:
            raise ValueError("frame_end must be non-negative")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.maximum_timestamp_delta_us < 0 or not 0.0 < self.maximum_view_angle_deg <= 90.0:
            raise ValueError("timestamp delta or view-angle gate is invalid")
        if self.max_source_pixels <= 0 or self.splat_radius_pixels < 0 or self.alpha_dilation_pixels < 0:
            raise ValueError("proxy sampling/raster parameters are invalid")
        if self.mask_component_mode not in {"largest", "all"}:
            raise ValueError("mask_component_mode must be 'largest' or 'all'")
        if not 0.0 <= self.minimum_source_mask_fraction <= 1.0 or self.temporal_fade_frames < 0:
            raise ValueError("source-mask gate or temporal fade is invalid")
        if not 0.0 < self.max_proxy_fraction <= 1.0:
            raise ValueError("max_proxy_fraction must be in (0,1]")
        if self.geometry_mode not in {"plane", "cuboid_faces"}:
            raise ValueError("geometry_mode must be 'plane' or 'cuboid_faces'")
        if not 0.0 <= self.cuboid_margin_m <= 1.0:
            raise ValueError("cuboid_margin_m must be in [0,1]")


@dataclass(frozen=True)
class SourceProxyStitchOptions:
    static_render_dir: Path
    chunk_dirs: tuple[Path, ...]
    output_dir: Path
    temporal_fade_frames: int = 5
    minimum_raw_proxy_fraction: float = 0.0
    minimum_contiguous_frames: int = 1

    def __post_init__(self) -> None:
        if not self.chunk_dirs or len(set(self.chunk_dirs)) != len(self.chunk_dirs):
            raise ValueError("chunk_dirs must be non-empty and unique")
        if self.temporal_fade_frames < 0:
            raise ValueError("temporal_fade_frames must be non-negative")
        if not 0.0 <= self.minimum_raw_proxy_fraction <= 1.0:
            raise ValueError("minimum_raw_proxy_fraction must be in [0,1]")
        if self.minimum_contiguous_frames < 1:
            raise ValueError("minimum_contiguous_frames must be at least one")


def _source_mask_pixels(mask: np.ndarray, maximum: int) -> np.ndarray:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    pixels = np.stack((xx, yy), axis=1).astype(np.float32)
    if len(pixels) > maximum:
        pixels = pixels[np.linspace(0, len(pixels) - 1, maximum, dtype=np.int64)]
    return pixels


def largest_mask_component(mask: np.ndarray) -> np.ndarray:
    """Keep one actor silhouette instead of reprojecting detached mask noise."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("source mask must be 2-D")
    if not value.any():
        return value
    try:
        from scipy.ndimage import label
    except ImportError as exc:  # pragma: no cover - environment optional dependency
        raise RuntimeError("largest-component source proxy masking requires scipy") from exc
    components, count = label(value)
    if count <= 1:
        return value
    sizes = np.bincount(components.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _proxy_for_frame(
    *,
    target_frame: dict[str, Any],
    dynamic_frame: dict[str, Any],
    track: dict[str, Any],
    source_camera_id: str,
    source_model: Any,
    loader: Any,
    dataset_root: Path,
    base_rgb: np.ndarray,
    options: SourceProxyCompositeOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    found = _source_record(dynamic_frame, source_camera_id, options.track_id)
    if found is None:
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {"accepted": False, "reason": "source_track_unobserved"}
    record, instance = found
    timestamp_us = int(dynamic_frame["reference_timestamp_end_us"])
    actor_pose = _interpolate_track_pose(track, timestamp_us)
    if actor_pose is None:
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {"accepted": False, "reason": "track_pose_unavailable"}
    source_c2w = np.asarray(record["camera_to_world_start"], dtype=np.float64)
    source_origin = source_c2w[:3, 3].astype(np.float32)
    target_w2c = np.asarray(target_frame["world_to_camera_start"], dtype=np.float32)
    target_origin = _target_origin(target_w2c)
    actor_center = actor_pose[:3, 3].astype(np.float32)
    angle = view_angle_degrees(actor_center, source_origin, target_origin)
    if angle > options.maximum_view_angle_deg:
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {"accepted": False, "reason": "view_angle_gate", "view_angle_deg": angle}
    image = np.asarray(loader.get_camera_sensor(source_camera_id).get_frame_image_array(int(record["source_frame_index"])), dtype=np.uint8)
    raw_mask = np.asarray(Image.open(dataset_root / str(instance["mask"])).convert("L"), dtype=np.uint8) > 0
    mask = largest_mask_component(raw_mask) if options.mask_component_mode == "largest" else raw_mask
    if image.shape[:2] != mask.shape:
        raise ValueError("source image and dynamic instance mask dimensions disagree")
    source_fraction = float(mask.mean())
    if source_fraction < options.minimum_source_mask_fraction:
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {
            "accepted": False, "reason": "source_mask_fraction_gate", "view_angle_deg": angle,
            "source_mask_fraction": source_fraction,
        }
    pixels = _source_mask_pixels(mask, options.max_source_pixels)
    if not len(pixels):
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {"accepted": False, "reason": "empty_source_mask", "view_angle_deg": angle}
    # This is an inference-only image warp.  Leaving autograd enabled is
    # harmless for a five-frame pilot but retains CUDA graphs over a long
    # sequence and eventually exhausts memory.
    import torch
    with torch.inference_mode():
        returned = source_model.image_points_to_world_rays_shutter_pose(
            pixels,
            np.asarray(record["camera_to_world_start"], dtype=np.float64),
            np.asarray(record["camera_to_world_end"], dtype=np.float64),
            int(record["timestamp_start_us"]), int(record["timestamp_end_us"]),
        )
    rays = returned.world_rays.detach().float().cpu().numpy()
    if options.geometry_mode == "cuboid_faces":
        points, valid_plane, face_axis = cuboid_face_intersections(
            rays[:, :3], rays[:, 3:], actor_pose, np.asarray(track["length_width_height"], np.float32),
            margin_m=options.cuboid_margin_m,
        )
    else:
        points, valid_plane = source_facing_plane_intersections(rays[:, :3], rays[:, 3:], actor_center, source_origin)
        face_axis = np.full(len(points), -1, dtype=np.int8)
    colors = image[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
    target_xy, target_depth, valid_target = project_pinhole(points, target_w2c, dict(target_frame["intrinsics"]))
    valid = valid_plane & valid_target
    result, alpha, proxy_depth = rasterize_proxy_samples(
        base_rgb, target_xy[valid], target_depth[valid], colors[valid], radius=options.splat_radius_pixels,
        alpha_dilation_pixels=options.alpha_dilation_pixels,
    )
    fraction = float(alpha.mean())
    if fraction > options.max_proxy_fraction:
        empty = np.zeros(base_rgb.shape[:2], dtype=bool)
        return base_rgb.copy(), empty, np.full(base_rgb.shape[:2], np.nan, np.float32), {
            "accepted": False, "reason": "proxy_fraction_gate", "view_angle_deg": angle,
            "source_mask_fraction": source_fraction, "pre_gate_proxy_fraction": fraction,
        }
    return result, alpha, proxy_depth, {
        "accepted": True, "source_frame_index": int(record["source_frame_index"]),
        "dynamic_reference_frame_index": int(dynamic_frame["reference_frame_index"]),
        "source_mask_pixels": int(mask.sum()), "source_mask_fraction": source_fraction,
        "raw_source_mask_pixels": int(raw_mask.sum()),
        "mask_component_mode": options.mask_component_mode, "sampled_source_pixels": int(len(pixels)),
        "plane_valid_samples": int(valid_plane.sum()), "target_valid_samples": int(valid.sum()),
        "geometry_mode": options.geometry_mode,
        "cuboid_face_samples": {str(axis): int(((face_axis == axis) & valid).sum()) for axis in range(3)} if options.geometry_mode == "cuboid_faces" else {},
        "view_angle_deg": angle, "proxy_fraction": fraction,
    }


def _composite_unit(
    input_dir: Path, output_dir: Path, *, loader: Any, source_model: Any, dataset: dict[str, Any], options: SourceProxyCompositeOptions,
) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    if manifest.get("status") != "complete" or not isinstance(manifest.get("frames"), list):
        raise ValueError(f"static-render-dir is not a complete sequence: {input_dir}")
    tracks = {int(item["track_id"]): item for item in dataset.get("tracks", []) if isinstance(item, dict) and "track_id" in item}
    if options.track_id not in tracks:
        raise ValueError(f"track {options.track_id} is not present in dynamic dataset")
    dataset_root = options.dynamic_dataset_manifest.parent
    dynamic_frames = [item for item in dataset.get("frames", []) if isinstance(item, dict)]
    pending: list[dict[str, Any]] = []
    selected = [item for item in manifest["frames"] if isinstance(item, dict) and item.get("complete")]
    for frame in selected:
        index = int(frame["index"])
        original_index = source_frame_index(frame)
        if (options.frame_start is not None and original_index < options.frame_start
                or options.frame_end is not None and original_index >= options.frame_end):
            continue
        outputs = frame.get("outputs")
        if not isinstance(outputs, dict) or not isinstance(outputs.get("rgb"), str):
            raise ValueError("target render frame lacks RGB output")
        if frame.get("camera_model") != "pinhole" or not isinstance(frame.get("intrinsics"), dict):
            raise ValueError("source proxy requires target pinhole render frames")
        base = np.asarray(Image.open(input_dir / outputs["rgb"]).convert("RGB"), dtype=np.uint8)
        dynamic = _nearest_dynamic_frame(dynamic_frames, int(frame["timestamp_us"]), options.maximum_timestamp_delta_us)
        if dynamic is None:
            result, alpha, depth, details = base.copy(), np.zeros(base.shape[:2], bool), np.full(base.shape[:2], np.nan, np.float32), {"accepted": False, "reason": "no_near_dynamic_reference"}
        else:
            result, alpha, depth, details = _proxy_for_frame(
                target_frame=frame, dynamic_frame=dynamic, track=tracks[options.track_id], source_camera_id=options.source_camera_id,
                source_model=source_model, loader=loader, dataset_root=dataset_root, base_rgb=base, options=options,
            )
        stem = Path(str(outputs["rgb"])).stem
        # Stream raw candidates immediately.  A full 299-frame sequence must
        # not retain every RGB/depth image in memory only to discover segment
        # fade weights at the end.
        _atomic_png(result, output_dir / "proxy_rgb_raw" / f"{stem}.png", "RGB")
        _atomic_png(alpha.astype(np.uint8) * 255, output_dir / "proxy_alpha_raw" / f"{stem}.png", "L")
        _atomic_npy(depth, output_dir / "dynamic_proxy_depth" / f"{stem}.npy")
        pending.append({"index": index, "source_frame_index": original_index, "frame_id": frame["frame_id"], "timestamp_us": frame["timestamp_us"], "stem": stem,
                        "base_rgb_relative": str(outputs["rgb"]), "details": details})
        print(f"Source proxy raw frame {index}: {details.get('reason', 'accepted')} coverage={float(alpha.mean()):.2%}", flush=True)
    if not pending:
        raise ValueError("requested target frame window is empty")
    weights = temporal_fade_weights(
        [int(item["index"]) for item in pending], [bool(item["details"].get("accepted")) for item in pending], options.temporal_fade_frames,
    )
    records: list[dict[str, Any]] = []
    for item, weight in zip(pending, weights.tolist(), strict=True):
        base = np.asarray(Image.open(input_dir / str(item["base_rgb_relative"])).convert("RGB"), dtype=np.uint8)
        stem = str(item["stem"])
        raw_proxy = np.asarray(Image.open(output_dir / "proxy_rgb_raw" / f"{stem}.png").convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(Image.open(output_dir / "proxy_alpha_raw" / f"{stem}.png").convert("L"), dtype=np.float32) / 255.0
        effective = alpha * float(weight)
        display = np.rint((1.0 - effective[..., None]) * base.astype(np.float32) + effective[..., None] * raw_proxy.astype(np.float32)).astype(np.uint8)
        _atomic_png(display, output_dir / "rgb_display" / f"{stem}.png", "RGB")
        _atomic_png(np.rint(effective * 255.0).astype(np.uint8), output_dir / "dynamic_proxy_alpha" / f"{stem}.png", "L")
        details = dict(item["details"]) | {"temporal_fade_weight": float(weight), "effective_proxy_fraction": float(effective.mean())}
        records.append({"index": item["index"], "source_frame_index": item["source_frame_index"], "frame_id": item["frame_id"], "timestamp_us": item["timestamp_us"],
                        "rgb_display": f"rgb_display/{stem}.png", "dynamic_proxy_alpha": f"dynamic_proxy_alpha/{stem}.png",
                        "dynamic_proxy_depth": f"dynamic_proxy_depth/{stem}.npy", "details": details})
        print(f"Source proxy frame {item['index']}: {details.get('reason', 'accepted')} coverage={float(effective.mean()):.2%} fade={float(weight):.2f}", flush=True)
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "static_render_dir": str(input_dir),
              "dynamic_dataset_manifest": str(options.dynamic_dataset_manifest.resolve()), "track_id": options.track_id,
              "source_camera_id": options.source_camera_id, "frames": records,
              "parameters": {"maximum_timestamp_delta_us": options.maximum_timestamp_delta_us, "maximum_view_angle_deg": options.maximum_view_angle_deg,
                             "max_source_pixels": options.max_source_pixels, "mask_component_mode": options.mask_component_mode,
                             "minimum_source_mask_fraction": options.minimum_source_mask_fraction,
                             "temporal_fade_frames": options.temporal_fade_frames,
                             "splat_radius_pixels": options.splat_radius_pixels, "alpha_dilation_pixels": options.alpha_dilation_pixels,
                             "max_proxy_fraction": options.max_proxy_fraction, "geometry_mode": options.geometry_mode,
                             "cuboid_margin_m": options.cuboid_margin_m},
              "usage_warning": "Display-only source-conditioned 2.5-D proxy. rgb_display is not sensor truth; renderer alpha and expected depth are not copied or modified. cuboid_faces uses track-box entrance faces only and leaves unknown actor surface transparent."}
    _atomic_json(report, output_dir / "manifest.json")
    return report


def composite_source_proxy(options: SourceProxyCompositeOptions) -> dict[str, Any]:
    """Create a short target-view source-image proxy experiment without training."""
    from ncore.impl.sensors.camera import FThetaCameraModel

    if options.output_root.exists() and any(options.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {options.output_root}")
    dataset = _read_json(options.dynamic_dataset_manifest)
    if dataset.get("status") != "complete":
        raise ValueError("dynamic dataset manifest is not complete")
    if Path(str(dataset.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic dataset ncore_path does not match --ncore-path")
    loader = _create_ncore_loader(options.ncore_path)
    source_model = FThetaCameraModel(loader.get_camera_sensor(options.source_camera_id).model_parameters, device=options.device)
    units = _render_units(options.static_render_root)
    options.output_root.mkdir(parents=True, exist_ok=True)
    multiple = len(units) > 1
    reports = {name or "camera": _composite_unit(directory, options.output_root / name if multiple else options.output_root,
                                                   loader=loader, source_model=source_model, dataset=dataset, options=options)
               for name, directory in sorted(units.items())}
    result = {"schema": SCHEMA, "version": VERSION, "status": "complete", "units": reports,
              "usage_warning": "Single-source actor texture planes are a display fallback only, not a dynamic reconstruction."}
    _atomic_json(result, options.output_root / "source_proxy_composite_report.json")
    return result


def stitch_source_proxy_chunks(options: SourceProxyStitchOptions) -> dict[str, Any]:
    """Stitch completed raw-proxy shards with one global temporal fade policy.

    Rendering source FTheta rays is intentionally split into bounded jobs; this
    finalizer is CPU-only and never reconstructs an actor.  It rejects missing
    or overlapping frame records rather than silently mixing chunk outputs.
    """
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {options.output_dir}")
    static_manifest = _read_json(options.static_render_dir / "manifest.json")
    if static_manifest.get("status") != "complete" or not isinstance(static_manifest.get("frames"), list):
        raise ValueError("static_render_dir must contain a completed render manifest")
    chunks: dict[int, tuple[Path, dict[str, Any]]] = {}
    for directory in options.chunk_dirs:
        manifest = _read_json(directory / "manifest.json")
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError(f"chunk is not a completed source-proxy result: {directory}")
        for record in manifest.get("frames", []):
            if not isinstance(record, dict):
                raise ValueError("source-proxy chunk has malformed frame record")
            index = int(record["index"])
            if index in chunks:
                raise ValueError(f"source-proxy chunks overlap at frame {index}")
            chunks[index] = (directory, record)
    static_frames = [item for item in static_manifest["frames"] if isinstance(item, dict) and item.get("complete")]
    indices = [int(item["index"]) for item in static_frames]
    missing = [index for index in indices if index not in chunks]
    extra = sorted(set(chunks) - set(indices))
    if missing or extra:
        raise ValueError(f"source-proxy chunks do not exactly cover static sequence; missing={missing[:8]} extra={extra[:8]}")
    # A source mask can remain formally accepted after it has collapsed to a
    # small fragment of an exiting actor.  Keeping that fragment as a valid
    # continuation makes the display actor abruptly shrink.  This is a
    # display-only completeness gate: the raw shard remains intact and an
    # ineligible frame simply falls back to the static render.
    raw_fractions: list[float] = []
    for index in indices:
        chunk_dir, record = chunks[index]
        stem = Path(str(record["rgb_display"])).stem
        alpha = np.asarray(
            Image.open(chunk_dir / "proxy_alpha_raw" / f"{stem}.png").convert("L"), dtype=np.float32,
        ) / 255.0
        raw_fractions.append(float(alpha.mean()))
    eligible = [
        bool(chunks[index][1].get("details", {}).get("accepted")) and fraction >= options.minimum_raw_proxy_fraction
        for index, fraction in zip(indices, raw_fractions, strict=True)
    ]
    # Suppress momentary fragments that would otherwise create short flashes
    # between static-only frames.  This is applied after the fraction gate so
    # it cannot invent a continuation from a missing raw observation.
    start = 0
    while start < len(eligible):
        if not eligible[start]:
            start += 1
            continue
        end = start + 1
        while end < len(eligible) and eligible[end] and indices[end] == indices[end - 1] + 1:
            end += 1
        if end - start < options.minimum_contiguous_frames:
            eligible[start:end] = [False] * (end - start)
        start = end
    weights = temporal_fade_weights(indices, eligible, options.temporal_fade_frames)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for static_frame, index, weight, raw_fraction in zip(
        static_frames, indices, weights.tolist(), raw_fractions, strict=True,
    ):
        chunk_dir, record = chunks[index]
        outputs = static_frame.get("outputs")
        if not isinstance(outputs, dict) or not isinstance(outputs.get("rgb"), str):
            raise ValueError("static render frame lacks RGB output")
        stem = Path(str(record["rgb_display"])).stem
        base = np.asarray(Image.open(options.static_render_dir / str(outputs["rgb"])).convert("RGB"), dtype=np.uint8)
        raw = np.asarray(Image.open(chunk_dir / "proxy_rgb_raw" / f"{stem}.png").convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(Image.open(chunk_dir / "proxy_alpha_raw" / f"{stem}.png").convert("L"), dtype=np.float32) / 255.0
        if raw.shape != base.shape or alpha.shape != base.shape[:2]:
            raise ValueError(f"raw proxy/static shape mismatch at frame {index}")
        effective = alpha * float(weight)
        display = np.rint((1.0 - effective[..., None]) * base.astype(np.float32) + effective[..., None] * raw.astype(np.float32)).astype(np.uint8)
        _atomic_png(display, options.output_dir / "rgb_display" / f"{stem}.png", "RGB")
        _atomic_png(np.rint(effective * 255.0).astype(np.uint8), options.output_dir / "dynamic_proxy_alpha" / f"{stem}.png", "L")
        source_depth = chunk_dir / "dynamic_proxy_depth" / f"{stem}.npy"
        destination_depth = options.output_dir / "dynamic_proxy_depth" / f"{stem}.npy"
        destination_depth.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_depth, destination_depth)
        details = dict(record.get("details", {})) | {
            "temporal_fade_weight": float(weight),
            "effective_proxy_fraction": float(effective.mean()),
            "raw_proxy_fraction": raw_fraction,
            "stitch_eligible": bool(weight > 0.0),
        }
        records.append({"index": index, "frame_id": static_frame["frame_id"], "timestamp_us": static_frame["timestamp_us"],
                        "rgb_display": f"rgb_display/{stem}.png", "dynamic_proxy_alpha": f"dynamic_proxy_alpha/{stem}.png",
                        "dynamic_proxy_depth": f"dynamic_proxy_depth/{stem}.npy", "details": details})
        if index % 25 == 0 or index == indices[-1]:
            print(f"Stitched source proxy frame {index}/{indices[-1]} fade={float(weight):.2f}", flush=True)
    result = {"schema": SCHEMA, "version": VERSION, "status": "complete", "static_render_dir": str(options.static_render_dir),
              "source_proxy_chunks": [str(item) for item in options.chunk_dirs], "frame_count": len(records), "frames": records,
              "parameters": {
                  "temporal_fade_frames": options.temporal_fade_frames,
                  "minimum_raw_proxy_fraction": options.minimum_raw_proxy_fraction,
                  "minimum_contiguous_frames": options.minimum_contiguous_frames,
              },
              "usage_warning": "Display-only source-conditioned proxy sequence. Static renderer alpha/depth remain external, unchanged geometry authority."}
    _atomic_json(result, options.output_dir / "manifest.json")
    return result
