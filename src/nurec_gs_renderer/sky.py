from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera import CameraFrame, interpolate_w2c


FACE_NAMES = ("+X", "-X", "-Y", "+Y", "+Z", "-Z")
ASSET_VERSION = 1
TEMPORAL_ASSET_VERSION = 1


@dataclass(frozen=True)
class SkyCubemap:
    """World-direction sky asset using the InstantNuRec cubemap convention."""

    rgb: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    sample_count: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 4 or rgb.shape[0] != 6 or rgb.shape[-1] != 3 or rgb.shape[1] != rgb.shape[2]:
            raise ValueError(f"cubemap rgb must have shape (6, S, S, 3), got {rgb.shape}")
        scalar_shape = rgb.shape[:3]
        for name, value in (
            ("valid", self.valid),
            ("confidence", self.confidence),
            ("sample_count", self.sample_count),
        ):
            if np.asarray(value).shape != scalar_shape:
                raise ValueError(f"cubemap {name} must have shape {scalar_shape}, got {np.asarray(value).shape}")
        if not np.issubdtype(rgb.dtype, np.floating):
            raise ValueError("cubemap rgb must use a floating dtype")
        if not np.all(np.isfinite(rgb)) or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("cubemap rgb must contain finite values in [0, 1]")
        confidence = np.asarray(self.confidence)
        if not np.all(np.isfinite(confidence)) or np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("cubemap confidence must contain finite values in [0, 1]")
        if tuple(self.metadata.get("face_order", ())) != FACE_NAMES:
            raise ValueError(f"cubemap face_order must be {FACE_NAMES}")
        if int(self.metadata.get("asset_version", -1)) != ASSET_VERSION:
            raise ValueError(f"unsupported sky cubemap asset_version: {self.metadata.get('asset_version')}")

    @property
    def face_size(self) -> int:
        return int(self.rgb.shape[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("sky cubemap output must use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rgb=np.asarray(self.rgb, dtype=np.float16),
            valid=np.asarray(self.valid, dtype=np.uint8),
            confidence=np.asarray(self.confidence, dtype=np.float16),
            sample_count=np.asarray(self.sample_count, dtype=np.uint16),
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SkyCubemap":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"rgb", "valid", "confidence", "sample_count", "metadata"}
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"sky cubemap is missing arrays: {sorted(missing)}")
                expected_dtypes = {
                    "rgb": np.dtype(np.float16),
                    "valid": np.dtype(np.uint8),
                    "confidence": np.dtype(np.float16),
                    "sample_count": np.dtype(np.uint16),
                }
                for name, expected_dtype in expected_dtypes.items():
                    if data[name].dtype != expected_dtype:
                        raise ValueError(
                            f"sky cubemap {name} must use dtype {expected_dtype}, got {data[name].dtype}"
                        )
                metadata = json.loads(str(data["metadata"].item()))
                if not isinstance(metadata, dict):
                    raise ValueError("sky cubemap metadata must be a JSON object")
                return cls(
                    rgb=np.asarray(data["rgb"], dtype=np.float32),
                    valid=np.asarray(data["valid"], dtype=np.uint8),
                    confidence=np.asarray(data["confidence"], dtype=np.float32),
                    sample_count=np.asarray(data["sample_count"], dtype=np.uint16),
                    metadata=metadata,
                )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load sky cubemap {path}: {exc}") from exc


class TemporalSkyCubemap:
    """Lazily loaded time-indexed cubemap sequence with linear-light blending."""

    def __init__(
        self,
        manifest_path: Path,
        timestamps_us: tuple[int, ...],
        asset_paths: tuple[Path, ...],
        metadata: dict[str, Any],
        *,
        cache_size: int = 3,
    ) -> None:
        if len(timestamps_us) == 0 or len(timestamps_us) != len(asset_paths):
            raise ValueError("temporal sky timestamps and assets must be non-empty and aligned")
        if any(right <= left for left, right in zip(timestamps_us, timestamps_us[1:])):
            raise ValueError("temporal sky timestamps must be strictly increasing")
        if cache_size < 2:
            raise ValueError("temporal sky cache_size must be at least two")
        self.manifest_path = manifest_path
        self.timestamps_us = timestamps_us
        self.asset_paths = asset_paths
        self.metadata = metadata
        self.cache_size = cache_size
        self._cache: OrderedDict[int, SkyCubemap] = OrderedDict()
        first = self._load_slot(0)
        self.face_size = first.face_size
        self.total_coverage = float(
            np.mean([float(value) for value in metadata.get("slot_coverage", [first.valid.mean()])])
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalSkyCubemap":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load temporal sky manifest {path}: {exc}") from exc
        if manifest.get("type") != "temporal_sky_cubemap":
            raise ValueError("temporal sky manifest has an invalid type")
        if int(manifest.get("asset_version", -1)) != TEMPORAL_ASSET_VERSION:
            raise ValueError(f"unsupported temporal sky asset version: {manifest.get('asset_version')}")
        slots = manifest.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("temporal sky manifest requires a non-empty slots list")
        timestamps = tuple(int(slot["timestamp_us"]) for slot in slots)
        assets = tuple((path.parent / str(slot["asset"])).resolve() for slot in slots)
        missing = [str(asset) for asset in assets if not asset.is_file()]
        if missing:
            raise FileNotFoundError(f"temporal sky slot assets are missing: {missing[:3]}")
        return cls(path.resolve(), timestamps, assets, manifest)

    def _load_slot(self, index: int) -> SkyCubemap:
        cached = self._cache.pop(index, None)
        if cached is None:
            cached = SkyCubemap.load(self.asset_paths[index])
            if self._cache and cached.face_size != next(iter(self._cache.values())).face_size:
                raise ValueError("temporal sky slot face sizes do not match")
        self._cache[index] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def bracket(self, timestamp_us: int | None) -> tuple[int, int, float]:
        if timestamp_us is None or timestamp_us <= self.timestamps_us[0]:
            return 0, 0, 0.0
        if timestamp_us >= self.timestamps_us[-1]:
            last = len(self.timestamps_us) - 1
            return last, last, 0.0
        upper = bisect.bisect_right(self.timestamps_us, int(timestamp_us))
        lower = upper - 1
        span = self.timestamps_us[upper] - self.timestamps_us[lower]
        weight = (int(timestamp_us) - self.timestamps_us[lower]) / span
        return lower, upper, float(weight)

    def sample(
        self,
        camera: CameraFrame,
        *,
        confidence_threshold: float,
        min_sample_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper, weight = self.bracket(camera.timestamp_us)
        rgb_lower, valid_lower = sample_sky_cubemap(
            self._load_slot(lower),
            camera,
            confidence_threshold=confidence_threshold,
            min_sample_count=min_sample_count,
        )
        if lower == upper:
            return rgb_lower, valid_lower
        rgb_upper, valid_upper = sample_sky_cubemap(
            self._load_slot(upper),
            camera,
            confidence_threshold=confidence_threshold,
            min_sample_count=min_sample_count,
        )
        lower_ok = valid_lower > 0.0
        upper_ok = valid_upper > 0.0
        both = lower_ok & upper_ok
        output = np.zeros_like(rgb_lower)
        if both.any():
            linear = (1.0 - weight) * srgb_to_linear(rgb_lower[both]) + weight * srgb_to_linear(
                rgb_upper[both]
            )
            output[both] = linear_to_srgb(linear)
        only_lower = lower_ok & ~upper_ok
        only_upper = upper_ok & ~lower_ok
        output[only_lower] = rgb_lower[only_lower]
        output[only_upper] = rgb_upper[only_upper]
        valid = np.where(
            both,
            (1.0 - weight) * valid_lower + weight * valid_upper,
            np.maximum(valid_lower, valid_upper),
        )
        return output.astype(np.float32), valid.astype(np.float32)


def load_sky_asset(path: str | Path) -> SkyCubemap | TemporalSkyCubemap:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return TemporalSkyCubemap.load(path)
    return SkyCubemap.load(path)


def sample_sky_asset(
    asset: SkyCubemap | TemporalSkyCubemap,
    camera: CameraFrame,
    *,
    confidence_threshold: float = 0.0,
    min_sample_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(asset, TemporalSkyCubemap):
        return asset.sample(
            camera,
            confidence_threshold=confidence_threshold,
            min_sample_count=min_sample_count,
        )
    return sample_sky_cubemap(
        asset,
        camera,
        confidence_threshold=confidence_threshold,
        min_sample_count=min_sample_count,
    )


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def cubemap_ray_directions(face_size: int) -> np.ndarray:
    """Return normalized world directions with shape (6, S, S, 3)."""
    if face_size <= 0:
        raise ValueError("face_size must be positive")
    px = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size * 2.0 - 1.0
    uu, vv = np.meshgrid(px, px, indexing="xy")
    front = np.stack((uu, vv, np.ones_like(uu)), axis=-1)
    front /= np.linalg.norm(front, axis=-1, keepdims=True)
    x, y, z = np.moveaxis(front, -1, 0)
    faces = np.stack(
        (
            np.stack((z, y, -x), axis=-1),
            np.stack((-z, y, x), axis=-1),
            np.stack((x, -z, y), axis=-1),
            np.stack((x, z, -y), axis=-1),
            front,
            np.stack((-x, y, -z), axis=-1),
        ),
        axis=0,
    )
    return faces.astype(np.float32)


def directions_to_cubemap_uv(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map directions to face indices and per-face coordinates in [-1, 1]."""
    directions = np.asarray(directions, dtype=np.float32)
    if directions.shape[-1] != 3:
        raise ValueError("directions must have a final dimension of 3")
    x, y, z = np.moveaxis(directions, -1, 0)
    absolute = np.abs(directions)
    dominant = np.argmax(absolute, axis=-1)
    scale = np.take_along_axis(absolute, dominant[..., None], axis=-1)[..., 0]
    scale = np.maximum(scale, 1e-12)
    positive = np.take_along_axis(directions, dominant[..., None], axis=-1)[..., 0] > 0.0
    face = np.where(
        dominant == 0,
        np.where(positive, 0, 1),
        np.where(dominant == 1, np.where(positive, 3, 2), np.where(positive, 4, 5)),
    ).astype(np.int64)
    u_faces = np.stack((-z / scale, z / scale, x / scale, x / scale, x / scale, -x / scale), axis=-1)
    v_faces = np.stack((y / scale, y / scale, z / scale, -z / scale, y / scale, y / scale), axis=-1)
    u = np.take_along_axis(u_faces, face[..., None], axis=-1)[..., 0]
    v = np.take_along_axis(v_faces, face[..., None], axis=-1)[..., 0]
    return face, u.astype(np.float32), v.astype(np.float32)


def _eval_poly(coefficients: tuple[float, ...], value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float64)
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _ftheta_camera_rays(camera: CameraFrame) -> tuple[np.ndarray, np.ndarray]:
    ftheta = camera.ftheta
    if ftheta is None:
        raise ValueError("FTheta rays require FTheta camera parameters")
    width, height = camera.intrinsics.width, camera.intrinsics.height
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64), indexing="xy")
    points = np.stack((xx - camera.intrinsics.cx, yy - camera.intrinsics.cy), axis=-1)
    c, d, e = ftheta.linear_cde
    determinant = c - e * d
    if abs(determinant) < 1e-12:
        raise ValueError("FTheta linear_cde is singular")
    inverse = np.asarray([[1.0, -d], [-e, c]], dtype=np.float64) / determinant
    distorted = points @ inverse.T
    radius = np.linalg.norm(distorted, axis=-1)
    if ftheta.reference_poly == "PIXELDIST_TO_ANGLE":
        theta = _eval_poly(ftheta.pixeldist_to_angle_poly, radius)
    else:
        theta = _eval_poly(ftheta.pixeldist_to_angle_poly, radius)
        coefficients = ftheta.angle_to_pixeldist_poly
        derivative = tuple(index * value for index, value in enumerate(coefficients))[1:]
        for _ in range(3):
            derivative_value = _eval_poly(derivative, theta)
            derivative_value = np.where(
                np.abs(derivative_value) < 1e-12,
                np.where(derivative_value < 0.0, -1e-12, 1e-12),
                derivative_value,
            )
            theta -= (_eval_poly(coefficients, theta) - radius) / derivative_value
    unit_xy = distorted / np.maximum(radius[..., None], 1e-6)
    rays = np.concatenate((np.sin(theta)[..., None] * unit_xy, np.cos(theta)[..., None]), axis=-1)
    rays[radius < 1e-6] = (0.0, 0.0, 1.0)
    valid = np.isfinite(theta) & (theta >= 0.0) & (theta <= ftheta.max_angle)
    return rays.astype(np.float32), valid


def _relative_shutter_times(camera: CameraFrame) -> np.ndarray:
    width, height = camera.intrinsics.width, camera.intrinsics.height
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64), indexing="xy")
    shutter = "GLOBAL" if camera.ftheta is None else camera.ftheta.shutter_type
    if shutter == "GLOBAL":
        return np.zeros((height, width), dtype=np.float64)
    if shutter == "ROLLING_TOP_TO_BOTTOM":
        return yy / max(height - 1, 1)
    if shutter == "ROLLING_BOTTOM_TO_TOP":
        return (height - 1 - yy) / max(height - 1, 1)
    if shutter == "ROLLING_LEFT_TO_RIGHT":
        return xx / max(width - 1, 1)
    if shutter == "ROLLING_RIGHT_TO_LEFT":
        return (width - 1 - xx) / max(width - 1, 1)
    raise ValueError(f"unsupported shutter type: {shutter}")


def camera_world_rays(camera: CameraFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate per-pixel world directions and projection-valid mask."""
    width, height = camera.intrinsics.width, camera.intrinsics.height
    if camera.camera_model == "pinhole":
        xx, yy = np.meshgrid(
            np.arange(width, dtype=np.float32) + 0.5,
            np.arange(height, dtype=np.float32) + 0.5,
            indexing="xy",
        )
        rays = np.stack(
            (
                (xx - camera.intrinsics.cx) / camera.intrinsics.fx,
                (yy - camera.intrinsics.cy) / camera.intrinsics.fy,
                np.ones_like(xx),
            ),
            axis=-1,
        )
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.ones((height, width), dtype=bool)
    else:
        rays, valid = _ftheta_camera_rays(camera)

    times = _relative_shutter_times(camera)
    if camera.world_to_camera_end is None:
        rotation = camera.world_to_camera[:3, :3]
        world = rays @ rotation
    else:
        rotations, _ = interpolate_w2c(camera.world_to_camera, camera.world_to_camera_end, times.reshape(-1))
        world = np.einsum("nij,nj->ni", np.swapaxes(rotations, 1, 2), rays.reshape(-1, 3)).reshape(
            height, width, 3
        )
    world /= np.maximum(np.linalg.norm(world, axis=-1, keepdims=True), 1e-12)
    return world.astype(np.float32), valid


def _bilinear_sample_face(values: np.ndarray, face: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    size = values.shape[1]
    px = np.clip((u + 1.0) * size * 0.5 - 0.5, 0.0, size - 1.0)
    py = np.clip((v + 1.0) * size * 0.5 - 0.5, 0.0, size - 1.0)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.minimum(x0 + 1, size - 1)
    y1 = np.minimum(y0 + 1, size - 1)
    wx = px - x0
    wy = py - y0
    suffix = (None,) * (values.ndim - 3)
    wx = wx[(...,) + suffix]
    wy = wy[(...,) + suffix]
    top = values[face, y0, x0] * (1.0 - wx) + values[face, y0, x1] * wx
    bottom = values[face, y1, x0] * (1.0 - wx) + values[face, y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def sample_sky_cubemap(
    cubemap: SkyCubemap,
    camera: CameraFrame,
    *,
    confidence_threshold: float = 0.0,
    min_sample_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if min_sample_count <= 0:
        raise ValueError("min_sample_count must be positive")
    directions, projection_valid = camera_world_rays(camera)
    face, u, v = directions_to_cubemap_uv(directions)
    rgb = _bilinear_sample_face(cubemap.rgb.astype(np.float32), face, u, v)
    valid = _bilinear_sample_face(cubemap.valid.astype(np.float32), face, u, v)
    confidence = _bilinear_sample_face(cubemap.confidence.astype(np.float32), face, u, v)
    sample_count = _bilinear_sample_face(cubemap.sample_count.astype(np.float32), face, u, v)
    quality_valid = (confidence >= confidence_threshold) & (sample_count >= min_sample_count)
    valid = np.clip(valid * projection_valid.astype(np.float32) * quality_valid.astype(np.float32), 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), valid.astype(np.float32)


def pad_sky_edges(
    rgb: np.ndarray,
    valid: np.ndarray,
    padding_pixels: int,
    *,
    valid_threshold: float = 0.5,
    feather_pixels: int = 0,
    smoothing_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend observed sky colors over a narrow target-image boundary band.

    The color is propagated independently from the validity matte so bilinear
    sampling never blends a silhouette edge directly with black. Large
    unobserved regions remain invalid.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    if rgb.shape[:2] != valid.shape or rgb.shape[-1] != 3:
        raise ValueError("sky rgb and valid shapes do not match")
    if padding_pixels < 0 or feather_pixels < 0 or feather_pixels > padding_pixels:
        raise ValueError("sky padding/feather parameters are invalid")
    if smoothing_iterations < 0:
        raise ValueError("sky smoothing iterations must be non-negative")
    if not 0.0 <= valid_threshold <= 1.0:
        raise ValueError("valid_threshold must be in [0, 1]")
    if padding_pixels == 0:
        return rgb.copy(), valid.copy()

    output_rgb = rgb.copy()
    current = valid >= valid_threshold
    original = current.copy()
    output_valid = np.where(current, 1.0, 0.0).astype(np.float32)
    height, width = current.shape
    for layer in range(1, padding_pixels + 1):
        padded_mask = np.pad(current.astype(np.float32), 1)
        padded_color = np.pad(output_rgb * current[..., None], ((1, 1), (1, 1), (0, 0)))
        neighbor_count = np.zeros((height, width), dtype=np.float32)
        neighbor_color = np.zeros((height, width, 3), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbor_count += padded_mask[dy : dy + height, dx : dx + width]
                neighbor_color += padded_color[dy : dy + height, dx : dx + width]
        added = ~current & (neighbor_count > 0.0)
        if not added.any():
            break
        output_rgb[added] = neighbor_color[added] / neighbor_count[added, None]
        if feather_pixels > 0 and layer > padding_pixels - feather_pixels:
            weight = (padding_pixels - layer + 1) / (feather_pixels + 1)
        else:
            weight = 1.0
        output_valid[added] = float(weight)
        current |= added
    filled = current & ~original
    for _ in range(smoothing_iterations):
        if not filled.any():
            break
        padded_mask = np.pad(current.astype(np.float32), 1)
        padded_color = np.pad(output_rgb * current[..., None], ((1, 1), (1, 1), (0, 0)))
        neighbor_count = np.zeros((height, width), dtype=np.float32)
        neighbor_color = np.zeros((height, width, 3), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbor_count += padded_mask[dy : dy + height, dx : dx + width]
                neighbor_color += padded_color[dy : dy + height, dx : dx + width]
        average = neighbor_color / np.maximum(neighbor_count[..., None], 1.0)
        output_rgb[filled] = average[filled]
    return np.clip(output_rgb, 0.0, 1.0), output_valid


def composite_sky(
    premultiplied_rgb: np.ndarray,
    geometry_alpha: np.ndarray,
    sky_rgb: np.ndarray,
    sky_valid: np.ndarray,
    background: tuple[float, float, float],
) -> np.ndarray:
    """Fill only the geometry-transmittance region with sky or fallback color."""
    background_rgb = np.asarray(background, dtype=np.float32)
    fill = sky_valid[..., None] * sky_rgb + (1.0 - sky_valid[..., None]) * background_rgb
    return np.clip(premultiplied_rgb + (1.0 - geometry_alpha[..., None]) * fill, 0.0, 1.0).astype(np.float32)
