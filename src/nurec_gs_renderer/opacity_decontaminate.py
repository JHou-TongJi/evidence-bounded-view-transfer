from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color_correction import SH_C0, file_sha256
from .dynamic import DynamicGaussianScene
from .opacity_correction import GaussianOpacityCorrection
from .ply_io import GaussianScene


@dataclass(frozen=True)
class DynamicProximityOpacityOptions:
    ply_path: Path
    dynamic_path: Path
    output: Path
    max_distance_m: float = 0.15
    max_color_distance: float = 0.10
    max_static_scale_m: float = 0.15
    opacity_scale: float = 0.0
    exclude_road: bool = True
    exclude_sky: bool = True

    def __post_init__(self) -> None:
        if self.output.suffix.lower() != ".npz":
            raise ValueError("opacity decontamination output must use the .npz extension")
        if self.max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be positive")
        if not 0.0 < self.max_color_distance <= 1.0:
            raise ValueError("max_color_distance must be in (0,1]")
        if self.max_static_scale_m <= 0.0:
            raise ValueError("max_static_scale_m must be positive")
        if not 0.0 <= self.opacity_scale <= 1.0:
            raise ValueError("opacity_scale must be in [0,1]")


def select_dynamic_proximity_candidates(
    scene: GaussianScene,
    dynamic_scene: DynamicGaussianScene,
    *,
    max_distance_m: float,
    max_color_distance: float,
    max_static_scale_m: float,
    road_mask: np.ndarray | None = None,
    sky_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select static Gaussians close and color-consistent with dynamic source points."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - optional offline dependency
        raise RuntimeError(
            "dynamic-proximity opacity decontamination requires scipy>=1.10"
        ) from exc
    if dynamic_scene.count == 0:
        raise ValueError("dynamic scene is empty")
    if road_mask is None:
        road_mask = np.zeros(scene.count, dtype=bool)
    if sky_mask is None:
        sky_mask = np.zeros(scene.count, dtype=bool)
    road_mask = np.asarray(road_mask, dtype=bool)
    sky_mask = np.asarray(sky_mask, dtype=bool)
    if road_mask.shape != (scene.count,) or sky_mask.shape != (scene.count,):
        raise ValueError("semantic exclusion masks must have shape [N]")

    tree = cKDTree(dynamic_scene.keyframe_positions[:, 1])
    distances, nearest = tree.query(scene.means, k=1, workers=-1)
    static_rgb = np.clip(scene.sh_coeffs[:, 0] * SH_C0 + 0.5, 0.0, 1.0)
    color_distances = np.abs(static_rgb - dynamic_scene.rgb[nearest]).mean(axis=1)
    static_scale = scene.scales.max(axis=1)
    selected_mask = (
        (distances <= max_distance_m)
        & (color_distances <= max_color_distance)
        & (static_scale <= max_static_scale_m)
        & ~road_mask
        & ~sky_mask
    )
    indices = np.flatnonzero(selected_mask).astype(np.int64)
    distance_confidence = np.clip(1.0 - distances[indices] / max_distance_m, 0.0, 1.0)
    color_confidence = np.clip(
        1.0 - color_distances[indices] / max_color_distance, 0.0, 1.0
    )
    confidence = np.sqrt(distance_confidence * color_confidence).astype(np.float32)
    return (
        indices,
        confidence,
        distances[indices].astype(np.float32),
        color_distances[indices].astype(np.float32),
    )


def _load_semantic_masks(
    path: Path,
    gaussian_count: int,
    *,
    exclude_road: bool,
    exclude_sky: bool,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - base dependency
        raise RuntimeError("opacity decontamination requires plyfile") from exc
    vertex = PlyData.read(str(path), mmap=True)["vertex"].data
    if len(vertex) != gaussian_count:
        raise ValueError("PLY semantic attribute count does not match scene")
    names = set(vertex.dtype.names or ())
    road = (
        np.asarray(vertex["road_mask"], dtype=np.uint8) > 0
        if exclude_road and "road_mask" in names
        else np.zeros(gaussian_count, dtype=bool)
    )
    sky = (
        np.asarray(vertex["sky_mask"], dtype=np.float32) > 0.5
        if exclude_sky and "sky_mask" in names
        else np.zeros(gaussian_count, dtype=bool)
    )
    return road, sky


def build_dynamic_proximity_opacity_correction(
    scene: GaussianScene,
    dynamic_scene: DynamicGaussianScene,
    options: DynamicProximityOpacityOptions,
) -> GaussianOpacityCorrection:
    road, sky = _load_semantic_masks(
        options.ply_path,
        scene.count,
        exclude_road=options.exclude_road,
        exclude_sky=options.exclude_sky,
    )
    indices, confidence, distances, color_distances = select_dynamic_proximity_candidates(
        scene,
        dynamic_scene,
        max_distance_m=options.max_distance_m,
        max_color_distance=options.max_color_distance,
        max_static_scale_m=options.max_static_scale_m,
        road_mask=road,
        sky_mask=sky,
    )
    if not len(indices):
        raise RuntimeError("dynamic proximity selected no static Gaussian candidates")
    original = scene.opacities[indices].astype(np.float32)
    corrected = (original * options.opacity_scale).astype(np.float32)
    details: dict[str, object] = {
        "method": "dynamic_source_geometry_proximity_v1",
        "dynamic_path": str(options.dynamic_path),
        "dynamic_sha256": file_sha256(options.dynamic_path),
        "dynamic_gaussian_count": dynamic_scene.count,
        "max_distance_m": options.max_distance_m,
        "max_color_distance": options.max_color_distance,
        "max_static_scale_m": options.max_static_scale_m,
        "opacity_scale": options.opacity_scale,
        "exclude_road": options.exclude_road,
        "exclude_sky": options.exclude_sky,
        "selected_gaussians": int(len(indices)),
        "selected_fraction": float(len(indices) / scene.count),
        "distance_m_percentiles": np.percentile(distances, [0, 25, 50, 75, 100]).tolist(),
        "color_distance_percentiles": np.percentile(
            color_distances, [0, 25, 50, 75, 100]
        ).tolist(),
    }
    correction = GaussianOpacityCorrection(
        indices=indices,
        original_opacities=original,
        corrected_opacities=corrected,
        confidence=confidence,
        support_count=np.ones(len(indices), dtype=np.uint16),
        ply_sha256=file_sha256(options.ply_path),
        gaussian_count=scene.count,
        metadata=details,
    )
    correction.save(options.output)
    return correction
