from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from .camera import CameraFrame, CameraPath, interpolate_w2c
from .dynamic import DYNAMIC_TIME_MODES, DynamicGaussianScene, infer_dynamic_source_slots
from .dynamic_association import DynamicActorAssociation
from .canonical_actor import CanonicalActorAsset, _interpolate_pose
from .ply_io import GaussianScene
from .sky import (
    SkyCubemap,
    TemporalSkyCubemap,
    composite_sky,
    pad_sky_edges,
    sample_sky_asset,
)


def canonical_visibility_allows_frame(metadata: dict[str, object], frame_id: str) -> bool:
    """Return whether a visibility-bound actor may appear at this source frame.

    Absent metadata preserves legacy canonical actor behaviour. A malformed
    explicit restriction fails closed so a single observed depth patch cannot
    silently become an all-time dynamic actor.
    """
    visibility_references = metadata.get("visibility_reference_indices")
    if visibility_references is None:
        return True
    try:
        allowed = {int(value) for value in visibility_references}  # type: ignore[union-attr]
        source_index = int(str(frame_id).split("_", 1)[0])
        return source_index in allowed
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class RenderOptions:
    device: str = "cuda"
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    far_plane: float = 10000.0
    packed: bool = True
    rasterize_mode: str = "classic"
    depth_alpha_threshold: float = 0.01
    sky_edge_padding_pixels: int = 2
    sky_edge_feather_pixels: int = 0
    sky_edge_smoothing_iterations: int = 0
    sky_valid_threshold: float = 0.5
    sky_confidence_threshold: float = 0.5
    sky_min_sample_count: int = 1
    sky_geometry_guard_pixels: int = 0
    sky_foreground_decontaminate_pixels: int = 0
    sky_foreground_core_alpha: float = 0.9
    sky_foreground_core_erosion_pixels: int = 0
    sky_foreground_boundary_min_alpha: float = 0.02
    sky_foreground_depth_absolute_tolerance: float = 2.0
    sky_foreground_depth_relative_tolerance: float = 0.1
    sky_foreground_alpha_gamma: float = 1.0
    dynamic_time_mode: str = "blend"
    dynamic_source_camera_index: int | None = None
    dynamic_source_slot: int | None = None
    dynamic_source_nearest_slot: bool = False
    static_opacity_scale: float = 1.0
    canonical_actor_opacity_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.background) != 3 or any(not 0.0 <= x <= 1.0 for x in self.background):
            raise ValueError("background must contain three values in [0, 1]")
        if not 0.0 <= self.depth_alpha_threshold <= 1.0:
            raise ValueError("depth_alpha_threshold must be in [0, 1]")
        if self.sky_edge_padding_pixels < 0:
            raise ValueError("sky_edge_padding_pixels must be non-negative")
        if not 0 <= self.sky_edge_feather_pixels <= self.sky_edge_padding_pixels:
            raise ValueError("sky_edge_feather_pixels must be between zero and sky padding")
        if self.sky_edge_smoothing_iterations < 0:
            raise ValueError("sky_edge_smoothing_iterations must be non-negative")
        if not 0.0 <= self.sky_valid_threshold <= 1.0:
            raise ValueError("sky_valid_threshold must be in [0, 1]")
        if not 0.0 <= self.sky_confidence_threshold <= 1.0:
            raise ValueError("sky_confidence_threshold must be in [0, 1]")
        if self.sky_min_sample_count <= 0:
            raise ValueError("sky_min_sample_count must be positive")
        if self.sky_geometry_guard_pixels < 0:
            raise ValueError("sky_geometry_guard_pixels must be non-negative")
        if self.sky_foreground_decontaminate_pixels < 0:
            raise ValueError("sky_foreground_decontaminate_pixels must be non-negative")
        if not 0.0 < self.sky_foreground_core_alpha <= 1.0:
            raise ValueError("sky_foreground_core_alpha must be in (0, 1]")
        if self.sky_foreground_core_erosion_pixels < 0:
            raise ValueError("sky_foreground_core_erosion_pixels must be non-negative")
        if not 0.0 <= self.sky_foreground_boundary_min_alpha < self.sky_foreground_core_alpha:
            raise ValueError("sky_foreground_boundary_min_alpha must be below the core alpha")
        if self.sky_foreground_depth_absolute_tolerance < 0.0:
            raise ValueError("sky_foreground_depth_absolute_tolerance must be non-negative")
        if self.sky_foreground_depth_relative_tolerance < 0.0:
            raise ValueError("sky_foreground_depth_relative_tolerance must be non-negative")
        if self.sky_foreground_alpha_gamma < 1.0:
            raise ValueError("sky_foreground_alpha_gamma must be at least 1")
        if self.dynamic_time_mode not in DYNAMIC_TIME_MODES:
            raise ValueError(
                f"dynamic_time_mode must be one of {', '.join(DYNAMIC_TIME_MODES)}"
            )
        if self.dynamic_source_camera_index is not None and self.dynamic_source_camera_index < 0:
            raise ValueError("dynamic_source_camera_index must be non-negative")
        if self.dynamic_source_slot is not None and self.dynamic_source_slot < 0:
            raise ValueError("dynamic_source_slot must be non-negative")
        if self.dynamic_source_nearest_slot and self.dynamic_source_camera_index is None:
            raise ValueError("dynamic_source_nearest_slot requires dynamic_source_camera_index")
        if not 0.0 <= self.static_opacity_scale <= 1.0:
            raise ValueError("static_opacity_scale must be in [0, 1]")
        if not 0.0 <= self.canonical_actor_opacity_scale <= 1.0:
            raise ValueError("canonical_actor_opacity_scale must be in [0, 1]")


@dataclass(frozen=True)
class RenderedFrame:
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray | None
    sky_rgb: np.ndarray | None = None
    sky_valid: np.ndarray | None = None
    dynamic_gaussian_count: int = 0


SEQUENCE_OUTPUT_COMPONENTS = frozenset(
    {"rgb", "alpha", "depth", "depth-preview", "sky-debug"}
)


@dataclass(frozen=True)
class SequenceRenderOptions:
    components: tuple[str, ...] = (
        "rgb",
        "alpha",
        "depth",
        "depth-preview",
        "sky-debug",
    )
    resume: bool = False
    overwrite_existing: bool = False
    progress_every: int = 1
    run_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.components).difference(SEQUENCE_OUTPUT_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown output components: {sorted(unknown)}")
        if not self.components:
            raise ValueError("at least one output component is required")
        if len(set(self.components)) != len(self.components):
            raise ValueError("output components must not contain duplicates")
        if self.resume and self.overwrite_existing:
            raise ValueError("resume and overwrite_existing are mutually exclusive")
        if self.progress_every <= 0:
            raise ValueError("progress_every must be positive")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _composite_background(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background: tuple[float, float, float],
) -> np.ndarray:
    """Composite premultiplied RGB over a solid background."""
    background_rgb = np.asarray(background, dtype=np.float32)
    return np.clip(rgb + (1.0 - alpha[..., None]) * background_rgb, 0.0, 1.0)


def _geometry_guard_alpha(alpha: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate geometry transmittance only for RGB sky compositing.

    The saved geometry alpha remains unchanged. This guard suppresses bright
    sky leaking through uncertain 3DGS silhouette pixels whose foreground
    colors were trained against the source sky.
    """
    output = np.asarray(alpha, dtype=np.float32).copy()
    if pixels == 0:
        return output
    height, width = output.shape
    for _ in range(pixels):
        padded = np.pad(output, 1, mode="edge")
        output = np.maximum.reduce(
            [padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)]
        )
    return output


def _geometry_guard_premultiplied(
    rgb: np.ndarray,
    alpha: np.ndarray,
    pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend foreground color and its temporary composite matte together."""
    rgb = np.asarray(rgb, dtype=np.float32)
    output_alpha = np.asarray(alpha, dtype=np.float32).copy()
    if pixels == 0:
        return rgb.copy(), output_alpha
    foreground = rgb / np.maximum(output_alpha[..., None], 1e-5)
    foreground[output_alpha <= 1e-5] = 0.0
    height, width = output_alpha.shape
    for _ in range(pixels):
        padded_alpha = np.pad(output_alpha, 1, mode="edge")
        padded_foreground = np.pad(foreground, ((1, 1), (1, 1), (0, 0)), mode="edge")
        alpha_neighbors = np.stack(
            [padded_alpha[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)],
            axis=-1,
        )
        foreground_neighbors = np.stack(
            [padded_foreground[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)],
            axis=-2,
        )
        choice = np.argmax(alpha_neighbors, axis=-1)
        best_alpha = np.take_along_axis(alpha_neighbors, choice[..., None], axis=-1)[..., 0]
        best_foreground = np.take_along_axis(
            foreground_neighbors, choice[..., None, None], axis=-2
        )[..., 0, :]
        replace = best_alpha > output_alpha
        output_alpha[replace] = best_alpha[replace]
        foreground[replace] = best_foreground[replace]
    output_rgb = np.clip(foreground * output_alpha[..., None], 0.0, 1.0)
    return output_rgb.astype(np.float32), output_alpha


def _decontaminate_foreground_edges(
    premultiplied_rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    pixels: int,
    *,
    core_alpha: float = 0.9,
    core_erosion_pixels: int = 0,
    boundary_min_alpha: float = 0.02,
    depth_absolute_tolerance: float = 2.0,
    depth_relative_tolerance: float = 0.1,
    alpha_gamma: float = 1.0,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate reliable foreground colors into a narrow alpha boundary.

    The rasterizer RGB is premultiplied. Source colors are taken only from
    high-alpha geometry and propagated through pixels that already contain
    geometry at a compatible expected depth. The returned alpha is a temporary
    RGB-compositing matte; callers keep and save the original geometry alpha.
    """
    rgb = np.asarray(premultiplied_rgb, dtype=np.float32)
    geometry_alpha = np.asarray(alpha, dtype=np.float32)
    expected_depth = np.asarray(depth, dtype=np.float32)
    if rgb.shape[:2] != geometry_alpha.shape or expected_depth.shape != geometry_alpha.shape:
        raise ValueError("foreground RGB, alpha, and depth shapes do not match")
    if allowed_mask is not None and np.asarray(allowed_mask).shape != geometry_alpha.shape:
        raise ValueError("foreground decontamination mask shape does not match alpha")
    if pixels < 0:
        raise ValueError("foreground decontamination pixels must be non-negative")
    if not 0.0 < core_alpha <= 1.0:
        raise ValueError("foreground core alpha must be in (0, 1]")
    if core_erosion_pixels < 0:
        raise ValueError("foreground core erosion pixels must be non-negative")
    if not 0.0 <= boundary_min_alpha < core_alpha:
        raise ValueError("foreground boundary alpha must be below the core alpha")
    if depth_absolute_tolerance < 0.0 or depth_relative_tolerance < 0.0:
        raise ValueError("foreground depth tolerances must be non-negative")
    if alpha_gamma < 1.0:
        raise ValueError("foreground alpha gamma must be at least 1")

    output_rgb = rgb.copy()
    composite_alpha = geometry_alpha.copy()
    if pixels == 0:
        return output_rgb, composite_alpha

    foreground = rgb / np.maximum(geometry_alpha[..., None], 1e-5)
    foreground = np.clip(foreground, 0.0, 1.0)
    source_color = foreground.copy()
    source_depth = expected_depth.copy()
    source_alpha = geometry_alpha.copy()
    assigned = (geometry_alpha >= core_alpha) & np.isfinite(expected_depth)
    height, width = geometry_alpha.shape
    for _ in range(core_erosion_pixels):
        padded = np.pad(assigned, 1, constant_values=False)
        assigned = np.logical_and.reduce(
            [padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)]
        )
    boundary = (geometry_alpha >= boundary_min_alpha) & ~assigned
    if allowed_mask is not None:
        boundary &= np.asarray(allowed_mask, dtype=bool)

    for _ in range(pixels):
        if not boundary.any():
            break
        padded_assigned = np.pad(assigned, 1, constant_values=False)
        padded_color = np.pad(source_color, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_depth = np.pad(source_depth, 1, mode="edge")
        padded_alpha = np.pad(source_alpha, 1, mode="edge")
        best_score = np.full((height, width), -np.inf, dtype=np.float32)
        best_color = np.zeros_like(source_color)
        best_depth = np.zeros_like(source_depth)
        best_alpha = np.zeros_like(source_alpha)
        target_depth_valid = np.isfinite(expected_depth)

        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                candidate_assigned = padded_assigned[dy : dy + height, dx : dx + width]
                candidate_depth = padded_depth[dy : dy + height, dx : dx + width]
                candidate_alpha = padded_alpha[dy : dy + height, dx : dx + width]
                depth_limit = depth_absolute_tolerance + depth_relative_tolerance * np.maximum(
                    np.abs(expected_depth), np.abs(candidate_depth)
                )
                depth_compatible = (
                    target_depth_valid
                    & np.isfinite(candidate_depth)
                    & (np.abs(expected_depth - candidate_depth) <= depth_limit)
                )
                usable = boundary & ~assigned & candidate_assigned & depth_compatible
                replace = usable & (candidate_alpha > best_score)
                if not replace.any():
                    continue
                candidate_color = padded_color[dy : dy + height, dx : dx + width]
                best_score[replace] = candidate_alpha[replace]
                best_color[replace] = candidate_color[replace]
                best_depth[replace] = candidate_depth[replace]
                best_alpha[replace] = candidate_alpha[replace]

        newly_assigned = np.isfinite(best_score)
        if not newly_assigned.any():
            break
        assigned[newly_assigned] = True
        source_color[newly_assigned] = best_color[newly_assigned]
        source_depth[newly_assigned] = best_depth[newly_assigned]
        source_alpha[newly_assigned] = best_alpha[newly_assigned]
        output_rgb[newly_assigned] = (
            best_color[newly_assigned] * geometry_alpha[newly_assigned, None]
        )
        if alpha_gamma != 1.0:
            composite_alpha[newly_assigned] = geometry_alpha[newly_assigned] ** alpha_gamma
            output_rgb[newly_assigned] = (
                best_color[newly_assigned] * composite_alpha[newly_assigned, None]
            )

    return np.clip(output_rgb, 0.0, 1.0).astype(np.float32), composite_alpha.astype(np.float32)


def _rolling_expected_depth(
    premultiplied_world: np.ndarray,
    alpha: np.ndarray,
    start_w2c: np.ndarray,
    end_w2c: np.ndarray,
    shutter_type: str,
) -> np.ndarray:
    """Convert alpha-weighted expected world positions to rolling-shutter z depth."""
    height, width = alpha.shape
    expected_world = premultiplied_world / np.maximum(alpha[..., None], 1e-8)
    vertical = shutter_type in {"ROLLING_TOP_TO_BOTTOM", "ROLLING_BOTTOM_TO_TOP"}
    count = height if vertical else width
    times = np.zeros(count, dtype=np.float64) if count == 1 else np.arange(count, dtype=np.float64) / (count - 1)
    if shutter_type in {"ROLLING_BOTTOM_TO_TOP", "ROLLING_RIGHT_TO_LEFT"}:
        times = 1.0 - times
    rotations, translations = interpolate_w2c(start_w2c, end_w2c, times)
    if vertical:
        depth = np.einsum("hwi,hi->hw", expected_world, rotations[:, 2, :]) + translations[:, 2, None]
    else:
        depth = np.einsum("hwi,wi->hw", expected_world, rotations[:, 2, :]) + translations[None, :, 2]
    return depth.astype(np.float32)


class GsplatRenderer:
    def __init__(
        self,
        scene: GaussianScene,
        options: RenderOptions,
        sky_cubemap: SkyCubemap | TemporalSkyCubemap | None = None,
        color_correction_metadata: dict[str, object] | None = None,
        opacity_correction_metadata: dict[str, object] | None = None,
        dynamic_scene: DynamicGaussianScene | None = None,
        dynamic_metadata: dict[str, object] | None = None,
        dynamic_actor_association: DynamicActorAssociation | None = None,
        canonical_actor_asset: CanonicalActorAsset | None = None,
        canonical_actor_metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            import torch
            from gsplat import RollingShutterType, rasterization
            from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType
        except ImportError as exc:  # pragma: no cover - GPU environment-dependent
            raise RuntimeError("Rendering requires PyTorch and gsplat") from exc
        self.torch = torch
        self.rasterization = rasterization
        self.FThetaCameraDistortionParameters = FThetaCameraDistortionParameters
        self.FThetaPolynomialType = FThetaPolynomialType
        self.RollingShutterType = RollingShutterType
        self.options = options
        self.sky_cubemap = sky_cubemap
        self.color_correction_metadata = color_correction_metadata
        self.opacity_correction_metadata = opacity_correction_metadata
        self.dynamic_metadata = dynamic_metadata
        self.dynamic_actor_association = dynamic_actor_association
        self.canonical_actor_asset = canonical_actor_asset
        self.canonical_actor_metadata = canonical_actor_metadata
        device = torch.device(options.device)
        self.means = torch.from_numpy(scene.means).to(device=device)
        self.scales = torch.from_numpy(scene.scales).to(device=device)
        self.quats = torch.from_numpy(scene.quats_wxyz).to(device=device)
        self.opacities = torch.from_numpy(scene.opacities).to(device=device)
        self.colors = torch.from_numpy(scene.sh_coeffs).to(device=device)
        self.sh_degree = scene.sh_degree
        if canonical_actor_asset is not None and dynamic_scene is not None:
            raise ValueError("canonical actor asset replaces --dynamic-gaussians; do not pass both")
        if canonical_actor_asset is None:
            self.canonical_positions = None
        else:
            canonical_actor_asset.validate()
            self.canonical_positions = torch.from_numpy(canonical_actor_asset.positions).to(device=device)
            local_quats = torch.from_numpy(canonical_actor_asset.rotations_wxyz).to(device=device)
            self.canonical_quats = local_quats / torch.linalg.vector_norm(local_quats, dim=-1, keepdim=True)
            self.canonical_scales = torch.from_numpy(canonical_actor_asset.scales).to(device=device)
            canonical_dc = torch.from_numpy(
                ((canonical_actor_asset.rgb - 0.5) / 0.28209479177387814).astype(np.float32)
            ).to(device=device)
            self.canonical_colors = (
                canonical_dc[:, None, :]
                if self.colors.shape[1] == 1
                else torch.cat(
                    [canonical_dc[:, None, :], torch.zeros((len(canonical_dc), self.colors.shape[1] - 1, 3), device=device)],
                    dim=1,
                )
            )
            self.canonical_opacities = torch.from_numpy(canonical_actor_asset.opacities).to(device=device)
            self.canonical_actor_indices = torch.from_numpy(canonical_actor_asset.actor_indices).to(device=device)
        self.dynamic_scene = dynamic_scene
        if dynamic_scene is None:
            self.dynamic_positions = None
            self.dynamic_timestamps_us = None
            self.dynamic_quats = None
            self.dynamic_scales = None
            self.dynamic_colors = None
            self.dynamic_max_opacities = None
            self.dynamic_source_slots = None
            self.dynamic_source_camera_indices = None
            self.dynamic_slot_timestamps_us = None
            self.dynamic_actor_tracks = None
            self.dynamic_actor_slot_available = None
            self.dynamic_actor_slot_support_min_us = None
            self.dynamic_actor_slot_support_max_us = None
        else:
            self.dynamic_positions = torch.from_numpy(dynamic_scene.keyframe_positions).to(device=device)
            self.dynamic_timestamps_us = torch.from_numpy(dynamic_scene.keyframe_timestamps_us).to(device=device)
            dynamic_quats = torch.from_numpy(dynamic_scene.rotations_wxyz).to(device=device)
            self.dynamic_quats = dynamic_quats / torch.linalg.vector_norm(
                dynamic_quats, dim=-1, keepdim=True
            )
            self.dynamic_scales = torch.from_numpy(dynamic_scene.scales).to(device=device)
            dynamic_dc = torch.from_numpy(dynamic_scene.sh_dc).to(device=device)
            if self.colors.shape[1] == 1:
                self.dynamic_colors = dynamic_dc
            else:
                self.dynamic_colors = torch.zeros(
                    (dynamic_scene.count, self.colors.shape[1], 3),
                    dtype=dynamic_dc.dtype,
                    device=device,
                )
                self.dynamic_colors[:, :1] = dynamic_dc
            self.dynamic_max_opacities = torch.from_numpy(dynamic_scene.max_opacities).to(device=device)
            source_slots, slot_timestamps_us = infer_dynamic_source_slots(dynamic_scene)
            self.dynamic_source_slots = torch.from_numpy(source_slots).to(device=device)
            self.dynamic_source_camera_indices = (
                None
                if dynamic_scene.source_camera_indices is None
                else torch.from_numpy(dynamic_scene.source_camera_indices).to(device=device)
            )
            self.dynamic_slot_timestamps_us = torch.from_numpy(slot_timestamps_us).to(device=device)
            if dynamic_actor_association is None:
                self.dynamic_actor_tracks = None
                self.dynamic_actor_slot_available = None
                self.dynamic_actor_slot_support_min_us = None
                self.dynamic_actor_slot_support_max_us = None
            else:
                actor_track_indices = dynamic_actor_association.track_indices
                actor_source_slots = dynamic_actor_association.source_slot_indices
                actor_tracks = torch.from_numpy(actor_track_indices).to(device=device)
                self.dynamic_actor_tracks = actor_tracks
                track_count = len(dynamic_actor_association.tracks)
                support_shape = (track_count, len(slot_timestamps_us))
                availability_np = np.zeros(support_shape, dtype=bool)
                support_min_np = np.full(support_shape, np.iinfo(np.int64).max, dtype=np.int64)
                support_max_np = np.full(support_shape, np.iinfo(np.int64).min, dtype=np.int64)
                for track_index, slot_index in np.unique(
                    np.stack(
                        (
                            actor_track_indices[actor_track_indices >= 0],
                            actor_source_slots[actor_track_indices >= 0],
                        ),
                        axis=1,
                    ),
                    axis=0,
                ):
                    members = (actor_track_indices == track_index) & (
                        actor_source_slots == slot_index
                    )
                    availability_np[track_index, slot_index] = True
                    support_min_np[track_index, slot_index] = int(
                        dynamic_scene.keyframe_timestamps_us[members, 0].min()
                    )
                    support_max_np[track_index, slot_index] = int(
                        dynamic_scene.keyframe_timestamps_us[members, 2].max()
                    )
                self.dynamic_actor_slot_available = torch.from_numpy(availability_np).to(
                    device=device
                )
                self.dynamic_actor_slot_support_min_us = torch.from_numpy(support_min_np).to(
                    device=device
                )
                self.dynamic_actor_slot_support_max_us = torch.from_numpy(support_max_np).to(
                    device=device
                )

    @staticmethod
    def _camera_dynamic_timestamp_us(camera: CameraFrame) -> int:
        if camera.timestamp_us is None:
            raise ValueError("dynamic rendering requires camera timestamps")
        if camera.timestamp_start_us is None:
            return int(camera.timestamp_us)
        return int(camera.timestamp_start_us + (camera.timestamp_us - camera.timestamp_start_us) // 2)

    def _gaussians_at_camera_time(self, camera: CameraFrame):
        torch = self.torch
        # This is intentionally a render-time multiplier instead of a PLY
        # mutation.  It supports reversible diagnostic decompositions such as
        # actor-only rendering while leaving the static source asset intact.
        static_opacities = self.opacities * self.options.static_opacity_scale
        if getattr(self, "canonical_positions", None) is not None:
            target_us = self._camera_dynamic_timestamp_us(camera)
            asset = self.canonical_actor_asset
            # A MASt3R metric-depth surface may have evidence for only one
            # NCore reference time.  Such an asset is intentionally *not* a
            # canonical, all-time actor: enabling it at other frames would
            # turn an observed patch into the very temporal ghost we are
            # trying to avoid.  Keep this policy in asset metadata so older
            # canonical actors preserve their existing behaviour.
            canonical_visible = canonical_visibility_allows_frame(asset.metadata, camera.frame_id)
            world_positions = torch.empty_like(self.canonical_positions)
            world_quats = torch.empty_like(self.canonical_quats)
            for actor_index, trajectory in enumerate(asset.trajectories):
                members = self.canonical_actor_indices == actor_index
                if not bool(members.any()):
                    continue
                pose = _interpolate_pose(trajectory, target_us)
                rotation = torch.from_numpy(pose[:3, :3]).to(device=self.means.device, dtype=self.means.dtype)
                translation = torch.from_numpy(pose[:3, 3]).to(device=self.means.device, dtype=self.means.dtype)
                world_positions[members] = self.canonical_positions[members] @ rotation.T + translation
                # wxyz Hamilton product: q_actor * q_local.
                local = self.canonical_quats[members]
                q_actor = torch.empty((4,), dtype=local.dtype, device=local.device)
                trace = rotation.trace()
                if float(trace) > 0.0:
                    scale = torch.sqrt(trace + 1.0) * 2.0
                    q_actor[:] = torch.stack((0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale))
                else:
                    # Trace-negative poses are uncommon for road tracks, but
                    # use the numerically stable matrix-to-quaternion branch.
                    diagonal = torch.diagonal(rotation)
                    axis = int(torch.argmax(diagonal))
                    if axis == 0:
                        scale = torch.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                        q_actor[:] = torch.stack(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale))
                    elif axis == 1:
                        scale = torch.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                        q_actor[:] = torch.stack(((rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale))
                    else:
                        scale = torch.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                        q_actor[:] = torch.stack(((rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
                q_actor = q_actor / torch.linalg.vector_norm(q_actor)
                aw, ax, ay, az = q_actor
                bw, bx, by, bz = local.unbind(dim=-1)
                world_quats[members] = torch.stack((
                    aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                ), dim=-1)
            return (
                torch.cat([self.means, world_positions], dim=0),
                torch.cat([self.quats, world_quats], dim=0),
                torch.cat([self.scales, self.canonical_scales], dim=0),
                torch.cat(
                    [
                        static_opacities,
                        self.canonical_opacities * self.options.canonical_actor_opacity_scale * float(canonical_visible),
                    ],
                    dim=0,
                ),
                torch.cat([self.colors, self.canonical_colors], dim=0),
                int(self.canonical_positions.shape[0]),
            )
        if getattr(self, "dynamic_positions", None) is None:
            return self.means, self.quats, self.scales, static_opacities, self.colors, 0
        target_us = self._camera_dynamic_timestamp_us(camera)
        times = self.dynamic_timestamps_us
        target = torch.as_tensor(target_us, dtype=torch.float64, device=times.device)
        times_float = times.to(dtype=torch.float64)
        left = target <= times_float[:, 1]
        left_u = (target - times_float[:, 0]) / (times_float[:, 1] - times_float[:, 0])
        right_u = (target - times_float[:, 1]) / (times_float[:, 2] - times_float[:, 1])
        u = torch.where(left, left_u, right_u).to(dtype=self.dynamic_positions.dtype)
        positions = torch.where(
            left[:, None],
            self.dynamic_positions[:, 0]
            + u[:, None] * (self.dynamic_positions[:, 1] - self.dynamic_positions[:, 0]),
            self.dynamic_positions[:, 1]
            + u[:, None] * (self.dynamic_positions[:, 2] - self.dynamic_positions[:, 1]),
        )
        if self.options.dynamic_time_mode == "nearest":
            selected_slot = torch.argmin(
                torch.abs(self.dynamic_slot_timestamps_us.to(dtype=torch.float64) - target)
            )
            selected = self.dynamic_source_slots == selected_slot
            in_support = (target >= times_float[selected, 0].min()) & (
                target <= times_float[selected, 2].max()
            )
            temporal_weight = (selected & in_support).to(
                dtype=self.dynamic_max_opacities.dtype
            )
        elif self.options.dynamic_time_mode == "actor-nearest":
            if self.dynamic_actor_tracks is None or self.dynamic_actor_slot_available is None:
                raise ValueError("actor-nearest mode requires a dynamic actor association")
            temporal_weight = torch.where(left, left_u, 1.0 - right_u).clamp(0.0, 1.0)
            associated = self.dynamic_actor_tracks >= 0
            if bool(associated.any()):
                supported_slots = (
                    self.dynamic_actor_slot_available
                    & (target >= self.dynamic_actor_slot_support_min_us.to(dtype=torch.float64))
                    & (target <= self.dynamic_actor_slot_support_max_us.to(dtype=torch.float64))
                )
                distances = torch.abs(
                    self.dynamic_slot_timestamps_us.to(dtype=torch.float64)[None, :] - target
                )
                distances = distances.expand_as(supported_slots).clone()
                distances[~supported_slots] = torch.inf
                selected_slots = torch.argmin(distances, dim=1)
                actor_has_support = supported_slots.any(dim=1)
                associated_supported = associated & actor_has_support[
                    self.dynamic_actor_tracks.clamp_min(0).to(dtype=torch.long)
                ]
                selected = associated_supported & (
                    self.dynamic_source_slots
                    == selected_slots[self.dynamic_actor_tracks.clamp_min(0).to(dtype=torch.long)]
                )
                support = selected & (target >= times_float[:, 0]) & (target <= times_float[:, 2])
                temporal_weight = torch.where(
                    associated_supported,
                    support.to(dtype=temporal_weight.dtype),
                    temporal_weight,
                )
        else:
            temporal_weight = torch.where(left, left_u, 1.0 - right_u).clamp(0.0, 1.0)
        if self.options.dynamic_source_nearest_slot:
            if self.dynamic_source_camera_indices is None:
                raise ValueError("dynamic_source_nearest_slot requires v2 source camera provenance")
            source_camera_mask = self.dynamic_source_camera_indices == self.options.dynamic_source_camera_index
            available_slots = torch.unique(self.dynamic_source_slots[source_camera_mask])
            if int(available_slots.numel()) == 0:
                raise ValueError(
                    f"dynamic source-camera index {self.options.dynamic_source_camera_index} has no Gaussians"
                )
            offsets = torch.abs(
                self.dynamic_slot_timestamps_us[available_slots].to(dtype=torch.float64) - target
            )
            selected_slot = available_slots[torch.argmin(offsets)]
            temporal_weight = temporal_weight * (self.dynamic_source_slots == selected_slot).to(
                dtype=temporal_weight.dtype
            )
        opacities = self.dynamic_max_opacities * temporal_weight.to(
            dtype=self.dynamic_max_opacities.dtype
        )
        active = opacities > 1.0e-4
        if self.options.dynamic_source_camera_index is not None:
            if self.dynamic_source_camera_indices is None:
                raise ValueError("dynamic_source_camera_index requires v2 source camera provenance")
            active &= self.dynamic_source_camera_indices == self.options.dynamic_source_camera_index
        if self.options.dynamic_source_slot is not None:
            active &= self.dynamic_source_slots == self.options.dynamic_source_slot
        dynamic_count = int(active.sum().item())
        if dynamic_count == 0:
            return self.means, self.quats, self.scales, static_opacities, self.colors, 0
        return (
            torch.cat([self.means, positions[active]], dim=0),
            torch.cat([self.quats, self.dynamic_quats[active]], dim=0),
            torch.cat([self.scales, self.dynamic_scales[active]], dim=0),
            torch.cat([static_opacities, opacities[active]], dim=0),
            torch.cat([self.colors, self.dynamic_colors[active]], dim=0),
            dynamic_count,
        )

    def render(self, camera: CameraFrame, *, include_depth: bool = True) -> RenderedFrame:
        torch = self.torch
        intrinsics = camera.intrinsics
        means, quats, scales, opacities, colors, dynamic_count = self._gaussians_at_camera_time(camera)
        viewmat = torch.from_numpy(camera.world_to_camera).to(self.means.device)[None]
        K = torch.from_numpy(intrinsics.K).to(self.means.device)[None]
        camera_kwargs: dict[str, object] = {"camera_model": camera.camera_model}
        packed = self.options.packed
        if camera.ftheta is not None:
            ftheta = camera.ftheta
            camera_kwargs.update(
                with_ut=True,
                ftheta_coeffs=self.FThetaCameraDistortionParameters(
                    reference_poly=self.FThetaPolynomialType[ftheta.reference_poly],
                    pixeldist_to_angle_poly=ftheta.pixeldist_to_angle_poly,
                    angle_to_pixeldist_poly=ftheta.angle_to_pixeldist_poly,
                    max_angle=ftheta.max_angle,
                    linear_cde=ftheta.linear_cde,
                ),
                rolling_shutter=self.RollingShutterType[ftheta.shutter_type],
            )
            # gsplat 1.5.3 requires unpacked tensors for the unscented
            # transform used by distortion and rolling shutter.
            packed = False
            if ftheta.is_rolling:
                camera_kwargs["viewmats_rs"] = torch.from_numpy(camera.world_to_camera_end).to(
                    self.means.device
                )[None]
        with torch.inference_mode():
            common_kwargs = dict(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                viewmats=viewmat,
                Ks=K,
                width=intrinsics.width,
                height=intrinsics.height,
                near_plane=self.options.near_plane,
                far_plane=self.options.far_plane,
                packed=packed,
                # gsplat 1.5.3 has conflicting background shape requirements
                # between RGB+ED and packed rasterization. Composite below
                # using the returned alpha instead.
                backgrounds=None,
                rasterize_mode=self.options.rasterize_mode,
            )
            need_depth = include_depth or self.options.sky_foreground_decontaminate_pixels > 0
            if camera.ftheta is not None and camera.ftheta.is_rolling:
                # gsplat 1.5.3's 2D-Gaussian rolling-shutter rasterizer
                # produces invalid, full-frame footprints even when the start
                # and end poses are identical. eval3d evaluates each Gaussian
                # along the actual per-pixel rolling-shutter ray and avoids
                # that failure. eval3d is RGB-only, so a second pass renders
                # expected world XYZ for exact camera-z expected depth.
                rendered, alpha, _ = self.rasterization(
                    colors=colors,
                    sh_degree=self.sh_degree,
                    render_mode="RGB",
                    with_eval3d=True,
                    **common_kwargs,
                    **camera_kwargs,
                )
                # Some gsplat eval3d builds reuse their output buffers on the
                # next invocation. Preserve RGB/alpha before the XYZ pass.
                rendered = rendered.clone()
                alpha = alpha.clone()
                if need_depth:
                    world_rendered, _, _ = self.rasterization(
                        colors=means,
                        sh_degree=None,
                        render_mode="RGB",
                        with_eval3d=True,
                        **common_kwargs,
                        **camera_kwargs,
                    )
                else:
                    world_rendered = None
            else:
                rendered, alpha, _ = self.rasterization(
                    colors=colors,
                    sh_degree=self.sh_degree,
                    render_mode="RGB+ED" if need_depth else "RGB",
                    **common_kwargs,
                    **camera_kwargs,
                )
                world_rendered = None
        rgb = rendered[0, ..., :3].float().cpu().numpy()
        alpha_np = alpha[0, ..., 0].float().clamp(0.0, 1.0).cpu().numpy()
        if not need_depth:
            depth = None
        elif world_rendered is None:
            depth = rendered[0, ..., 3].float().cpu().numpy()
        else:
            premultiplied_world = world_rendered[0].float().cpu().numpy()
            depth = _rolling_expected_depth(
                premultiplied_world,
                alpha_np,
                camera.world_to_camera,
                camera.world_to_camera_end,
                camera.ftheta.shutter_type,
            )
        sky_rgb: np.ndarray | None = None
        sky_valid: np.ndarray | None = None
        sky_cubemap = getattr(self, "sky_cubemap", None)
        if sky_cubemap is None:
            rgb = _composite_background(rgb, alpha_np, self.options.background)
        else:
            sky_rgb, sky_valid = sample_sky_asset(
                sky_cubemap,
                camera,
                confidence_threshold=self.options.sky_confidence_threshold,
                min_sample_count=self.options.sky_min_sample_count,
            )
            sky_rgb, sky_valid = pad_sky_edges(
                sky_rgb,
                sky_valid,
                self.options.sky_edge_padding_pixels,
                valid_threshold=self.options.sky_valid_threshold,
                feather_pixels=self.options.sky_edge_feather_pixels,
                smoothing_iterations=self.options.sky_edge_smoothing_iterations,
            )
            if self.options.sky_foreground_decontaminate_pixels > 0:
                if depth is None:
                    raise RuntimeError("foreground edge decontamination requires depth")
                composite_rgb, composite_alpha = _decontaminate_foreground_edges(
                    rgb,
                    alpha_np,
                    depth,
                    self.options.sky_foreground_decontaminate_pixels,
                    core_alpha=self.options.sky_foreground_core_alpha,
                    core_erosion_pixels=self.options.sky_foreground_core_erosion_pixels,
                    boundary_min_alpha=self.options.sky_foreground_boundary_min_alpha,
                    depth_absolute_tolerance=self.options.sky_foreground_depth_absolute_tolerance,
                    depth_relative_tolerance=self.options.sky_foreground_depth_relative_tolerance,
                    alpha_gamma=self.options.sky_foreground_alpha_gamma,
                    allowed_mask=sky_valid >= self.options.sky_valid_threshold,
                )
            else:
                composite_rgb, composite_alpha = rgb, alpha_np
            composite_rgb, composite_alpha = _geometry_guard_premultiplied(
                composite_rgb, composite_alpha, self.options.sky_geometry_guard_pixels
            )
            rgb = composite_sky(composite_rgb, composite_alpha, sky_rgb, sky_valid, self.options.background)
        if depth is not None:
            depth = np.where(
                alpha_np >= self.options.depth_alpha_threshold, depth, np.nan
            ).astype(np.float32)
        return RenderedFrame(
            rgb=rgb,
            alpha=alpha_np,
            depth=depth,
            sky_rgb=sky_rgb,
            sky_valid=sky_valid,
            dynamic_gaussian_count=dynamic_count,
        )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _save_png_atomic(array: np.ndarray, mode: str, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        Image.fromarray(array, mode=mode).save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_npy_atomic(array: np.ndarray, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(value: object, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_depth_preview(depth: np.ndarray, path: Path) -> tuple[float | None, float | None]:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        _save_png_atomic(np.zeros(depth.shape, dtype=np.uint16), "I;16", path)
        return None, None
    lo, hi = np.percentile(valid, [1.0, 99.0]).astype(float)
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.nan_to_num((depth - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
    preview = np.clip(normalized, 0.0, 1.0)
    _save_png_atomic(np.round(preview * 65535.0).astype(np.uint16), "I;16", path)
    return lo, hi


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _camera_manifest(camera: CameraFrame) -> dict[str, object]:
    intrinsics = camera.intrinsics
    return {
        "frame_id": camera.frame_id,
        "timestamp_us": camera.timestamp_us,
        "timestamp_start_us": camera.timestamp_start_us,
        "camera_model": camera.camera_model,
        "shutter_type": None if camera.ftheta is None else camera.ftheta.shutter_type,
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "ftheta": None if camera.ftheta is None else asdict(camera.ftheta),
        "world_to_camera_start": camera.world_to_camera.tolist(),
        "world_to_camera_end": None
        if camera.world_to_camera_end is None
        else camera.world_to_camera_end.tolist(),
    }


def _sequence_fingerprint(
    renderer: GsplatRenderer,
    camera_path: CameraPath,
    options: SequenceRenderOptions,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {
        "pose_source": camera_path.source,
        "path_metadata": camera_path.metadata,
        "frames": [_camera_manifest(camera) for camera in camera_path.frames],
        "render_options": asdict(renderer.options),
        "components": list(options.components),
        "run_metadata": options.run_metadata,
        "gaussian_color_correction": getattr(renderer, "color_correction_metadata", None),
        "gaussian_opacity_correction": getattr(renderer, "opacity_correction_metadata", None),
        "dynamic_gaussians": getattr(renderer, "dynamic_metadata", None),
        "canonical_actors": getattr(renderer, "canonical_actor_metadata", None),
        "sky_cubemap": None
        if renderer.sky_cubemap is None
        else {
            "metadata": renderer.sky_cubemap.metadata,
            "face_size": renderer.sky_cubemap.face_size,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _frame_outputs_complete(
    output_dir: Path,
    record: dict[str, object],
    components: set[str],
    has_sky: bool,
) -> bool:
    paths = record.get("outputs", {})
    if not isinstance(paths, dict):
        return False
    component_keys = {
        "rgb": {"rgb"},
        "alpha": {"alpha"},
        "depth": {"depth_npy"},
        "depth-preview": {"depth_preview"},
        "sky-debug": {"sky_rgb", "sky_valid"} if has_sky else set(),
    }
    required = set().union(*(component_keys[value] for value in components))
    if not required.issubset(paths):
        return False
    for value in paths.values():
        if value is None:
            continue
        path = output_dir / str(value)
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return bool(record.get("complete"))


def _resume_manifest_compatible(
    prior: dict[str, object],
    current: dict[str, object],
    camera_path: CameraPath,
) -> bool:
    """Strict semantic fallback for fingerprints made by another process.

    NCore pose evaluation and third-party metadata may serialize equivalent
    floating values differently. Resume remains safe when all input hashes,
    render parameters, sequence selection metadata, and already completed
    camera records match exactly as parsed JSON values.
    """
    def equivalent(left: object, right: object) -> bool:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right, sort_keys=True, separators=(",", ":")
        )

    for key in (
        "schema_version",
        "pose_source",
        "path_metadata",
        "run_metadata",
        "render_options",
        "output_components",
        "frame_count",
        "gaussian_color_correction",
        "gaussian_opacity_correction",
        "dynamic_gaussians",
        "sky_cubemap",
    ):
        prior_value = prior.get(key)
        current_value = current.get(key)
        if key == "path_metadata":
            prior_value = json.loads(json.dumps(prior_value))
            current_value = json.loads(json.dumps(current_value))
            for value in (prior_value, current_value):
                if isinstance(value, dict):
                    chunk = value.get("chunk")
                    if isinstance(chunk, dict) and isinstance(chunk.get("camera_ids"), list):
                        chunk["camera_ids"] = sorted(chunk["camera_ids"])
        if not equivalent(prior_value, current_value):
            return False
    for record in prior.get("frames", []):
        if not isinstance(record, dict):
            return False
        index = int(record.get("index", -1))
        if not 0 <= index < len(camera_path.frames):
            return False
        expected = _camera_manifest(camera_path.frames[index])
        for key, value in expected.items():
            if key in {"world_to_camera_start", "world_to_camera_end"}:
                actual = record.get(key)
                if actual is None or value is None:
                    if actual is not value:
                        return False
                elif not np.allclose(actual, value, atol=1e-6, rtol=0.0):
                    return False
            elif not equivalent(record.get(key), value):
                return False
    return True


def render_camera_path(
    renderer: GsplatRenderer,
    camera_path: CameraPath,
    output_dir: str | Path,
    sequence_options: SequenceRenderOptions | None = None,
) -> None:
    output_dir = Path(output_dir)
    sequence_options = SequenceRenderOptions() if sequence_options is None else sequence_options
    components = set(sequence_options.components)
    manifest_path = output_dir / "manifest.json"
    partial_path = output_dir / "manifest.partial.json"
    existing_nonempty = output_dir.exists() and any(output_dir.iterdir())
    if existing_nonempty and not sequence_options.resume and not sequence_options.overwrite_existing:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --resume or --overwrite-existing"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for component, directory_name in (
        ("rgb", "rgb"),
        ("alpha", "alpha"),
        ("depth", "depth"),
        ("depth-preview", "depth"),
        ("sky-debug", "sky_rgb"),
        ("sky-debug", "sky_valid"),
    ):
        if component in components:
            (output_dir / directory_name).mkdir(parents=True, exist_ok=True)

    sky_cubemap = getattr(renderer, "sky_cubemap", None)
    temporal_sky = isinstance(sky_cubemap, TemporalSkyCubemap)
    fingerprint, fingerprint_payload = _sequence_fingerprint(
        renderer, camera_path, sequence_options
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "in_progress",
        "run_fingerprint": fingerprint,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "pose_source": camera_path.source,
        "path_metadata": camera_path.metadata,
        "run_metadata": sequence_options.run_metadata,
        "render_options": asdict(renderer.options),
        "output_components": list(sequence_options.components),
        "frame_count": len(camera_path.frames),
        "completed_frame_count": 0,
        "gaussian_color_correction": getattr(renderer, "color_correction_metadata", None),
        "gaussian_opacity_correction": getattr(renderer, "opacity_correction_metadata", None),
        "dynamic_gaussians": getattr(renderer, "dynamic_metadata", None),
        "sky_cubemap": None
        if sky_cubemap is None
        else {
            "sequence_id": sky_cubemap.metadata.get("sequence_id"),
            "chunk_index": sky_cubemap.metadata.get("chunk_index"),
            "face_size": sky_cubemap.face_size,
            "temporal": temporal_sky,
            "temporal_manifest": str(sky_cubemap.manifest_path)
            if temporal_sky
            else None,
            "slot_timestamps_us": list(sky_cubemap.timestamps_us)
            if temporal_sky
            else None,
            "total_coverage": sky_cubemap.total_coverage
            if temporal_sky
            else float(np.asarray(sky_cubemap.valid, dtype=bool).mean()),
            "edge_padding_pixels": renderer.options.sky_edge_padding_pixels,
            "edge_feather_pixels": renderer.options.sky_edge_feather_pixels,
            "edge_smoothing_iterations": renderer.options.sky_edge_smoothing_iterations,
            "valid_threshold": renderer.options.sky_valid_threshold,
            "confidence_threshold": renderer.options.sky_confidence_threshold,
            "min_sample_count": renderer.options.sky_min_sample_count,
            "geometry_guard_pixels": renderer.options.sky_geometry_guard_pixels,
            "foreground_decontaminate_pixels": renderer.options.sky_foreground_decontaminate_pixels,
            "foreground_core_alpha": renderer.options.sky_foreground_core_alpha,
            "foreground_core_erosion_pixels": renderer.options.sky_foreground_core_erosion_pixels,
            "foreground_boundary_min_alpha": renderer.options.sky_foreground_boundary_min_alpha,
            "foreground_depth_absolute_tolerance": renderer.options.sky_foreground_depth_absolute_tolerance,
            "foreground_depth_relative_tolerance": renderer.options.sky_foreground_depth_relative_tolerance,
            "foreground_alpha_gamma": renderer.options.sky_foreground_alpha_gamma,
        },
        "frames": [],
    }
    prior_records: dict[tuple[int, str], dict[str, object]] = {}
    if sequence_options.resume:
        resume_path = partial_path if partial_path.is_file() else manifest_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"no partial or complete manifest to resume in {output_dir}")
        with resume_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("run_fingerprint") != fingerprint and not _resume_manifest_compatible(
            prior, manifest, camera_path
        ):
            raise ValueError(
                "resume fingerprint mismatch; camera path, inputs, render parameters, or outputs changed: "
                f"existing={prior.get('run_fingerprint')} current={fingerprint}"
            )
        if prior.get("run_fingerprint") != fingerprint:
            print(
                "Resume fingerprint bytes differ, but strict manifest compatibility passed; continuing.",
                flush=True,
            )
        manifest["created_at"] = prior.get("created_at", manifest["created_at"])
        for record in prior.get("frames", []):
            if isinstance(record, dict):
                prior_records[(int(record["index"]), str(record["frame_id"]))] = record

    needs_depth = bool({"depth", "depth-preview"}.intersection(components))
    started = time.perf_counter()
    rendered_count = 0
    skipped_count = 0
    for index, camera in enumerate(camera_path.frames):
        name = f"{index:06d}_{_safe_name(camera.frame_id)}"
        key = (index, camera.frame_id)
        prior_record = prior_records.get(key)
        if prior_record is not None and _frame_outputs_complete(
            output_dir, prior_record, components, sky_cubemap is not None
        ):
            manifest["frames"].append(prior_record)
            skipped_count += 1
            continue
        frame_started = time.perf_counter()
        rendered = renderer.render(camera, include_depth=needs_depth)
        outputs: dict[str, str | None] = {}
        if "rgb" in components:
            path = output_dir / "rgb" / f"{name}.png"
            _save_png_atomic(np.round(rendered.rgb * 255.0).astype(np.uint8), "RGB", path)
            outputs["rgb"] = f"rgb/{name}.png"
        if "alpha" in components:
            path = output_dir / "alpha" / f"{name}.png"
            _save_png_atomic(np.round(rendered.alpha * 255.0).astype(np.uint8), "L", path)
            outputs["alpha"] = f"alpha/{name}.png"
        depth_lo: float | None = None
        depth_hi: float | None = None
        if "depth" in components:
            if rendered.depth is None:
                raise RuntimeError("renderer did not return requested depth")
            path = output_dir / "depth" / f"{name}.npy"
            _save_npy_atomic(rendered.depth, path)
            outputs["depth_npy"] = f"depth/{name}.npy"
        if "depth-preview" in components:
            if rendered.depth is None:
                raise RuntimeError("renderer did not return requested depth preview input")
            path = output_dir / "depth" / f"{name}.png"
            depth_lo, depth_hi = _save_depth_preview(rendered.depth, path)
            outputs["depth_preview"] = f"depth/{name}.png"
        sky_rgb_path: str | None = None
        sky_valid_path: str | None = None
        if (
            "sky-debug" in components
            and rendered.sky_rgb is not None
            and rendered.sky_valid is not None
        ):
            sky_rgb_dir = output_dir / "sky_rgb"
            sky_valid_dir = output_dir / "sky_valid"
            _save_png_atomic(
                np.round(rendered.sky_rgb * 255.0).astype(np.uint8),
                "RGB",
                sky_rgb_dir / f"{name}.png",
            )
            _save_png_atomic(
                np.round(rendered.sky_valid * 255.0).astype(np.uint8),
                "L",
                sky_valid_dir / f"{name}.png",
            )
            sky_rgb_path = f"sky_rgb/{name}.png"
            sky_valid_path = f"sky_valid/{name}.png"
            outputs["sky_rgb"] = sky_rgb_path
            outputs["sky_valid"] = sky_valid_path
        frame_seconds = time.perf_counter() - frame_started
        record = {
                "index": index,
                **_camera_manifest(camera),
                "outputs": outputs,
                "depth_preview_percentiles_m": [depth_lo, depth_hi],
                "render_seconds": frame_seconds,
                "active_dynamic_gaussians": rendered.dynamic_gaussian_count,
                "complete": True,
            }
        manifest["frames"].append(record)
        rendered_count += 1
        manifest["completed_frame_count"] = len(manifest["frames"])
        manifest["updated_at"] = _utc_now()
        _write_json_atomic(manifest, partial_path)
        completed = len(manifest["frames"])
        if completed % sequence_options.progress_every == 0 or completed == len(camera_path.frames):
            elapsed = time.perf_counter() - started
            average = elapsed / max(rendered_count, 1)
            remaining = len(camera_path.frames) - completed
            memory = ""
            if renderer.means.is_cuda:
                allocated = renderer.torch.cuda.max_memory_allocated(renderer.means.device) / (1024**3)
                memory = f" gpu_peak={allocated:.2f}GiB"
            print(
                f"[{completed:04d}/{len(camera_path.frames):04d}] "
                f"source={camera.frame_id} frame={frame_seconds:.2f}s "
                f"average={average:.2f}s eta={average * remaining:.1f}s{memory}",
                flush=True,
            )

    manifest["status"] = "complete"
    manifest["completed_frame_count"] = len(manifest["frames"])
    manifest["rendered_this_run"] = rendered_count
    manifest["resumed_frames"] = skipped_count
    manifest["updated_at"] = _utc_now()
    manifest["fingerprint_payload"] = fingerprint_payload
    _write_json_atomic(manifest, partial_path)
    partial_path.replace(manifest_path)
