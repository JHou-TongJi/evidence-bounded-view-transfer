from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814
DYNAMIC_TIME_MODES = ("blend", "nearest", "actor-nearest")
# InstantNuRec samples 18 sparse source frames per chunk.  A source frame's
# rolling-shutter timestamps span only tens of milliseconds, while adjacent
# sampled frames are hundreds of milliseconds apart.  The v1 asset does not
# store a source-frame index, so recover it from those separated timestamp
# clusters.  Future asset versions should store the index explicitly.
SOURCE_SLOT_GAP_US = 100_000


@dataclass(frozen=True)
class DynamicGaussianScene:
    """Three-keyframe dynamic Gaussians in NCore world coordinates."""

    keyframe_positions: np.ndarray       # [N, 3, 3]
    keyframe_timestamps_us: np.ndarray   # [N, 3]
    rotations_wxyz: np.ndarray           # [N, 4]
    scales: np.ndarray                    # [N, 3]
    rgb: np.ndarray                       # [N, 3], activated sRGB
    max_opacities: np.ndarray             # [N], activated
    metadata: dict[str, object]
    # Version 2 adds the observation provenance that InstantNuRec had while
    # constructing the dense layer.  They remain optional so a pre-existing
    # v1 asset is still a valid renderer input.
    source_track_ids: np.ndarray | None = None       # [N], NCore numeric ID or -1
    source_camera_indices: np.ndarray | None = None  # [N], NCore sensor index
    source_frame_slots: np.ndarray | None = None     # [N], source temporal slot
    association_sources: np.ndarray | None = None    # [N], 0 none / 1 point / 2 ray
    actor_local_positions: np.ndarray | None = None  # [N, 3], source cuboid frame

    @property
    def count(self) -> int:
        return int(self.keyframe_positions.shape[0])

    @property
    def sh_dc(self) -> np.ndarray:
        return ((self.rgb - 0.5) / SH_C0).astype(np.float32)[:, None, :]


def load_dynamic_gaussians(path: str | Path) -> DynamicGaussianScene:
    path = Path(path)
    required = {
        "keyframe_positions",
        "keyframe_timestamps_us",
        "rotations_wxyz",
        "scales",
        "rgb",
        "max_opacities",
        "metadata",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing dynamic Gaussian fields: {', '.join(missing)}")
        metadata = json.loads(str(data["metadata"].item()))
        version = int(metadata.get("version", -1))
        optional: dict[str, np.ndarray | None] = {
            "source_track_ids": None,
            "source_camera_indices": None,
            "source_frame_slots": None,
            "association_sources": None,
            "actor_local_positions": None,
        }
        if version >= 2:
            missing_v2 = sorted(name for name in optional if name not in data.files)
            if missing_v2:
                raise ValueError(f"{path} is a v2 dynamic asset missing fields: {', '.join(missing_v2)}")
            optional = {
                "source_track_ids": np.asarray(data["source_track_ids"], dtype=np.int64),
                "source_camera_indices": np.asarray(data["source_camera_indices"], dtype=np.int16),
                "source_frame_slots": np.asarray(data["source_frame_slots"], dtype=np.int16),
                "association_sources": np.asarray(data["association_sources"], dtype=np.uint8),
                "actor_local_positions": np.asarray(data["actor_local_positions"], dtype=np.float32),
            }
        scene = DynamicGaussianScene(
            keyframe_positions=np.asarray(data["keyframe_positions"], dtype=np.float32),
            keyframe_timestamps_us=np.asarray(data["keyframe_timestamps_us"], dtype=np.int64),
            rotations_wxyz=np.asarray(data["rotations_wxyz"], dtype=np.float32),
            scales=np.asarray(data["scales"], dtype=np.float32),
            rgb=np.asarray(data["rgb"], dtype=np.float32),
            max_opacities=np.asarray(data["max_opacities"], dtype=np.float32),
            metadata=metadata,
            **optional,
        )
    _validate_dynamic_scene(scene, path)
    return scene


def _validate_dynamic_scene(scene: DynamicGaussianScene, path: Path | str = "dynamic asset") -> None:
    count = scene.count
    expected = {
        "keyframe_positions": (count, 3, 3),
        "keyframe_timestamps_us": (count, 3),
        "rotations_wxyz": (count, 4),
        "scales": (count, 3),
        "rgb": (count, 3),
        "max_opacities": (count,),
    }
    for name, shape in expected.items():
        if getattr(scene, name).shape != shape:
            raise ValueError(f"{path} field {name} must have shape {shape}")
    if scene.metadata.get("schema") != "instant-nurec-dynamic-gaussians":
        raise ValueError(f"{path} has an unsupported dynamic Gaussian schema")
    version = int(scene.metadata.get("version", -1))
    if version not in (1, 2):
        raise ValueError(f"{path} has an unsupported dynamic Gaussian version")
    if scene.metadata.get("coordinate_frame") != "ncore_world":
        raise ValueError(f"{path} must use the ncore_world coordinate frame")
    arrays = (
        scene.keyframe_positions,
        scene.rotations_wxyz,
        scene.scales,
        scene.rgb,
        scene.max_opacities,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{path} contains non-finite dynamic Gaussian values")
    if count and np.any(np.diff(scene.keyframe_timestamps_us, axis=1) <= 0):
        raise ValueError(f"{path} keyframe timestamps must be strictly increasing")
    if np.any(scene.scales <= 0.0):
        raise ValueError(f"{path} contains non-positive dynamic Gaussian scales")
    if np.any((scene.max_opacities < 0.0) | (scene.max_opacities > 1.0)):
        raise ValueError(f"{path} dynamic Gaussian opacities must be in [0, 1]")
    if np.any((scene.rgb < 0.0) | (scene.rgb > 1.0)):
        raise ValueError(f"{path} dynamic Gaussian RGB must be in [0, 1]")
    if count:
        norms = np.linalg.norm(scene.rotations_wxyz, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError(f"{path} contains zero-length dynamic Gaussian quaternions")
    v2_fields = (
        "source_track_ids",
        "source_camera_indices",
        "source_frame_slots",
        "association_sources",
        "actor_local_positions",
    )
    if version == 2:
        if any(getattr(scene, name) is None for name in v2_fields):
            raise ValueError(f"{path} v2 provenance fields are incomplete")
        for name in v2_fields[:-1]:
            if getattr(scene, name).shape != (count,):
                raise ValueError(f"{path} field {name} must have shape ({count},)")
        if scene.actor_local_positions.shape != (count, 3):
            raise ValueError(f"{path} field actor_local_positions must have shape ({count}, 3)")
        if np.any((scene.association_sources < 0) | (scene.association_sources > 2)):
            raise ValueError(f"{path} has invalid association_sources; expected 0, 1 or 2")
        if not np.all(np.isfinite(scene.actor_local_positions)):
            raise ValueError(f"{path} contains non-finite actor-local positions")


def infer_dynamic_source_slots(
    scene: DynamicGaussianScene,
    *,
    gap_us: int = SOURCE_SLOT_GAP_US,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-Gaussian source-slot labels and median slot timestamps.

    Version 1 dynamic assets retain per-pixel rolling-shutter timestamps but
    not their source-frame indices.  Sorting the middle/source timestamps and
    splitting gaps larger than ``gap_us`` reconstructs the sparse InstantNuRec
    source observations without merging adjacent scanlines into different
    slots.
    """
    if gap_us <= 0:
        raise ValueError("dynamic source-slot gap must be positive")
    if scene.source_frame_slots is not None:
        slots = scene.source_frame_slots.astype(np.int32, copy=False)
        if scene.count == 0:
            return slots, np.empty((0,), dtype=np.int64)
        unique = np.unique(slots)
        if unique[0] < 0 or not np.array_equal(unique, np.arange(len(unique), dtype=unique.dtype)):
            raise ValueError("v2 source_frame_slots must be contiguous non-negative indices")
        centers = np.asarray(
            [np.median(scene.keyframe_timestamps_us[slots == slot, 1]) for slot in unique],
            dtype=np.int64,
        )
        return slots, centers
    if scene.count == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int64)
    middle = scene.keyframe_timestamps_us[:, 1]
    order = np.argsort(middle, kind="stable")
    sorted_middle = middle[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(sorted_middle) > gap_us).astype(np.int64) + 1,
            np.asarray([scene.count], dtype=np.int64),
        )
    )
    labels = np.empty((scene.count,), dtype=np.int32)
    centers = np.empty((len(boundaries) - 1,), dtype=np.int64)
    for slot, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        labels[order[start:end]] = slot
        centers[slot] = int(np.median(sorted_middle[start:end]))
    return labels, centers


def interpolate_dynamic_gaussians(
    scene: DynamicGaussianScene,
    timestamp_us: int,
    *,
    time_mode: str = "blend",
    actor_track_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and time-selected opacities.

    ``blend`` uses the original triangular temporal opacity. ``nearest`` keeps
    only the complete source-time cluster nearest to the target timestamp at
    full source opacity. ``actor-nearest`` independently chooses the nearest
    source slot available to every associated actor; unassociated Gaussians
    retain triangular blending. Positions are always moved to target time
    using each Gaussian's three track-warped keyframes.
    """
    if time_mode not in DYNAMIC_TIME_MODES:
        raise ValueError(f"unsupported dynamic time mode: {time_mode}")
    t = float(timestamp_us)
    times = scene.keyframe_timestamps_us.astype(np.float64)
    left = t <= times[:, 1]
    left_u = (t - times[:, 0]) / (times[:, 1] - times[:, 0])
    right_u = (t - times[:, 1]) / (times[:, 2] - times[:, 1])
    u = np.where(left, left_u, right_u)
    positions = np.where(
        left[:, None],
        scene.keyframe_positions[:, 0] + u[:, None] * (
            scene.keyframe_positions[:, 1] - scene.keyframe_positions[:, 0]
        ),
        scene.keyframe_positions[:, 1] + u[:, None] * (
            scene.keyframe_positions[:, 2] - scene.keyframe_positions[:, 1]
        ),
    )
    blend_weight = np.where(left, left_u, 1.0 - right_u)
    blend_weight = np.clip(blend_weight, 0.0, 1.0).astype(np.float32)
    if time_mode == "nearest" and scene.count:
        source_slots, slot_timestamps = infer_dynamic_source_slots(scene)
        selected_slot = int(np.argmin(np.abs(slot_timestamps.astype(np.float64) - t)))
        selected = source_slots == selected_slot
        in_support = (
            t >= float(times[selected, 0].min())
            and t <= float(times[selected, 2].max())
        )
        temporal_weight = selected.astype(np.float32) if in_support else np.zeros(
            (scene.count,), dtype=np.float32
        )
    elif time_mode == "actor-nearest" and scene.count:
        if actor_track_indices is None:
            raise ValueError("actor-nearest mode requires actor_track_indices")
        actor_tracks = np.asarray(actor_track_indices)
        if actor_tracks.dtype.kind not in "iu" or actor_tracks.shape != (scene.count,):
            raise ValueError("actor_track_indices must be a one-dimensional integer array")
        if np.any(actor_tracks < -1):
            raise ValueError("actor_track_indices must use -1 for unassociated Gaussians")
        source_slots, slot_timestamps = infer_dynamic_source_slots(scene)
        temporal_weight = blend_weight.copy()
        for track_index in np.unique(actor_tracks[actor_tracks >= 0]):
            track_mask = actor_tracks == track_index
            available_slots = np.unique(source_slots[track_mask])
            supported_slots = np.asarray(
                [
                    slot
                    for slot in available_slots
                    if np.any(
                        track_mask
                        & (source_slots == slot)
                        & (t >= times[:, 0])
                        & (t <= times[:, 2])
                    )
                ],
                dtype=np.int32,
            )
            if not len(supported_slots):
                continue
            selected_slot = supported_slots[
                np.argmin(np.abs(slot_timestamps[supported_slots].astype(np.float64) - t))
            ]
            selected = track_mask & (source_slots == selected_slot)
            support = selected & (t >= times[:, 0]) & (t <= times[:, 2])
            temporal_weight[track_mask] = 0.0
            temporal_weight[support] = 1.0
    else:
        temporal_weight = blend_weight
    opacities = scene.max_opacities * temporal_weight.astype(np.float32)
    return positions.astype(np.float32), opacities.astype(np.float32)
