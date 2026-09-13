"""Rigid, object-local Gaussian fusion for vehicle-only dynamic experiments.

This module intentionally performs *initialisation*, not a hidden replacement
for a dynamic reconstruction trainer.  It converts duplicated source-slot
Gaussians into one actor-local set, records every input/provenance count, and
leaves later RGB/depth optimisation explicit and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .color_correction import file_sha256
from .dynamic import DynamicGaussianScene
from .dynamic_association import DynamicActorAssociation


CANONICAL_SCHEMA = "nurec-gs-canonical-actors"
CANONICAL_VERSION = 1


@dataclass(frozen=True)
class ActorTrajectory:
    """NCore actor-to-world poses sampled at cuboid track timestamps."""

    timestamps_us: np.ndarray  # [T]
    actor_to_world: np.ndarray  # [T,4,4]

    def __post_init__(self) -> None:
        if self.timestamps_us.ndim != 1 or self.actor_to_world.shape != (len(self.timestamps_us), 4, 4):
            raise ValueError("actor trajectory requires [T] timestamps and [T,4,4] poses")
        if len(self.timestamps_us) < 2 or np.any(np.diff(self.timestamps_us) <= 0):
            raise ValueError("actor trajectory needs at least two increasing timestamps")
        if not np.all(np.isfinite(self.actor_to_world)):
            raise ValueError("actor trajectory contains non-finite poses")


@dataclass(frozen=True)
class CanonicalActorAsset:
    """One deduplicated Gaussian set per rigid actor plus its world trajectory."""

    positions: np.ndarray              # [M,3], actor-local
    rotations_wxyz: np.ndarray         # [M,4], actor-local
    scales: np.ndarray                 # [M,3]
    rgb: np.ndarray                    # [M,3], sRGB
    opacities: np.ndarray              # [M]
    actor_indices: np.ndarray          # [M], index into actor_ids
    visibility_counts: np.ndarray      # [M], unique source camera×slot observations
    source_counts: np.ndarray          # [M], merged source Gaussians
    actor_ids: tuple[int, ...]
    trajectories: tuple[ActorTrajectory, ...]
    metadata: dict[str, object]

    @property
    def count(self) -> int:
        return int(len(self.positions))

    def validate(self) -> None:
        count = self.count
        for name, shape in (
            ("positions", (count, 3)), ("rotations_wxyz", (count, 4)),
            ("scales", (count, 3)), ("rgb", (count, 3)), ("opacities", (count,)),
            ("actor_indices", (count,)), ("visibility_counts", (count,)),
            ("source_counts", (count,)),
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"canonical asset field {name} must have shape {shape}")
        if len(self.actor_ids) != len(self.trajectories) or not self.actor_ids:
            raise ValueError("canonical asset actor IDs and trajectories are inconsistent")
        if np.any(self.actor_indices < 0) or np.any(self.actor_indices >= len(self.actor_ids)):
            raise ValueError("canonical asset has invalid actor indices")
        if np.any(self.scales <= 0) or np.any((self.opacities < 0) | (self.opacities > 1)):
            raise ValueError("canonical asset scales/opacities are invalid")
        if not all(np.all(np.isfinite(getattr(self, field))) for field in ("positions", "rotations_wxyz", "scales", "rgb", "opacities")):
            raise ValueError("canonical asset contains non-finite Gaussian data")
        if self.metadata.get("schema") != CANONICAL_SCHEMA or int(self.metadata.get("version", -1)) != CANONICAL_VERSION:
            raise ValueError("unsupported canonical actor asset schema")
        for trajectory in self.trajectories:
            trajectory.__post_init__()

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        max_poses = max(len(item.timestamps_us) for item in self.trajectories)
        timestamps = np.full((len(self.trajectories), max_poses), -1, dtype=np.int64)
        poses = np.zeros((len(self.trajectories), max_poses, 4, 4), dtype=np.float32)
        valid = np.zeros((len(self.trajectories), max_poses), dtype=np.uint8)
        for index, trajectory in enumerate(self.trajectories):
            length = len(trajectory.timestamps_us)
            timestamps[index, :length] = trajectory.timestamps_us
            poses[index, :length] = trajectory.actor_to_world
            valid[index, :length] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positions=self.positions.astype(np.float32), rotations_wxyz=self.rotations_wxyz.astype(np.float32),
            scales=self.scales.astype(np.float32), rgb=self.rgb.astype(np.float32),
            opacities=self.opacities.astype(np.float32), actor_indices=self.actor_indices.astype(np.int32),
            visibility_counts=self.visibility_counts.astype(np.uint16), source_counts=self.source_counts.astype(np.uint16),
            actor_ids=np.asarray(self.actor_ids, dtype=np.int64), track_timestamps_us=timestamps,
            actor_to_world=poses, track_pose_valid=valid, metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalActorAsset":
        path = Path(path)
        required = {
            "positions", "rotations_wxyz", "scales", "rgb", "opacities", "actor_indices",
            "visibility_counts", "source_counts", "actor_ids", "track_timestamps_us", "actor_to_world",
            "track_pose_valid", "metadata",
        }
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} is missing canonical fields: {', '.join(missing)}")
            metadata = json.loads(str(data["metadata"].item()))
            timestamps = np.asarray(data["track_timestamps_us"], dtype=np.int64)
            poses = np.asarray(data["actor_to_world"], dtype=np.float32)
            valid = np.asarray(data["track_pose_valid"], dtype=bool)
            if timestamps.shape != valid.shape or poses.shape[:2] != timestamps.shape or poses.shape[2:] != (4, 4):
                raise ValueError(f"{path} has invalid track trajectory arrays")
            trajectories = tuple(
                ActorTrajectory(timestamps[index, valid[index]], poses[index, valid[index]])
                for index in range(len(timestamps))
            )
            asset = cls(
                positions=np.asarray(data["positions"], dtype=np.float32),
                rotations_wxyz=np.asarray(data["rotations_wxyz"], dtype=np.float32),
                scales=np.asarray(data["scales"], dtype=np.float32), rgb=np.asarray(data["rgb"], dtype=np.float32),
                opacities=np.asarray(data["opacities"], dtype=np.float32),
                actor_indices=np.asarray(data["actor_indices"], dtype=np.int32),
                visibility_counts=np.asarray(data["visibility_counts"], dtype=np.uint16),
                source_counts=np.asarray(data["source_counts"], dtype=np.uint16),
                actor_ids=tuple(int(value) for value in np.asarray(data["actor_ids"], dtype=np.int64)),
                trajectories=trajectories, metadata=metadata,
            )
        asset.validate()
        return asset


def _rotation_matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _wxyz_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif axis == 1:
            s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return (q / max(float(np.linalg.norm(q)), 1e-12)).astype(np.float32)


def _interpolate_pose(trajectory: ActorTrajectory, timestamp_us: int) -> np.ndarray:
    """Translation-linear, rotation-SVD interpolation suitable for initialisation."""
    times = trajectory.timestamps_us
    right = int(np.searchsorted(times, timestamp_us, side="right"))
    right = min(max(right, 1), len(times) - 1)
    left = right - 1
    u = float((timestamp_us - times[left]) / max(int(times[right] - times[left]), 1))
    u = float(np.clip(u, 0.0, 1.0))
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = (1 - u) * trajectory.actor_to_world[left, :3, 3] + u * trajectory.actor_to_world[right, :3, 3]
    blended = (1 - u) * trajectory.actor_to_world[left, :3, :3] + u * trajectory.actor_to_world[right, :3, :3]
    u_mat, _, v_mat = np.linalg.svd(blended)
    output[:3, :3] = u_mat @ v_mat
    return output.astype(np.float32)


def recover_v1_actor_provenance(
    scene: DynamicGaussianScene,
    association: DynamicActorAssociation,
    trajectories: Mapping[int, ActorTrajectory],
) -> DynamicGaussianScene:
    """Lift a hash-validated v1 actor association into v2-like provenance.

    The original ``pa-front`` export predates native track/source fields.  Its
    association sidecar nevertheless binds every dynamic Gaussian to an NCore
    track index and a sparse source-time slot.  For rigid vehicle actors, the
    middle keyframe can therefore be transformed into that track's canonical
    frame deterministically.  The result deliberately marks its provenance as
    *recovered*: source-camera identity and point-vs-ray association were not
    present in the v1 file, so they must never be represented as native data.
    """
    if scene.source_track_ids is not None:
        raise ValueError("v1 provenance recovery is only valid for a v1 dynamic asset")
    if association.gaussian_count != scene.count:
        raise ValueError("association Gaussian count does not match the dynamic scene")
    track_id_by_index: dict[int, int] = {}
    for index, record in enumerate(association.tracks):
        try:
            track_id = int(record["track_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"association track[{index}] has no numeric track_id") from exc
        track_id_by_index[index] = track_id

    source_track_ids = np.full((scene.count,), -1, dtype=np.int64)
    local_positions = np.zeros((scene.count, 3), dtype=np.float32)
    recovered = np.zeros((scene.count,), dtype=bool)
    for association_index, track_id in track_id_by_index.items():
        trajectory = trajectories.get(track_id)
        if trajectory is None:
            continue
        members = np.flatnonzero(association.track_indices == association_index)
        for member in members:
            pose = _interpolate_pose(trajectory, int(scene.keyframe_timestamps_us[member, 1]))
            # Row-vector convention: p_world = p_local @ R.T + t.
            local_positions[member] = (
                scene.keyframe_positions[member, 1] - pose[:3, 3]
            ) @ pose[:3, :3]
        source_track_ids[members] = track_id
        recovered[members] = True

    metadata = dict(scene.metadata)
    metadata.update({
        "version": 2,
        "provenance": "recovered_from_v1_dynamic_actor_association",
        "native_source_camera_indices": False,
        "native_association_sources": False,
        "recovered_associated_gaussian_count": int(recovered.sum()),
    })
    return DynamicGaussianScene(
        keyframe_positions=scene.keyframe_positions,
        keyframe_timestamps_us=scene.keyframe_timestamps_us,
        rotations_wxyz=scene.rotations_wxyz,
        scales=scene.scales,
        rgb=scene.rgb,
        max_opacities=scene.max_opacities,
        metadata=metadata,
        source_track_ids=source_track_ids,
        # v1 did not retain source camera identity.  Slot remains useful as a
        # conservative unique-observation proxy in the fusion statistics.
        source_camera_indices=np.full((scene.count,), -1, dtype=np.int16),
        source_frame_slots=association.source_slot_indices.astype(np.int16, copy=True),
        # The v1 sidecar only establishes that the middle point was associated
        # with a padded cuboid; retain it as a recovered association, not as a
        # fabricated point-vs-ray distinction.
        association_sources=recovered.astype(np.uint8),
        actor_local_positions=local_positions,
    )


def build_canonical_actor_asset(
    scene: DynamicGaussianScene,
    trajectories: Mapping[int, ActorTrajectory],
    *,
    dynamic_path: str | Path,
    eligible_track_ids: set[int],
    voxel_size_m: float = 0.05,
) -> CanonicalActorAsset:
    """Fuse only explicitly eligible rigid actors from a v2 dynamic asset."""
    if scene.source_track_ids is None or scene.actor_local_positions is None:
        raise ValueError("canonical fusion requires a v2 dynamic asset with source actor provenance")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    trajectory_ids = {track_id for track_id in eligible_track_ids if track_id in trajectories}
    if not trajectory_ids:
        raise ValueError("no eligible vehicle track has a trajectory")
    associated = scene.association_sources > 0 if scene.association_sources is not None else np.zeros(scene.count, dtype=bool)
    observed_ids = {int(track_id) for track_id in scene.source_track_ids[associated] if int(track_id) in trajectory_ids}
    actor_ids = tuple(sorted(observed_ids))
    if not actor_ids:
        raise ValueError("no v2 dynamic Gaussian is point/ray associated with an eligible vehicle")
    actor_index = {track_id: index for index, track_id in enumerate(actor_ids)}
    selected = np.asarray([int(track) in actor_index for track in scene.source_track_ids], dtype=bool)
    selected &= associated
    if not selected.any():
        raise ValueError("no v2 dynamic Gaussian is point/ray associated with an eligible vehicle")
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index in np.flatnonzero(selected):
        local = scene.actor_local_positions[index]
        voxel = tuple(np.floor(local / voxel_size_m).astype(np.int64))
        groups.setdefault((actor_index[int(scene.source_track_ids[index])], *voxel), []).append(int(index))
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    opacities: list[float] = []
    indices: list[int] = []
    visibilities: list[int] = []
    source_counts: list[int] = []
    for (actor_idx, *_), members in groups.items():
        member = np.asarray(members, dtype=np.int64)
        weights = np.maximum(scene.max_opacities[member].astype(np.float64), 1e-4)
        weights /= weights.sum()
        positions.append((scene.actor_local_positions[member] * weights[:, None]).sum(axis=0))
        scales.append((scene.scales[member] * weights[:, None]).sum(axis=0))
        rgb.append((scene.rgb[member] * weights[:, None]).sum(axis=0))
        opacities.append(float(np.clip(scene.max_opacities[member].max(), 0.0, 1.0)))
        # Convert each source orientation into actor-local before taking the
        # highest-confidence representative.  A full covariance merge belongs
        # to the subsequent optimizer, not this deterministic initialiser.
        best = member[int(np.argmax(scene.max_opacities[member]))]
        pose = _interpolate_pose(trajectories[actor_ids[actor_idx]], int(scene.keyframe_timestamps_us[best, 1]))
        local_rotation = pose[:3, :3].T @ _rotation_matrix_from_wxyz(scene.rotations_wxyz[best])
        rotations.append(_wxyz_from_rotation_matrix(local_rotation))
        indices.append(actor_idx)
        source_counts.append(len(member))
        camera = scene.source_camera_indices[member] if scene.source_camera_indices is not None else np.full(len(member), -1)
        slot = scene.source_frame_slots[member] if scene.source_frame_slots is not None else np.full(len(member), -1)
        visibilities.append(len(set(zip(camera.tolist(), slot.tolist(), strict=True))))
    metadata = {
        "schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION,
        "coordinate_frame": "actor_local", "source_dynamic_path": str(Path(dynamic_path).resolve()),
        "source_dynamic_sha256": file_sha256(dynamic_path), "voxel_size_m": voxel_size_m,
        "eligible_track_ids": list(actor_ids), "association_policy": "v2 point/ray cuboid associated only",
        "optimizer_status": "initialization_only; no RGB/depth optimisation applied",
    }
    asset = CanonicalActorAsset(
        positions=np.asarray(positions, dtype=np.float32), rotations_wxyz=np.asarray(rotations, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32), rgb=np.asarray(rgb, dtype=np.float32),
        opacities=np.asarray(opacities, dtype=np.float32), actor_indices=np.asarray(indices, dtype=np.int32),
        visibility_counts=np.asarray(visibilities, dtype=np.uint16), source_counts=np.asarray(source_counts, dtype=np.uint16),
        actor_ids=actor_ids, trajectories=tuple(trajectories[track_id] for track_id in actor_ids), metadata=metadata,
    )
    asset.validate()
    return asset
