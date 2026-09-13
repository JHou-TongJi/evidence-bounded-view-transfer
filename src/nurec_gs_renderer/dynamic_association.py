from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from .color_correction import file_sha256
from .dynamic import DynamicGaussianScene, infer_dynamic_source_slots


ASSOCIATION_SCHEMA = "instant-nurec-dynamic-actor-association"
ASSOCIATION_VERSION = 1


@dataclass(frozen=True)
class DynamicActorAssociation:
    """Hash-bound actor/source-slot labels for a dynamic Gaussian asset."""

    track_indices: np.ndarray
    source_slot_indices: np.ndarray
    slot_timestamps_us: np.ndarray
    dynamic_sha256: str
    gaussian_count: int
    tracks: tuple[dict[str, object], ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        track_indices = np.asarray(self.track_indices)
        source_slots = np.asarray(self.source_slot_indices)
        slot_timestamps = np.asarray(self.slot_timestamps_us)
        if track_indices.dtype != np.int32 or track_indices.shape != (self.gaussian_count,):
            raise ValueError("actor track_indices must have shape [N] and dtype int32")
        if source_slots.dtype != np.int16 or source_slots.shape != (self.gaussian_count,):
            raise ValueError("actor source_slot_indices must have shape [N] and dtype int16")
        if slot_timestamps.dtype != np.int64 or slot_timestamps.ndim != 1:
            raise ValueError("actor slot_timestamps_us must be one-dimensional int64")
        if self.gaussian_count < 0:
            raise ValueError("actor gaussian_count must be non-negative")
        if len(self.dynamic_sha256) != 64:
            raise ValueError("actor dynamic Gaussian SHA-256 is invalid")
        if len(slot_timestamps) and np.any(np.diff(slot_timestamps) <= 0):
            raise ValueError("actor source-slot timestamps must be strictly increasing")
        if len(source_slots) and (
            source_slots.min() < 0 or source_slots.max() >= len(slot_timestamps)
        ):
            raise ValueError("actor source_slot_indices contain an out-of-range slot")
        if len(track_indices) and (
            track_indices.min() < -1 or track_indices.max(initial=-1) >= len(self.tracks)
        ):
            raise ValueError("actor track_indices contain an out-of-range track")

    @property
    def associated_count(self) -> int:
        return int(np.count_nonzero(self.track_indices >= 0))

    @classmethod
    def load(cls, path: str | Path) -> "DynamicActorAssociation":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "track_indices",
                    "source_slot_indices",
                    "slot_timestamps_us",
                    "metadata",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(f"actor association is missing arrays: {sorted(missing)}")
                container = json.loads(str(archive["metadata"].item()))
                if container.get("schema") != ASSOCIATION_SCHEMA:
                    raise ValueError("unsupported actor association schema")
                if int(container.get("version", -1)) != ASSOCIATION_VERSION:
                    raise ValueError(
                        f"unsupported actor association version: {container.get('version')}"
                    )
                return cls(
                    track_indices=np.asarray(archive["track_indices"]),
                    source_slot_indices=np.asarray(archive["source_slot_indices"]),
                    slot_timestamps_us=np.asarray(archive["slot_timestamps_us"]),
                    dynamic_sha256=str(container["dynamic_sha256"]),
                    gaussian_count=int(container["gaussian_count"]),
                    tracks=tuple(container["tracks"]),
                    metadata=dict(container.get("details", {})),
                )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load dynamic actor association {path}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("dynamic actor association must use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        container = {
            "schema": ASSOCIATION_SCHEMA,
            "version": ASSOCIATION_VERSION,
            "dynamic_sha256": self.dynamic_sha256,
            "gaussian_count": self.gaussian_count,
            "tracks": list(self.tracks),
            "details": self.metadata,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    track_indices=self.track_indices,
                    source_slot_indices=self.source_slot_indices,
                    slot_timestamps_us=self.slot_timestamps_us,
                    metadata=np.asarray(json.dumps(container, ensure_ascii=False)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def validate_source(
        self,
        dynamic_path: str | Path,
        scene: DynamicGaussianScene,
    ) -> None:
        if scene.count != self.gaussian_count:
            raise ValueError(
                f"actor association expects {self.gaussian_count:,} dynamic Gaussians, "
                f"got {scene.count:,}"
            )
        actual_hash = file_sha256(dynamic_path)
        if actual_hash != self.dynamic_sha256:
            raise ValueError(
                "actor association dynamic asset hash mismatch: "
                f"expected {self.dynamic_sha256}, got {actual_hash}"
            )
        source_slots, slot_timestamps = infer_dynamic_source_slots(scene)
        if not np.array_equal(source_slots.astype(np.int16), self.source_slot_indices):
            raise ValueError("actor association source-slot labels do not match dynamic asset")
        if not np.array_equal(slot_timestamps, self.slot_timestamps_us):
            raise ValueError("actor association source-slot timestamps do not match dynamic asset")

