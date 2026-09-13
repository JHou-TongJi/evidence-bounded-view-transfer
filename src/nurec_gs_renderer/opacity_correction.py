from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .color_correction import file_sha256
from .ply_io import GaussianScene


OPACITY_CORRECTION_VERSION = 1


@dataclass(frozen=True)
class GaussianOpacityCorrection:
    """Hash-bound, reversible replacement of selected activated opacities."""

    indices: np.ndarray
    original_opacities: np.ndarray
    corrected_opacities: np.ndarray
    confidence: np.ndarray
    support_count: np.ndarray
    ply_sha256: str
    gaussian_count: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        original = np.asarray(self.original_opacities)
        corrected = np.asarray(self.corrected_opacities)
        confidence = np.asarray(self.confidence)
        support = np.asarray(self.support_count)
        count = len(indices)
        if indices.dtype != np.int64 or indices.shape != (count,):
            raise ValueError("opacity correction indices must be a one-dimensional int64 array")
        for name, value, dtype in (
            ("original_opacities", original, np.float32),
            ("corrected_opacities", corrected, np.float32),
            ("confidence", confidence, np.float32),
        ):
            if value.dtype != dtype or value.shape != (count,):
                raise ValueError(f"{name} must have shape [M] and dtype {dtype.__name__}")
        if support.dtype != np.uint16 or support.shape != (count,):
            raise ValueError("support_count must have shape [M] and dtype uint16")
        if self.gaussian_count <= 0:
            raise ValueError("gaussian_count must be positive")
        if count and (indices.min() < 0 or indices.max() >= self.gaussian_count):
            raise ValueError("opacity correction contains an out-of-range Gaussian index")
        if len(np.unique(indices)) != count:
            raise ValueError("opacity correction indices must be unique")
        if not all(np.all(np.isfinite(value)) for value in (original, corrected, confidence)):
            raise ValueError("opacity correction contains non-finite values")
        if np.any((original < 0.0) | (original > 1.0)) or np.any(
            (corrected < 0.0) | (corrected > 1.0)
        ):
            raise ValueError("opacity values must be in [0,1]")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("opacity correction confidence must be in [0,1]")
        if len(self.ply_sha256) != 64:
            raise ValueError("opacity correction PLY SHA-256 is invalid")

    @property
    def count(self) -> int:
        return int(len(self.indices))

    @classmethod
    def load(cls, path: str | Path) -> "GaussianOpacityCorrection":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "indices",
                    "original_opacities",
                    "corrected_opacities",
                    "confidence",
                    "support_count",
                    "metadata",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(f"opacity correction is missing arrays: {sorted(missing)}")
                metadata = json.loads(str(archive["metadata"].item()))
                if int(metadata.get("version", -1)) != OPACITY_CORRECTION_VERSION:
                    raise ValueError(
                        f"unsupported opacity correction version: {metadata.get('version')}"
                    )
                return cls(
                    indices=np.asarray(archive["indices"]),
                    original_opacities=np.asarray(archive["original_opacities"]),
                    corrected_opacities=np.asarray(archive["corrected_opacities"]),
                    confidence=np.asarray(archive["confidence"]),
                    support_count=np.asarray(archive["support_count"]),
                    ply_sha256=str(metadata["ply_sha256"]),
                    gaussian_count=int(metadata["gaussian_count"]),
                    metadata=dict(metadata.get("details", {})),
                )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load Gaussian opacity correction {path}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Gaussian opacity correction must use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": OPACITY_CORRECTION_VERSION,
            "ply_sha256": self.ply_sha256,
            "gaussian_count": self.gaussian_count,
            "details": self.metadata,
        }
        np.savez_compressed(
            path,
            indices=self.indices,
            original_opacities=self.original_opacities,
            corrected_opacities=self.corrected_opacities,
            confidence=self.confidence,
            support_count=self.support_count,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )

    def validate_source(self, ply_path: str | Path, gaussian_count: int) -> None:
        if gaussian_count != self.gaussian_count:
            raise ValueError(
                f"opacity correction expects {self.gaussian_count:,} Gaussians, got {gaussian_count:,}"
            )
        actual_hash = file_sha256(ply_path)
        if actual_hash != self.ply_sha256:
            raise ValueError(
                "opacity correction PLY hash mismatch: "
                f"expected {self.ply_sha256}, got {actual_hash}"
            )

    def apply_to_scene(
        self,
        scene: GaussianScene,
        *,
        verify_original: bool = True,
        atol: float = 2e-6,
    ) -> GaussianScene:
        if scene.count != self.gaussian_count:
            raise ValueError(
                f"opacity correction expects {self.gaussian_count:,} Gaussians, got {scene.count:,}"
            )
        current = scene.opacities[self.indices]
        if verify_original and not np.allclose(
            current, self.original_opacities, atol=atol, rtol=0.0
        ):
            delta = float(np.max(np.abs(current - self.original_opacities)))
            raise ValueError(
                f"scene opacities do not match correction source (max delta {delta:.3g})"
            )
        opacities = scene.opacities.copy()
        opacities[self.indices] = self.corrected_opacities
        return replace(scene, opacities=opacities)
