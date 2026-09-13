from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .ply_io import GaussianScene


CORRECTION_VERSION = 1
SH_C0 = 0.28209479177387814


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return np.asarray(f_dc, dtype=np.float32) * SH_C0 + 0.5


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0


@dataclass(frozen=True)
class GaussianColorCorrection:
    indices: np.ndarray
    original_f_dc: np.ndarray
    corrected_f_dc: np.ndarray
    confidence: np.ndarray
    support_count: np.ndarray
    ply_sha256: str
    gaussian_count: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        original = np.asarray(self.original_f_dc)
        corrected = np.asarray(self.corrected_f_dc)
        confidence = np.asarray(self.confidence)
        support = np.asarray(self.support_count)
        count = len(indices)
        if indices.dtype != np.int64 or indices.shape != (count,):
            raise ValueError("correction indices must be a one-dimensional int64 array")
        if original.dtype != np.float32 or original.shape != (count, 3):
            raise ValueError("original_f_dc must have shape [M,3] and dtype float32")
        if corrected.dtype != np.float32 or corrected.shape != (count, 3):
            raise ValueError("corrected_f_dc must have shape [M,3] and dtype float32")
        if confidence.dtype != np.float32 or confidence.shape != (count,):
            raise ValueError("confidence must have shape [M] and dtype float32")
        if support.dtype != np.uint16 or support.shape != (count,):
            raise ValueError("support_count must have shape [M] and dtype uint16")
        if self.gaussian_count <= 0:
            raise ValueError("gaussian_count must be positive")
        if count and (indices.min() < 0 or indices.max() >= self.gaussian_count):
            raise ValueError("correction contains an out-of-range Gaussian index")
        if len(np.unique(indices)) != count:
            raise ValueError("correction indices must be unique")
        if not all(np.all(np.isfinite(value)) for value in (original, corrected, confidence)):
            raise ValueError("correction contains non-finite values")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("correction confidence must be in [0,1]")
        if len(self.ply_sha256) != 64:
            raise ValueError("correction PLY SHA-256 is invalid")

    @property
    def count(self) -> int:
        return int(len(self.indices))

    @classmethod
    def load(cls, path: str | Path) -> "GaussianColorCorrection":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "indices",
                    "original_f_dc",
                    "corrected_f_dc",
                    "confidence",
                    "support_count",
                    "metadata",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(f"correction is missing arrays: {sorted(missing)}")
                metadata = json.loads(str(archive["metadata"].item()))
                if int(metadata.get("version", -1)) != CORRECTION_VERSION:
                    raise ValueError(f"unsupported correction version: {metadata.get('version')}")
                return cls(
                    indices=np.asarray(archive["indices"]),
                    original_f_dc=np.asarray(archive["original_f_dc"]),
                    corrected_f_dc=np.asarray(archive["corrected_f_dc"]),
                    confidence=np.asarray(archive["confidence"]),
                    support_count=np.asarray(archive["support_count"]),
                    ply_sha256=str(metadata["ply_sha256"]),
                    gaussian_count=int(metadata["gaussian_count"]),
                    metadata=dict(metadata.get("details", {})),
                )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load Gaussian color correction {path}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Gaussian color correction must use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": CORRECTION_VERSION,
            "ply_sha256": self.ply_sha256,
            "gaussian_count": self.gaussian_count,
            "details": self.metadata,
        }
        np.savez_compressed(
            path,
            indices=self.indices,
            original_f_dc=self.original_f_dc,
            corrected_f_dc=self.corrected_f_dc,
            confidence=self.confidence,
            support_count=self.support_count,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )

    def validate_source(self, ply_path: str | Path, gaussian_count: int) -> None:
        if gaussian_count != self.gaussian_count:
            raise ValueError(
                f"correction expects {self.gaussian_count:,} Gaussians, got {gaussian_count:,}"
            )
        actual_hash = file_sha256(ply_path)
        if actual_hash != self.ply_sha256:
            raise ValueError(
                "correction PLY hash mismatch: "
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
                f"correction expects {self.gaussian_count:,} Gaussians, got {scene.count:,}"
            )
        if scene.sh_coeffs.shape[1] < 1:
            raise ValueError("scene does not contain a DC spherical-harmonic coefficient")
        current = scene.sh_coeffs[self.indices, 0]
        if verify_original and not np.allclose(current, self.original_f_dc, atol=atol, rtol=0.0):
            delta = float(np.max(np.abs(current - self.original_f_dc)))
            raise ValueError(f"scene DC colors do not match correction source (max delta {delta:.3g})")
        colors = scene.sh_coeffs.copy()
        colors[self.indices, 0] = self.corrected_f_dc
        return replace(scene, sh_coeffs=colors)


def bake_color_correction(
    ply_path: str | Path,
    correction_path: str | Path,
    output_path: str | Path,
) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:  # pragma: no cover - base dependency
        raise RuntimeError("PLY color baking requires plyfile") from exc

    ply_path = Path(ply_path)
    output_path = Path(output_path)
    if ply_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite the source PLY")
    correction = GaussianColorCorrection.load(correction_path)
    correction.validate_source(ply_path, correction.gaussian_count)
    ply = PlyData.read(str(ply_path))
    vertex_element = ply["vertex"]
    data = vertex_element.data.copy()
    if len(data) != correction.gaussian_count:
        raise ValueError("PLY vertex count does not match correction")
    names = set(data.dtype.names or ())
    required = {"f_dc_0", "f_dc_1", "f_dc_2"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"PLY is missing DC color fields: {sorted(missing)}")
    for channel, field in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        original = np.asarray(data[field][correction.indices], dtype=np.float32)
        if not np.allclose(original, correction.original_f_dc[:, channel], atol=2e-6, rtol=0.0):
            raise ValueError(f"PLY field {field} does not match correction source")
        data[field][correction.indices] = correction.corrected_f_dc[:, channel]
    rgb = np.clip(sh_dc_to_rgb(correction.corrected_f_dc), 0.0, 1.0)
    for channel, field in enumerate(("red", "green", "blue")):
        if field in names:
            data[field][correction.indices] = np.round(rgb[:, channel] * 255.0).astype(data[field].dtype)

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(data, "vertex", comments=element.comments))
        else:
            elements.append(element)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(str(output_path))

