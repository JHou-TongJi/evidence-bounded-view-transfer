from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np


@dataclass(frozen=True)
class GaussianScene:
    """Gaussian parameters in the activation space required by gsplat."""

    means: np.ndarray          # [N, 3]
    scales: np.ndarray         # [N, 3], activated positive scale
    quats_wxyz: np.ndarray     # [N, 4]
    opacities: np.ndarray      # [N]
    sh_coeffs: np.ndarray      # [N, K, 3]
    sh_degree: int

    @property
    def count(self) -> int:
        return int(self.means.shape[0])


def _numbered_fields(names: set[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    found = [(int(match.group(1)), name) for name in names if (match := pattern.match(name))]
    return [name for _, name in sorted(found)]


def load_gaussian_ply(path: str | Path) -> GaussianScene:
    """Load InstantNuRec or Graphdeco 3DGS PLY data.

    InstantNuRec exports DC-only SH colors. Graphdeco files may additionally
    contain f_rest_* fields. Extra InstantNuRec fields such as normals,
    road_mask, sky_mask, and display RGB are intentionally ignored.
    """
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("plyfile is required: pip install plyfile") from exc

    path = Path(path)
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing Gaussian fields: {', '.join(missing)}")

    def columns(field_names: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in field_names], axis=-1)

    means = columns(["x", "y", "z"])
    scales = np.exp(columns(["scale_0", "scale_1", "scale_2"])).astype(np.float32)
    quats = columns(["rot_0", "rot_1", "rot_2", "rot_3"])
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-12):
        raise ValueError(f"{path} contains zero-length Gaussian quaternions")
    quats = (quats / quat_norm).astype(np.float32)
    opacity_logits = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -30.0, 30.0)))).astype(np.float32)

    dc = columns(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    rest_fields = _numbered_fields(names, "f_rest_")
    if rest_fields:
        if len(rest_fields) % 3 != 0:
            raise ValueError(f"f_rest field count {len(rest_fields)} is not divisible by 3")
        rest_per_channel = len(rest_fields) // 3
        basis_count = rest_per_channel + 1
        degree = int(round(math.sqrt(basis_count) - 1))
        if (degree + 1) ** 2 != basis_count:
            raise ValueError(f"f_rest fields imply invalid SH basis count {basis_count}")
        # Graphdeco writes [N, 3, K-1] flattened channel-major.
        rest = columns(rest_fields).reshape(-1, 3, rest_per_channel).transpose(0, 2, 1)
        sh_coeffs = np.concatenate([dc, rest], axis=1).astype(np.float32)
    else:
        degree = 0
        sh_coeffs = dc.astype(np.float32)

    arrays = (means, scales, quats, opacities, sh_coeffs)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{path} contains non-finite Gaussian parameters")

    return GaussianScene(
        means=means,
        scales=scales,
        quats_wxyz=quats,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        sh_degree=degree,
    )

