from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_dense_surface_fuse import Mast3rDenseSurfaceFuseOptions, fuse_mast3r_dense_surfaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse two accepted MASt3R dense surfaces without cross-camera colour averaging")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--primary-dense-audit", required=True, type=Path)
    parser.add_argument("--secondary-dense-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--voxel-size-m", type=float, default=.025)
    parser.add_argument("--gaussian-scale-m", type=float, default=.018)
    parser.add_argument("--opacity", type=float, default=.72)
    args = parser.parse_args(argv)
    fuse_mast3r_dense_surfaces(Mast3rDenseSurfaceFuseOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
