from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_dense_surface_layer import Mast3rDenseSurfaceLayerOptions, build_mast3r_dense_surface_layer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a visibility-bound actor from an accepted MASt3R metric dense-depth audit")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dense-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--voxel-size-m", type=float, default=.025)
    parser.add_argument("--gaussian-scale-m", type=float, default=.018)
    parser.add_argument("--opacity", type=float, default=.72)
    args = parser.parse_args(argv)
    build_mast3r_dense_surface_layer(Mast3rDenseSurfaceLayerOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
