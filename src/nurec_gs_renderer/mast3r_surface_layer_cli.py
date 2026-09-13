from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_surface_layer import Mast3rSurfaceLayerOptions, build_mast3r_surface_layer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a conservative MASt3R observed-surface canonical actor")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--correspondence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--voxel-size-m", type=float, default=.05)
    parser.add_argument("--gaussian-scale-m", type=float, default=.035)
    parser.add_argument("--opacity", type=float, default=.65)
    parser.add_argument("--source-camera", help="Use colour only from this original correspondence camera")
    parser.add_argument("--reference-index", type=int, help="Use geometry only from one accepted reference frame")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    build_mast3r_surface_layer(Mast3rSurfaceLayerOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
