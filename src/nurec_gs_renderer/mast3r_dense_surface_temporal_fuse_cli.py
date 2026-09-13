from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_dense_surface_temporal_fuse import Mast3rDenseSurfaceTemporalFuseOptions, fuse_mast3r_dense_surfaces_temporally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse per-reference observed dense actors into one short-window rigid canonical surface")
    parser.add_argument("--actors", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--priority-reference", required=True, type=int)
    parser.add_argument("--voxel-size-m", type=float, default=.025)
    parser.add_argument("--gaussian-scale-m", type=float, default=.018)
    parser.add_argument("--opacity", type=float, default=.72)
    args = parser.parse_args(argv)
    fuse_mast3r_dense_surfaces_temporally(Mast3rDenseSurfaceTemporalFuseOptions(tuple(args.actors), args.output, args.track_id, args.priority_reference, args.voxel_size_m, args.gaussian_scale_m, args.opacity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
