"""CLI for a non-destructive LiDAR-anchored Depth Anything V2 2DGS prior."""
from __future__ import annotations

import argparse
from pathlib import Path

from .two_dgs_depth_anything import TwoDgsDepthAnythingOptions, build_two_dgs_depth_anything_prior


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LiDAR-anchored DAV2 pseudo-depth for a static NCore 2DGS proxy")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--min-lidar-pixels", type=int, default=128)
    parser.add_argument("--support-radius-px", type=float, default=16.0)
    parser.add_argument("--max-log-gradient", type=float, default=.22)
    parser.add_argument("--min-depth-m", type=float, default=2.0)
    parser.add_argument("--max-depth-m", type=float, default=60.0)
    args = parser.parse_args(argv)
    build_two_dgs_depth_anything_prior(TwoDgsDepthAnythingOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
