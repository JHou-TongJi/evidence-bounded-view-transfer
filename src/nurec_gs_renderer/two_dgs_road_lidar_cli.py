"""CLI for conservative raw-LiDAR road seed generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .two_dgs_road_lidar import TwoDgsRoadLidarSeedOptions, build_two_dgs_road_lidar_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build static road LiDAR seeds for an official 2DGS proxy")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-mask-manifest", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--source-camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=299)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--voxel-size-m", type=float, default=.15)
    parser.add_argument("--min-voxel-observations", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--min-range-m", type=float, default=3.0)
    parser.add_argument("--max-range-m", type=float, default=60.0)
    parser.add_argument("--ground-local-z-min-m", type=float, default=-3.2)
    parser.add_argument("--ground-local-z-max-m", type=float, default=-1.4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    build_two_dgs_road_lidar_seed(TwoDgsRoadLidarSeedOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
