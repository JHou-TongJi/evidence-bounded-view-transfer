"""CLI for the no-download LiDAR-BEV dynamic occupancy gate audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from .bev_occupancy_gate import BevOccupancyGateOptions, audit_bev_occupancy_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit raw-LiDAR BEV occupancy support on held-out NCore actor masks")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--actor-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-ids", required=True, help="Comma-separated NCore camera IDs")
    parser.add_argument("--reference-start", type=int, required=True)
    parser.add_argument("--reference-end", type=int, required=True)
    parser.add_argument("--reference-parity", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size-m", type=float, default=.10)
    parser.add_argument("--dilation-cells-xy", type=int, default=1)
    parser.add_argument("--dilation-cells-z", type=int, default=1)
    parser.add_argument("--roi-margin-pixels", type=int, default=16)
    args = vars(parser.parse_args(argv))
    args["camera_ids"] = tuple(value.strip() for value in args["camera_ids"].split(",") if value.strip())
    audit_bev_occupancy_gate(BevOccupancyGateOptions(**args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
