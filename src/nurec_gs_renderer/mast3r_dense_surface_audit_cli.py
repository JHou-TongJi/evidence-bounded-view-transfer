from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_dense_surface_audit import Mast3rDenseSurfaceAuditOptions, audit_mast3r_dense_surface


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a MASt3R dense surface against its exact original source instance mask")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--static-render-dir", type=Path, required=True)
    parser.add_argument("--actor-render-dir", type=Path, required=True)
    parser.add_argument("--composite-render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-id", default="camera_cross_right_120fov")
    parser.add_argument("--alpha-threshold", type=float, default=.02)
    parser.add_argument("--min-iou", type=float, default=.60)
    parser.add_argument("--min-recall", type=float, default=.75)
    parser.add_argument("--min-mae-improvement-percent", type=float, default=3.0)
    parser.add_argument("--max-outside-change-fraction", type=float, default=.02)
    args = parser.parse_args(argv)
    audit_mast3r_dense_surface(Mast3rDenseSurfaceAuditOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
