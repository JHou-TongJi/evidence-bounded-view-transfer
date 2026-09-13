from __future__ import annotations

import argparse
from pathlib import Path

from .roma_multiview_audit import RomaMultiviewAuditOptions, audit_roma_multiview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local RoMa cross-camera matches with NCore FTheta rolling-shutter geometry")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--roma-root", type=Path, required=True)
    parser.add_argument("--roma-weights", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-a", default="camera_front_tele_30fov")
    parser.add_argument("--camera-b", default="camera_front_wide_120fov")
    parser.add_argument("--reference-start", type=int)
    parser.add_argument("--reference-end", type=int)
    parser.add_argument("--max-references", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-fov-deg", type=float, default=20.0)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--minimum-certainty", type=float, default=.05)
    parser.add_argument("--min-ray-angle-deg", type=float, default=1.0)
    parser.add_argument("--max-ray-separation-m", type=float, default=.35)
    parser.add_argument("--max-reprojection-error-px", type=float, default=3.0)
    parser.add_argument("--cuboid-margin-m", type=float, default=.15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    audit_roma_multiview(RomaMultiviewAuditOptions(**vars(args)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
