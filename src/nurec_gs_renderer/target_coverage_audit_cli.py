"""Command-line entry point for target-rig coverage auditing."""
from __future__ import annotations

import argparse
from pathlib import Path

from .target_coverage_audit import TargetCoverageAuditOptions, audit_target_rig_coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a rendered named target-camera rig into geometry / sky / unknown coverage maps."
    )
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--sky-valid-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-geometric-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-lower-half-geometric-fraction", type=float, default=0.35)
    parser.add_argument("--maximum-unknown-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    report = audit_target_rig_coverage(TargetCoverageAuditOptions(
        render_root=args.render_root, output=args.output, alpha_threshold=args.alpha_threshold,
        sky_valid_threshold=args.sky_valid_threshold,
        minimum_geometric_fraction=args.minimum_geometric_fraction,
        minimum_lower_half_geometric_fraction=args.minimum_lower_half_geometric_fraction,
        maximum_unknown_fraction=args.maximum_unknown_fraction,
    ))
    print(
        f"Wrote target-rig coverage audit: {args.output} "
        f"({report['camera_count']} cameras; all_geometry_ready="
        f"{report['all_cameras_geometric_road_ready']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
