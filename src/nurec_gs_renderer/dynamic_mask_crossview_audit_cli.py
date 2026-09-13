from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_mask_crossview_audit import DynamicMaskCrossviewAuditOptions, audit_dynamic_mask_crossview


def _track_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track IDs must be comma-separated integers") from exc
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("track IDs must be unique non-negative integers")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact NCore FTheta/rolling-shutter cross-view audit of SAM2-added masks")
    parser.add_argument("--refinement-manifest", type=Path, required=True)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", type=_track_ids, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-observations", type=int, default=0, help="0 evaluates every cross-view-observable SAM2 addition")
    parser.add_argument("--cuboid-margin-m", type=float, default=.25)
    parser.add_argument("--static-occluder-depth-margin-m", type=float, default=.30)
    parser.add_argument("--minimum-source-lidar-points", type=int, default=8)
    parser.add_argument("--minimum-target-lidar-points", type=int, default=4)
    parser.add_argument("--minimum-target-mask-fraction", type=float, default=.50)
    parser.add_argument("--static-width", type=int, default=480, help="static expected-depth render width")
    parser.add_argument("--static-height", type=int, default=270, help="static expected-depth render height")
    args = parser.parse_args(argv)
    audit_dynamic_mask_crossview(DynamicMaskCrossviewAuditOptions(
        refinement_manifest=args.refinement_manifest, static_ply=args.static_ply, output=args.output,
        track_ids=args.track_ids, device=args.device, max_source_observations=args.max_source_observations,
        cuboid_margin_m=args.cuboid_margin_m, static_occluder_depth_margin_m=args.static_occluder_depth_margin_m,
        minimum_source_lidar_points=args.minimum_source_lidar_points,
        minimum_target_lidar_points=args.minimum_target_lidar_points,
        minimum_target_mask_fraction=args.minimum_target_mask_fraction,
        static_width=args.static_width, static_height=args.static_height,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
