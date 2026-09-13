from __future__ import annotations

import argparse
from pathlib import Path

from .actor_target_visibility_audit import ActorTargetVisibilityAuditOptions, audit_actor_target_visibility


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit target-view 2DGS actor footprint continuity without modifying a sequence")
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=.50)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--max-minor-component-fraction", type=float, default=.10)
    parser.add_argument("--max-adjacent-area-ratio", type=float, default=2.5)
    args = parser.parse_args(argv)
    audit_actor_target_visibility(ActorTargetVisibilityAuditOptions(
        actor_output=args.actor_output, camera_id=args.camera_id, output=args.output,
        alpha_threshold=args.alpha_threshold, min_component_pixels=args.min_component_pixels,
        max_minor_component_fraction=args.max_minor_component_fraction,
        max_adjacent_area_ratio=args.max_adjacent_area_ratio,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
