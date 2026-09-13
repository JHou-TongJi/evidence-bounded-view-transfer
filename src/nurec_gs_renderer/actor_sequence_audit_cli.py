from __future__ import annotations

import argparse
from pathlib import Path

from .actor_sequence_audit import ActorSequenceAuditOptions, audit_actor_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an actor-over-static target sequence without clip-specific assumptions")
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--static-rgb-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    args = parser.parse_args(argv)
    audit_actor_sequence(ActorSequenceAuditOptions(
        actor_output=args.actor_output, static_rgb_dir=args.static_rgb_dir, camera_id=args.camera_id,
        output=args.output, alpha_threshold=args.alpha_threshold,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
