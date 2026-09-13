from __future__ import annotations

import argparse
from pathlib import Path

from .actor_expert_registry import ActorExpertRegistryOptions, build_actor_expert_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a reusable validated rigid-actor expert registry")
    parser.add_argument("--batch-manifest", type=Path, action="append", default=[])
    parser.add_argument("--expert-spec", type=Path, action="append", default=[], help="Historical independently evaluated actor candidates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-masked-rgb-mae", type=float, default=.07)
    parser.add_argument("--min-alpha-iou", type=float, default=.82)
    parser.add_argument("--min-alpha-area-ratio", type=float, default=.85)
    parser.add_argument("--max-alpha-area-ratio", type=float, default=1.20)
    parser.add_argument("--overlap-fraction", type=float, default=.50)
    args = parser.parse_args(argv)
    build_actor_expert_registry(ActorExpertRegistryOptions(
        batch_manifests=tuple(args.batch_manifest), expert_specs=tuple(args.expert_spec), output=args.output,
        max_masked_rgb_mae=args.max_masked_rgb_mae, min_alpha_iou=args.min_alpha_iou,
        min_alpha_area_ratio=args.min_alpha_area_ratio, max_alpha_area_ratio=args.max_alpha_area_ratio,
        overlap_fraction=args.overlap_fraction,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
