from __future__ import annotations

import argparse
from pathlib import Path

from .clean_static_masks import CleanStaticMaskOptions, build_clean_static_masks


def _csv_track_ids(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("instance track IDs must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("instance track IDs must be unique non-negative integers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build conservative NCore dynamic masks for clean InstantNuRec static inference")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--model", choices=("pa-front", "pa-multiview", "pq-front"), default="pa-front")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuboid-margin-pixels", type=int, default=12)
    parser.add_argument("--semantic-probability-threshold", type=float, default=0.70)
    parser.add_argument("--semantic-margin-threshold", type=float, default=0.10)
    parser.add_argument("--semantic-dilation-pixels", type=int, default=4)
    parser.add_argument("--movement-threshold-m", type=float, default=1.50,
                        help="mask vehicle cuboids only after this travel; 1.5m matches InstantNuRec's dynamic-track default")
    parser.add_argument(
        "--instance-track-ids", type=_csv_track_ids, default=(),
        help="target track IDs whose cuboid-attached vehicle semantic component is also excluded",
    )
    parser.add_argument("--instance-component-min-overlap-pixels", type=int, default=16)
    parser.add_argument("--instance-component-dilation-pixels", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume only a matching mask output with manifest.partial.json")
    args = parser.parse_args(argv)
    build_clean_static_masks(CleanStaticMaskOptions(
        ncore_path=args.ncore_path, segformer_model_dir=args.segformer_model_dir, output=args.output,
        chunk_index=args.chunk_index, model=args.model, device=args.device,
        cuboid_margin_pixels=args.cuboid_margin_pixels,
        semantic_probability_threshold=args.semantic_probability_threshold,
        semantic_margin_threshold=args.semantic_margin_threshold,
        semantic_dilation_pixels=args.semantic_dilation_pixels, movement_threshold_m=args.movement_threshold_m,
        instance_track_ids=args.instance_track_ids,
        instance_component_min_overlap_pixels=args.instance_component_min_overlap_pixels,
        instance_component_dilation_pixels=args.instance_component_dilation_pixels,
        resume=args.resume,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
