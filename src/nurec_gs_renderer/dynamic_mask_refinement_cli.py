from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_mask_refinement import DynamicMaskRefinementOptions, refine_dynamic_instance_masks
from .sky_build import DEFAULT_CAMERA_IDS


def _camera_ids(value: str) -> tuple[str, ...]:
    if value.strip().casefold() == "all":
        return DEFAULT_CAMERA_IDS
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    if not output or len(set(output)) != len(output):
        raise argparse.ArgumentTypeError("camera IDs must be unique and non-empty")
    return output


def _track_ids(value: str) -> tuple[int, ...]:
    try:
        output = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track IDs must be comma-separated integers") from exc
    if not output or len(set(output)) != len(output):
        raise argparse.ArgumentTypeError("track IDs must be unique and non-empty")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservative SAM2 bidirectional refinement of dynamic instance masks")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncore-path", type=Path)
    parser.add_argument("--camera-ids", type=_camera_ids, default=None)
    parser.add_argument("--track-ids", type=_track_ids, default=None)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-stride", type=int, default=15)
    parser.add_argument("--probability-threshold", type=float, default=.50)
    parser.add_argument("--minimum-directional-iou", type=float, default=.35)
    parser.add_argument("--minimum-seed-iou", type=float, default=.05)
    parser.add_argument("--minimum-mask-pixels", type=int, default=24)
    parser.add_argument("--maximum-cuboid-fill-fraction", type=float, default=.92)
    parser.add_argument("--cuboid-margin-pixels", type=int)
    parser.add_argument("--max-vision-features-cache-size", type=int, default=2)
    parser.add_argument("--write-previews", action="store_true")
    parser.add_argument("--preview-stride", type=int, default=15)
    parser.add_argument("--no-lidar-depth", action="store_true")
    parser.add_argument("--max-lidar-timestamp-delta-us", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    refine_dynamic_instance_masks(DynamicMaskRefinementOptions(
        dataset_manifest=args.dataset_manifest, model_dir=args.model_dir, output=args.output, ncore_path=args.ncore_path,
        camera_ids=args.camera_ids, track_ids=args.track_ids, frame_start=args.frame_start, frame_end=args.frame_end,
        device=args.device, anchor_stride=args.anchor_stride, probability_threshold=args.probability_threshold,
        minimum_directional_iou=args.minimum_directional_iou, minimum_seed_iou=args.minimum_seed_iou,
        minimum_mask_pixels=args.minimum_mask_pixels, maximum_cuboid_fill_fraction=args.maximum_cuboid_fill_fraction,
        cuboid_margin_pixels=args.cuboid_margin_pixels, max_vision_features_cache_size=args.max_vision_features_cache_size,
        write_previews=args.write_previews, preview_stride=args.preview_stride, resume=args.resume,
        write_lidar_depth=not args.no_lidar_depth, max_lidar_timestamp_delta_us=args.max_lidar_timestamp_delta_us,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
