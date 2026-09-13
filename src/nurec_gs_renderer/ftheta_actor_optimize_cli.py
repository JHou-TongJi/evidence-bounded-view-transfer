from __future__ import annotations

import argparse
from pathlib import Path

from .ftheta_actor_optimize import DIRECT_DEFAULT_CAMERAS, FThetaActorOptimizeOptions, optimize_ftheta_actors


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer frame indices") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("frame indices must be non-negative and non-empty")
    return values


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated camera list")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimise canonical vehicle Gaussians with native NCore FTheta/rolling-shutter supervision")
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--initial-actors", type=Path, required=True)
    parser.add_argument("--actor-dataset-manifest", type=Path, required=True)
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-indices", type=_csv_ints, default=())
    parser.add_argument("--validation-frame-indices", type=_csv_ints, default=())
    parser.add_argument("--clean-static-mask-slots", type=_csv_ints, default=(),
                        help="Use exact source frames from these clean-static manifest slots instead of --frame-indices.")
    parser.add_argument("--validation-clean-static-mask-slots", type=_csv_ints, default=())
    parser.add_argument("--camera-ids", type=_csv_strings, default=DIRECT_DEFAULT_CAMERAS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--max-gaussians-per-actor", type=int, default=12000)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--clean-static-mask-dir", type=Path,
                        help="Exact NCore source mask manifest used to gate dynamic supervision.")
    parser.add_argument("--min-source-mask-coverage", type=float, default=0.0,
                        help="Require this fraction of a semantic actor ROI to be masked in the clean-static input.")
    parser.add_argument("--min-lidar-pixels", type=int, default=0,
                        help="Require this many sparse LiDAR depth pixels in a source-visible actor ROI.")
    parser.add_argument("--max-lidar-timestamp-delta-us", type=int, default=100000)
    parser.add_argument("--camera-appearance", action="store_true",
                        help="Enable bounded per-camera affine-in-time RGB residuals for diagnostic optimisation.")
    parser.add_argument("--camera-appearance-max-delta", type=float, default=0.12)
    args = parser.parse_args(argv)
    optimize_ftheta_actors(FThetaActorOptimizeOptions(
        static_ply=args.static_ply, initial_actors=args.initial_actors, actor_dataset_manifest=args.actor_dataset_manifest,
        ncore_path=args.ncore_path, segformer_model_dir=args.segformer_model_dir, output=args.output,
        frame_indices=args.frame_indices, validation_frame_indices=args.validation_frame_indices,
        clean_static_mask_slots=args.clean_static_mask_slots,
        validation_clean_static_mask_slots=args.validation_clean_static_mask_slots, camera_ids=args.camera_ids,
        device=args.device, width=args.width, height=args.height, max_gaussians_per_actor=args.max_gaussians_per_actor,
        iterations=args.iterations, learning_rate=args.learning_rate, checkpoint_every=args.checkpoint_every,
        resume_checkpoint=args.resume_checkpoint, clean_static_mask_dir=args.clean_static_mask_dir,
        min_source_mask_coverage=args.min_source_mask_coverage, min_lidar_pixels=args.min_lidar_pixels,
        max_lidar_timestamp_delta_us=args.max_lidar_timestamp_delta_us,
        camera_appearance_enabled=args.camera_appearance,
        camera_appearance_max_delta=args.camera_appearance_max_delta,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
