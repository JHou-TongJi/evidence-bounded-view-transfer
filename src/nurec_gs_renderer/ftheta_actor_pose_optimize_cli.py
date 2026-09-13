from __future__ import annotations

import argparse
from pathlib import Path

from .ftheta_actor_optimize import DIRECT_DEFAULT_CAMERAS
from .ftheta_actor_optimize_cli import _csv_ints, _csv_strings
from .ftheta_actor_pose_optimize import FThetaActorPoseOptimizeOptions, optimize_ftheta_actor_poses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pose-only native FTheta/rolling-shutter canonical actor refinement")
    for name in ("static-ply", "initial-actors", "actor-dataset-manifest", "ncore-path", "segformer-model-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frame-indices", type=_csv_ints, required=True)
    parser.add_argument("--validation-frame-indices", type=_csv_ints, default=())
    parser.add_argument("--pose-knot-frame-indices", type=_csv_ints, required=True)
    parser.add_argument("--camera-ids", type=_csv_strings, default=DIRECT_DEFAULT_CAMERAS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--max-gaussians-per-actor", type=int, default=12000); parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--temporal-rgb-weight", type=float, default=1.0)
    parser.add_argument("--pose-translation-limit-m", type=float, default=0.25)
    parser.add_argument("--pose-rotation-limit-deg", type=float, default=4.0)
    parser.add_argument("--pose-prior-weight", type=float, default=0.02)
    parser.add_argument("--pose-velocity-weight", type=float, default=0.10)
    parser.add_argument("--pose-acceleration-weight", type=float, default=0.20)
    args = parser.parse_args(argv)
    optimize_ftheta_actor_poses(FThetaActorPoseOptimizeOptions(
        static_ply=args.static_ply, initial_actors=args.initial_actors, actor_dataset_manifest=args.actor_dataset_manifest,
        ncore_path=args.ncore_path, segformer_model_dir=args.segformer_model_dir, output=args.output,
        frame_indices=args.frame_indices, validation_frame_indices=args.validation_frame_indices, pose_knot_frame_indices=args.pose_knot_frame_indices,
        camera_ids=args.camera_ids, device=args.device, width=args.width, height=args.height,
        max_gaussians_per_actor=args.max_gaussians_per_actor, iterations=args.iterations, learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every, temporal_rgb_weight=args.temporal_rgb_weight,
        pose_translation_limit_m=args.pose_translation_limit_m, pose_rotation_limit_deg=args.pose_rotation_limit_deg,
        pose_prior_weight=args.pose_prior_weight, pose_velocity_weight=args.pose_velocity_weight,
        pose_acceleration_weight=args.pose_acceleration_weight,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
