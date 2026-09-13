"""CLI for the bounded native-FTheta static scene optimisation pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

from .static_scene_optimize import DEFAULT_CAMERA_IDS, StaticSceneOptimizeOptions, optimize_static_scene


def _indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("indices must be unique non-negative integers")
    return result


def _cameras(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("camera IDs must be non-empty and unique")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded native NCore FTheta/rolling static-Gaussian optimisation pilot")
    parser.add_argument("--initial-ply", type=Path, required=True)
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--dynamic-mask-manifest", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output-ply", type=Path)
    outputs.add_argument("--color-correction-output", type=Path)
    parser.add_argument("--initial-color-correction", type=Path)
    parser.add_argument("--train-reference-indices", type=_indices, required=True)
    parser.add_argument("--validation-reference-indices", type=_indices, required=True)
    parser.add_argument("--camera-ids", type=_cameras, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--trainable-gaussians", type=int, default=100_000)
    parser.add_argument("--selection-mode", choices=("opacity", "residual"), default="opacity")
    parser.add_argument("--optimization-mode", choices=("joint", "color_only", "geometry_lidar", "geometry_lidar_densify"), default="joint")
    parser.add_argument("--lidar-support-radius-m", type=float, default=.50)
    parser.add_argument("--minimum-lidar-cameras", type=int, default=2)
    parser.add_argument("--minimum-lidar-references", type=int, default=2)
    parser.add_argument("--seed-voxel-size-m", type=float, default=.10)
    parser.add_argument("--seed-residual-support-radius-m", type=float, default=.30)
    parser.add_argument("--seed-scale-m", type=float, default=.08)
    parser.add_argument("--maximum-seed-gaussians", type=int, default=2048)
    parser.add_argument("--minimum-seed-gaussians", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-every", type=int, default=25)
    args = parser.parse_args(argv)
    optimize_static_scene(StaticSceneOptimizeOptions(
        initial_ply=args.initial_ply, ncore_path=args.ncore_path, dynamic_mask_manifest=args.dynamic_mask_manifest,
        segformer_model_dir=args.segformer_model_dir, output_ply=args.output_ply,
        train_reference_indices=args.train_reference_indices, validation_reference_indices=args.validation_reference_indices,
        color_correction_output=args.color_correction_output, initial_color_correction=args.initial_color_correction,
        camera_ids=args.camera_ids, device=args.device, width=args.width, height=args.height,
        trainable_gaussians=args.trainable_gaussians, selection_mode=args.selection_mode,
        optimization_mode=args.optimization_mode, lidar_support_radius_m=args.lidar_support_radius_m,
        minimum_lidar_cameras=args.minimum_lidar_cameras, minimum_lidar_references=args.minimum_lidar_references,
        seed_voxel_size_m=args.seed_voxel_size_m, seed_residual_support_radius_m=args.seed_residual_support_radius_m,
        seed_scale_m=args.seed_scale_m, maximum_seed_gaussians=args.maximum_seed_gaussians,
        minimum_seed_gaussians=args.minimum_seed_gaussians,
        iterations=args.iterations, learning_rate=args.learning_rate,
        validation_every=args.validation_every,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
