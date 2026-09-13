from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_reconstruction_dataset import DynamicReconstructionDatasetOptions, build_dynamic_reconstruction_dataset
from .sky_build import DEFAULT_CAMERA_IDS


def _camera_ids(value: str) -> tuple[str, ...]:
    if value.strip().casefold() == "all":
        return DEFAULT_CAMERA_IDS
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("camera IDs must be a non-empty unique comma-separated list or 'all'")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build exact-FTheta cross-camera dynamic reconstruction supervision")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--camera-ids", type=_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--reference-camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int, help="exclusive offset inside chunk")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    parser.add_argument("--vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--person-probability-threshold", type=float, default=0.70)
    parser.add_argument("--person-margin-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-mask-pixels", type=int, default=24)
    parser.add_argument("--movement-threshold-m", type=float, default=0.50)
    parser.add_argument("--no-lidar-depth", action="store_true")
    parser.add_argument("--max-lidar-timestamp-delta-us", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    build_dynamic_reconstruction_dataset(DynamicReconstructionDatasetOptions(
        ncore_path=args.ncore_path, segformer_model_dir=args.segformer_model_dir, output=args.output,
        chunk_index=args.chunk_index, camera_ids=args.camera_ids, reference_camera_id=args.reference_camera_id,
        frame_start=args.frame_start, frame_end=args.frame_end, stride=args.stride,
        validation_stride=args.validation_stride, device=args.device, cuboid_margin_pixels=args.cuboid_margin_pixels,
        vehicle_probability_threshold=args.vehicle_probability_threshold, vehicle_margin_threshold=args.vehicle_margin_threshold,
        person_probability_threshold=args.person_probability_threshold, person_margin_threshold=args.person_margin_threshold,
        minimum_mask_pixels=args.minimum_mask_pixels, movement_threshold_m=args.movement_threshold_m,
        write_lidar_depth=not args.no_lidar_depth, max_lidar_timestamp_delta_us=args.max_lidar_timestamp_delta_us,
        resume=args.resume,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
