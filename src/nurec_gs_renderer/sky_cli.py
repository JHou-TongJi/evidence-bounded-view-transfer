from __future__ import annotations

import argparse
from pathlib import Path

from .sky_build import DEFAULT_CAMERA_IDS, SEGFORMER_MODEL_PAGE, SkyBuildOptions, build_sky_cubemap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-sky-build",
        description="Build a per-chunk world-direction sky cubemap from NCore cameras.",
        epilog=f"SegFormer weights are not downloaded automatically. Model page: {SEGFORMER_MODEL_PAGE}",
    )
    parser.add_argument("--ncore-path", required=True, type=Path, help="NCore V4 sequence JSON")
    parser.add_argument("--chunk-index", required=True, type=int)
    parser.add_argument("--model-dir", required=True, type=Path, help="Complete local SegFormer snapshot")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz sky cubemap")
    parser.add_argument(
        "--temporal-output",
        type=Path,
        help="Optional .json manifest plus one locally anchored cubemap per chunk slot",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for the removable ~1.1 GiB temporal fusion memmap",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument("--working-width", type=int, default=960)
    parser.add_argument("--working-height", type=int, default=540)
    parser.add_argument("--sky-threshold", type=float, default=0.85)
    parser.add_argument("--sky-margin-threshold", type=float, default=0.30)
    parser.add_argument("--erosion-pixels", type=int, default=6)
    parser.add_argument("--edge-exclusion-pixels", type=int, default=4)
    parser.add_argument("--edge-gradient-threshold", type=float, default=0.12)
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        default=DEFAULT_CAMERA_IDS,
        help="NCore camera ids; defaults to the seven PhysicalAI cameras",
    )
    parser.add_argument("--reference-camera-id", default="camera_front_wide_120fov")
    parser.add_argument(
        "--reference-slot",
        type=int,
        help="Primary temporal slot; defaults to the middle slot. Use 0 to match the first source frame.",
    )
    parser.add_argument("--min-observation-support", type=int, default=2)
    parser.add_argument("--color-outlier-log-threshold", type=float, default=0.35)
    parser.add_argument("--coherence-block-size", type=int, default=16)
    parser.add_argument("--reference-camera-priority", type=float, default=0.25)
    parser.add_argument(
        "--no-exposure-normalization",
        action="store_true",
        help="Disable robust linear-light per-camera exposure matching",
    )
    parser.add_argument("--exposure-gain-min", type=float, default=0.80)
    parser.add_argument("--exposure-gain-max", type=float, default=1.25)
    parser.add_argument("--exposure-min-overlap", type=int, default=512)
    parser.add_argument("--exposure-compensation-ev", type=float, default=0.0)
    parser.add_argument(
        "--geometry-mask-dir",
        type=Path,
        help="Optional source-view geometry alpha masks: <camera>/<frame:06d>.png",
    )
    parser.add_argument("--geometry-mask-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_sky_cubemap(
        SkyBuildOptions(
            ncore_path=args.ncore_path,
            chunk_index=args.chunk_index,
            model_dir=args.model_dir,
            output=args.output,
            temporal_output=args.temporal_output,
            cache_dir=args.cache_dir,
            device=args.device,
            face_size=args.face_size,
            working_width=args.working_width,
            working_height=args.working_height,
            sky_threshold=args.sky_threshold,
            sky_margin_threshold=args.sky_margin_threshold,
            erosion_pixels=args.erosion_pixels,
            edge_exclusion_pixels=args.edge_exclusion_pixels,
            edge_gradient_threshold=args.edge_gradient_threshold,
            camera_ids=tuple(args.camera_ids),
            reference_camera_id=args.reference_camera_id,
            reference_slot=args.reference_slot,
            min_observation_support=args.min_observation_support,
            color_outlier_log_threshold=args.color_outlier_log_threshold,
            coherence_block_size=args.coherence_block_size,
            reference_camera_priority=args.reference_camera_priority,
            normalize_exposure=not args.no_exposure_normalization,
            exposure_gain_min=args.exposure_gain_min,
            exposure_gain_max=args.exposure_gain_max,
            exposure_min_overlap=args.exposure_min_overlap,
            exposure_compensation_ev=args.exposure_compensation_ev,
            geometry_mask_dir=args.geometry_mask_dir,
            geometry_mask_threshold=args.geometry_mask_threshold,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
