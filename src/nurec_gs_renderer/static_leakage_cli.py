from __future__ import annotations

import argparse
from pathlib import Path

from .ftheta_actor_optimize_cli import _csv_ints, _csv_strings
from .static_leakage import DEFAULT_CAMERA_IDS, StaticLeakageOptions, build_multiview_static_leakage_correction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a conservative multi-view static dynamic-leakage opacity sidecar")
    for name in ("ply", "ncore-path", "actor-dataset-manifest", "segformer-model-dir", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--reference-frame-indices", required=True, type=_csv_ints)
    parser.add_argument("--camera-ids", type=_csv_strings, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuboid-margin-m", type=float, default=0.10); parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    parser.add_argument("--static-depth-tolerance-m", type=float, default=0.50); parser.add_argument("--lidar-depth-tolerance-m", type=float, default=0.75)
    parser.add_argument("--min-view-support", type=int, default=3); parser.add_argument("--min-lidar-support", type=int, default=0)
    parser.add_argument("--opacity-scale", type=float, default=0.50)
    args = parser.parse_args(argv)
    build_multiview_static_leakage_correction(StaticLeakageOptions(
        ply_path=args.ply, ncore_path=args.ncore_path, actor_dataset_manifest=args.actor_dataset_manifest,
        segformer_model_dir=args.segformer_model_dir, output=args.output, track_id=args.track_id,
        reference_frame_indices=args.reference_frame_indices, camera_ids=args.camera_ids, width=args.width,
        height=args.height, device=args.device, cuboid_margin_m=args.cuboid_margin_m,
        cuboid_margin_pixels=args.cuboid_margin_pixels, static_depth_tolerance_m=args.static_depth_tolerance_m,
        lidar_depth_tolerance_m=args.lidar_depth_tolerance_m, min_view_support=args.min_view_support,
        min_lidar_support=args.min_lidar_support, opacity_scale=args.opacity_scale,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
