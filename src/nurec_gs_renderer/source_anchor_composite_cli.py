from __future__ import annotations

import argparse
from pathlib import Path

from .source_anchor_composite import SourceAnchorCompositeOptions, composite_source_anchored_dynamic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay source-mask + LiDAR supported NCore dynamic RGB onto target renders.")
    parser.add_argument("--static-render-dir", required=True, type=Path)
    parser.add_argument("--dynamic-dataset-manifest", required=True, type=Path)
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-timestamp-delta-us", type=int, default=40_000)
    parser.add_argument("--static-occlusion-tolerance-m", type=float, default=0.75)
    parser.add_argument("--splat-radius-pixels", type=int, default=2)
    parser.add_argument("--min-lidar-samples-per-instance", type=int, default=12)
    parser.add_argument("--maximum-depth-propagation-pixels", type=float, default=0.0,
                        help="Display-only nearest-LiDAR depth propagation inside an original instance mask; 0 keeps sparse samples.")
    parser.add_argument("--max-samples-per-instance", type=int, default=30_000)
    args = parser.parse_args(argv)
    report = composite_source_anchored_dynamic(SourceAnchorCompositeOptions(
        static_render_root=args.static_render_dir, dynamic_dataset_manifest=args.dynamic_dataset_manifest,
        ncore_path=args.ncore_path, output_root=args.output_dir, device=args.device,
        maximum_timestamp_delta_us=args.maximum_timestamp_delta_us,
        static_occlusion_tolerance_m=args.static_occlusion_tolerance_m,
        splat_radius_pixels=args.splat_radius_pixels,
        min_lidar_samples_per_instance=args.min_lidar_samples_per_instance,
        maximum_depth_propagation_pixels=args.maximum_depth_propagation_pixels,
        max_samples_per_instance=args.max_samples_per_instance,
    ))
    print(f"Wrote source-anchored target-view hybrid: {args.output_dir} ({len(report['units'])} unit(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
