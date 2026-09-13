from __future__ import annotations

import argparse
from pathlib import Path

from .source_proxy_composite import SourceProxyCompositeOptions, composite_source_proxy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project one masked NCore actor image as a source-conditioned 2.5-D target display proxy.")
    parser.add_argument("--static-render-dir", required=True, type=Path)
    parser.add_argument("--dynamic-dataset-manifest", required=True, type=Path)
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--source-camera-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--maximum-timestamp-delta-us", type=int, default=40_000)
    parser.add_argument("--maximum-view-angle-deg", type=float, default=25.0)
    parser.add_argument("--max-source-pixels", type=int, default=100_000)
    parser.add_argument("--mask-component-mode", choices=("largest", "all"), default="largest",
                        help="keep one connected actor silhouette by default; 'all' is diagnostic only")
    parser.add_argument("--minimum-source-mask-fraction", type=float, default=.0,
                        help="fail closed when the selected source actor silhouette is smaller than this image fraction")
    parser.add_argument("--temporal-fade-frames", type=int, default=0,
                        help="linear fade-in/out at the boundaries of each consecutive accepted segment")
    parser.add_argument("--splat-radius-pixels", type=int, default=1)
    parser.add_argument("--alpha-dilation-pixels", type=int, default=1)
    parser.add_argument("--max-proxy-fraction", type=float, default=.25)
    parser.add_argument("--geometry-mode", choices=("plane", "cuboid_faces"), default="plane",
                        help="plane is the legacy image warp; cuboid_faces uses conservative tracked-box entrance depths")
    parser.add_argument("--cuboid-margin-m", type=float, default=.10,
                        help="small track-box tolerance used only by cuboid_faces")
    args = parser.parse_args(argv)
    report = composite_source_proxy(SourceProxyCompositeOptions(
        static_render_root=args.static_render_dir, dynamic_dataset_manifest=args.dynamic_dataset_manifest,
        ncore_path=args.ncore_path, output_root=args.output_dir, track_id=args.track_id,
        source_camera_id=args.source_camera_id, device=args.device, frame_start=args.frame_start, frame_end=args.frame_end,
        maximum_timestamp_delta_us=args.maximum_timestamp_delta_us, maximum_view_angle_deg=args.maximum_view_angle_deg,
        max_source_pixels=args.max_source_pixels, mask_component_mode=args.mask_component_mode,
        minimum_source_mask_fraction=args.minimum_source_mask_fraction, temporal_fade_frames=args.temporal_fade_frames,
        splat_radius_pixels=args.splat_radius_pixels,
        alpha_dilation_pixels=args.alpha_dilation_pixels, max_proxy_fraction=args.max_proxy_fraction,
        geometry_mode=args.geometry_mode, cuboid_margin_m=args.cuboid_margin_m,
    ))
    print(f"Wrote source-conditioned 2.5-D proxy: {args.output_dir} ({len(report['units'])} unit(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
