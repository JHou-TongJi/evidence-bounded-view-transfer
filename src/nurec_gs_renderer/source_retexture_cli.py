from __future__ import annotations

import argparse
from pathlib import Path

from .source_retexture import DEFAULT_SOURCE_CAMERAS, SourceRetextureOptions, retexture_static_render


def _cameras(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("source cameras must be non-empty and unique")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproject source-camera texture detail onto a completed static target render.")
    parser.add_argument("--static-render-dir", type=Path, required=True)
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-cameras", type=_cameras, default=DEFAULT_SOURCE_CAMERAS)
    parser.add_argument("--dynamic-mask-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-alpha", type=float, default=.90)
    parser.add_argument("--maximum-depth-m", type=float, default=150.0)
    parser.add_argument("--minimum-view-cosine", type=float, default=.80)
    parser.add_argument("--color-blend", type=float, default=.35)
    parser.add_argument("--detail-strength", type=float, default=.65)
    parser.add_argument("--detail-blur-radius", type=float, default=1.2)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    args = parser.parse_args(argv)
    retexture_static_render(SourceRetextureOptions(
        static_render_root=args.static_render_dir, ncore_path=args.ncore_path, output_root=args.output_dir,
        source_camera_ids=args.source_cameras, dynamic_mask_manifest=args.dynamic_mask_manifest, device=args.device,
        minimum_alpha=args.minimum_alpha, maximum_depth_m=args.maximum_depth_m, minimum_view_cosine=args.minimum_view_cosine,
        color_blend=args.color_blend, detail_strength=args.detail_strength, detail_blur_radius=args.detail_blur_radius,
        frame_start=args.frame_start, frame_end=args.frame_end,
    ))
    print(f"Wrote source-guided static retexture: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
