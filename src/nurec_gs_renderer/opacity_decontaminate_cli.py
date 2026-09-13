from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic import load_dynamic_gaussians
from .opacity_decontaminate import (
    DynamicProximityOpacityOptions,
    build_dynamic_proximity_opacity_correction,
)
from .ply_io import load_gaussian_ply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-opacity-decontaminate",
        description=(
            "Build a reversible static-opacity sidecar from proximity to InstantNuRec "
            "dynamic source Gaussians."
        ),
    )
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--dynamic-gaussians", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-distance-m", type=float, default=0.15)
    parser.add_argument("--max-color-distance", type=float, default=0.10)
    parser.add_argument("--max-static-scale-m", type=float, default=0.15)
    parser.add_argument("--opacity-scale", type=float, default=0.0)
    parser.add_argument("--include-road", action="store_true")
    parser.add_argument("--include-sky", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = DynamicProximityOpacityOptions(
        ply_path=args.ply,
        dynamic_path=args.dynamic_gaussians,
        output=args.output,
        max_distance_m=args.max_distance_m,
        max_color_distance=args.max_color_distance,
        max_static_scale_m=args.max_static_scale_m,
        opacity_scale=args.opacity_scale,
        exclude_road=not args.include_road,
        exclude_sky=not args.include_sky,
    )
    scene = load_gaussian_ply(options.ply_path)
    dynamic_scene = load_dynamic_gaussians(options.dynamic_path)
    print(
        f"Loaded {scene.count:,} static and {dynamic_scene.count:,} dynamic Gaussians",
        flush=True,
    )
    correction = build_dynamic_proximity_opacity_correction(scene, dynamic_scene, options)
    print(
        f"Wrote opacity correction for {correction.count:,} static Gaussians to {options.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
