"""CLI for reversible quality-priority static renderer-output fusion."""
from __future__ import annotations

import argparse
from pathlib import Path

from .static_layer_fusion import StaticLayerFusionOptions, fuse_static_render_layers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prefer valid geometry from a fidelity PLY render; use a coverage PLY render only as a hard fallback."
    )
    parser.add_argument("--primary-render-root", required=True, type=Path)
    parser.add_argument("--fallback-render-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--primary-alpha-threshold", type=float, default=0.01)
    parser.add_argument(
        "--maximum-relative-depth-difference", type=float, default=None,
        help="If set, defer primary pixels to fallback when both valid depths differ by more than this relative amount.",
    )
    args = parser.parse_args(argv)
    report = fuse_static_render_layers(StaticLayerFusionOptions(
        primary_root=args.primary_render_root,
        fallback_root=args.fallback_render_root,
        output_root=args.output_dir,
        primary_alpha_threshold=args.primary_alpha_threshold,
        maximum_relative_depth_difference=args.maximum_relative_depth_difference,
    ))
    print(f"Wrote quality-priority static fusion: {args.output_dir} ({len(report['units'])} unit(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
