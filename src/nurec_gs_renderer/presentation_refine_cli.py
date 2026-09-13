"""CLI entry point for RGB-only L4 presentation refinement."""
from __future__ import annotations

import argparse
from pathlib import Path

from .presentation_refine import PresentationRefineOptions, refine_render_presentation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an RGB-only, lower-geometry bilateral display derivative of a completed render run."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lower-start-ratio", type=float, default=0.43)
    parser.add_argument("--feather-ratio", type=float, default=0.12)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.78)
    parser.add_argument("--diameter", type=int, default=9)
    parser.add_argument("--sigma-color", type=float, default=35.0)
    parser.add_argument("--sigma-space", type=float, default=7.0)
    args = parser.parse_args(argv)
    report = refine_render_presentation(PresentationRefineOptions(
        input_root=args.input_dir,
        output_root=args.output_dir,
        lower_start_ratio=args.lower_start_ratio,
        feather_ratio=args.feather_ratio,
        alpha_threshold=args.alpha_threshold,
        strength=args.strength,
        diameter=args.diameter,
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
    ))
    print(f"Wrote RGB-only presentation refinement: {args.output_dir} ({len(report['units'])} unit(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
