from __future__ import annotations

import argparse
from pathlib import Path

from .color_correction import GaussianColorCorrection, bake_color_correction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-color-bake",
        description="Bake a validated Gaussian color-correction sidecar into a new PLY.",
    )
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--correction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    correction = GaussianColorCorrection.load(args.correction)
    print(f"Loaded correction for {correction.count:,} / {correction.gaussian_count:,} Gaussians")
    bake_color_correction(args.ply, args.correction, args.output)
    print(f"Wrote corrected PLY to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
