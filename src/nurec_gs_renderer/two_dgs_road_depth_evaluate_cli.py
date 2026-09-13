"""CLI for sparse LiDAR road-depth evaluation of an official 2DGS checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from .two_dgs_road_depth_evaluate import TwoDgsRoadDepthEvaluateOptions, evaluate_two_dgs_road_depth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate held-out sparse LiDAR road depth for an official 2DGS model")
    parser.add_argument("--two-dgs-root", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "holdout"), default="holdout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-views", type=int)
    args = parser.parse_args(argv)
    evaluate_two_dgs_road_depth(TwoDgsRoadDepthEvaluateOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
