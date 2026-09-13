from __future__ import annotations
import argparse
from pathlib import Path
from .two_dgs_actor_dataset import TwoDgsActorDatasetOptions, build_two_dgs_actor_dataset

def _csv(value: str) -> tuple[str, ...]: return tuple(part.strip() for part in value.split(",") if part.strip())
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a rigid actor-local canonical 2DGS proxy dataset")
    parser.add_argument("--ncore-path", required=True, type=Path); parser.add_argument("--dynamic-dataset-manifest", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--camera-ids", type=_csv, default=()); parser.add_argument("--holdout-camera-ids", type=_csv, default=()); parser.add_argument("--reference-start", type=int); parser.add_argument("--reference-end", type=int)
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270); parser.add_argument("--horizontal-fov-deg", type=float, default=45.0); parser.add_argument("--minimum-mask-pixels", type=int, default=1000); parser.add_argument("--holdout-every", type=int, default=6); parser.add_argument("--initial-surface-spacing-m", type=float, default=.35); parser.add_argument("--max-observations", type=int, default=120); parser.add_argument("--rectify-device", default="cuda")
    args = parser.parse_args(argv); build_two_dgs_actor_dataset(TwoDgsActorDatasetOptions(**vars(args))); return 0
if __name__ == "__main__": raise SystemExit(main())
