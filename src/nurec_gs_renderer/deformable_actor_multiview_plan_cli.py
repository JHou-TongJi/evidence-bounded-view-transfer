from __future__ import annotations

import argparse
from pathlib import Path

from .deformable_actor_multiview_plan import DeformableActorMultiviewPlanOptions, build_deformable_actor_multiview_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatically select synchronous multi-camera deformable-actor windows")
    parser.add_argument("--dynamic-dataset-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mask-pixels", type=int, default=800); parser.add_argument("--min-synchronized-frames", type=int, default=8); parser.add_argument("--window-frames", type=int, default=32); parser.add_argument("--max-windows-per-track", type=int, default=2)
    parser.add_argument("--max-timestamp-spread-us", type=int, default=50_000); parser.add_argument("--min-camera-baseline-m", type=float, default=.30); parser.add_argument("--min-triangulation-angle-deg", type=float, default=5.0); parser.add_argument("--holdout-every", type=int, default=5)
    args = parser.parse_args(argv); build_deformable_actor_multiview_plan(DeformableActorMultiviewPlanOptions(**vars(args))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
