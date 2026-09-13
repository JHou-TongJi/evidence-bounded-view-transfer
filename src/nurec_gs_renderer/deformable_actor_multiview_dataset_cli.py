from __future__ import annotations

import argparse
from pathlib import Path

from .deformable_actor_multiview_dataset import DeformableActorMultiviewDatasetOptions, build_deformable_actor_multiview_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build all automatic deformable multi-camera source datasets")
    parser.add_argument("--ncore-path", type=Path, required=True); parser.add_argument("--dynamic-dataset-manifest", type=Path, required=True); parser.add_argument("--multiview-plan", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270); parser.add_argument("--horizontal-fov-deg", type=float, default=40.0); parser.add_argument("--minimum-mask-pixels", type=int, default=800); parser.add_argument("--rectify-device", default="cuda")
    args = parser.parse_args(argv); build_deformable_actor_multiview_dataset(DeformableActorMultiviewDatasetOptions(**vars(args))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
