from __future__ import annotations

import argparse
from pathlib import Path

from .deformable_actor_dataset import DeformableActorDatasetOptions, build_deformable_actor_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build short-window time-conditioned source supervision for a deformable NCore actor")
    parser.add_argument("--ncore-path", type=Path, required=True); parser.add_argument("--dynamic-dataset-manifest", type=Path, required=True)
    parser.add_argument("--deformable-plan", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True); parser.add_argument("--source-camera", required=True)
    parser.add_argument("--reference-start", type=int); parser.add_argument("--reference-end", type=int)
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270); parser.add_argument("--horizontal-fov-deg", type=float, default=40.)
    parser.add_argument("--minimum-mask-pixels", type=int, default=800); parser.add_argument("--holdout-every", type=int, default=5); parser.add_argument("--max-observations", type=int); parser.add_argument("--rectify-device", default="cuda")
    args = parser.parse_args(argv)
    build_deformable_actor_dataset(DeformableActorDatasetOptions(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
