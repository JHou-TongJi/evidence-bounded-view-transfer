from __future__ import annotations
import argparse
from pathlib import Path
from .two_dgs_actor_batch import ActorBatchOptions, run_actor_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-plan/build/train rigid canonical 2DGS actor experts")
    parser.add_argument("--ncore-path", type=Path, required=True); parser.add_argument("--dynamic-dataset-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "build", "train"), required=True); parser.add_argument("--two-dgs-root", type=Path)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--gpu-id")
    parser.add_argument("--width", type=int, default=480); parser.add_argument("--height", type=int, default=270); parser.add_argument("--horizontal-fov-deg", type=float, default=45.)
    parser.add_argument("--min-observations", type=int, default=18); parser.add_argument("--min-source-mask-pixels", type=int, default=4000); parser.add_argument("--max-experts-per-track", type=int, default=2); parser.add_argument("--max-tracks", type=int); parser.add_argument("--max-observations-per-expert", type=int, default=100)
    parser.add_argument("--quality-window-frames", type=int, help="Train from the highest-mask contiguous source window instead of a full-trajectory subsample")
    parser.add_argument("--iterations", type=int, default=1750); parser.add_argument("--min-alpha-iou", type=float, default=.70); parser.add_argument("--max-masked-rgb-mae", type=float, default=.10); parser.add_argument("--min-alpha-area-ratio", type=float, default=.75); parser.add_argument("--max-alpha-area-ratio", type=float, default=1.30)
    args=parser.parse_args(argv); print(run_actor_batch(ActorBatchOptions(**vars(args))), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
