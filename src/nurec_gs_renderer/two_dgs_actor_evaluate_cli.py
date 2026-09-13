from __future__ import annotations
import argparse
from pathlib import Path
from .two_dgs_actor_evaluate import TwoDgsActorEvaluateOptions, evaluate_two_dgs_actor
def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="Evaluate canonical actor 2DGS on explicit held-out actor masks")
    p.add_argument("--two-dgs-root", required=True, type=Path); p.add_argument("--source-path", required=True, type=Path); p.add_argument("--model-path", required=True, type=Path); p.add_argument("--output", required=True, type=Path); p.add_argument("--iteration", type=int, default=-1); p.add_argument("--device", default="cuda"); p.add_argument("--save-previews", type=int, default=3)
    args=p.parse_args(argv); evaluate_two_dgs_actor(TwoDgsActorEvaluateOptions(**vars(args))); return 0
if __name__ == "__main__": raise SystemExit(main())
