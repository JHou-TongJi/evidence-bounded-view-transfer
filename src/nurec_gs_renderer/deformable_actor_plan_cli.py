from __future__ import annotations
import argparse
from pathlib import Path
from .deformable_actor_plan import DeformableActorPlanOptions, build_deformable_actor_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan short trusted supervision windows for deformable actors")
    parser.add_argument("--dynamic-dataset-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-mask-pixels", type=int, default=800); parser.add_argument("--min-observations", type=int, default=12); parser.add_argument("--window-frames", type=int, default=32); parser.add_argument("--max-windows-per-track", type=int, default=2)
    args = parser.parse_args(argv); build_deformable_actor_plan(DeformableActorPlanOptions(**vars(args))); return 0


if __name__ == "__main__": raise SystemExit(main())
