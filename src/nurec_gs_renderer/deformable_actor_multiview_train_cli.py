from __future__ import annotations
import argparse
from pathlib import Path
from .deformable_actor_multiview_train import DeformableActorMultiviewTrainOptions, train_deformable_actor_multiview
def main(argv: list[str] | None=None)->int:
 p=argparse.ArgumentParser(description="Train a cross-camera canonical deformable neural-surface pilot"); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--iterations",type=int,default=1000); p.add_argument("--device",default="cuda"); p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--samples-per-ray",type=int,default=24); p.add_argument("--rays-per-step",type=int,default=1024); p.add_argument("--learning-rate",type=float,default=.002); p.add_argument("--validation-every",type=int,default=100); a=p.parse_args(argv); train_deformable_actor_multiview(DeformableActorMultiviewTrainOptions(**vars(a))); return 0
if __name__=="__main__": raise SystemExit(main())
