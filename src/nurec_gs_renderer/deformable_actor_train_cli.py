from __future__ import annotations

import argparse
from pathlib import Path

from .deformable_actor_train import DeformableActorTrainOptions, train_deformable_actor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a source-view time/camera-conditioned deformable actor dynamic-plane proxy")
    parser.add_argument("--dataset", type=Path, action="append", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000); parser.add_argument("--device", default="cuda"); parser.add_argument("--atlas-size", type=int, default=256); parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--actor-samples", type=int, default=2048); parser.add_argument("--ring-samples", type=int, default=1024); parser.add_argument("--ring-loss-weight", type=float, default=.75); parser.add_argument("--learning-rate", type=float, default=.002); parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--max-holdout-rgb-mae", type=float, default=.12); parser.add_argument("--min-holdout-alpha-iou", type=float, default=.60); parser.add_argument("--min-holdout-alpha-area-ratio", type=float, default=.70); parser.add_argument("--max-holdout-alpha-area-ratio", type=float, default=1.30); parser.add_argument("--max-holdout-ring-alpha", type=float, default=.15)
    args = parser.parse_args(argv); train_deformable_actor(DeformableActorTrainOptions(datasets=tuple(args.dataset), output=args.output, iterations=args.iterations, device=args.device, atlas_size=args.atlas_size, hidden_dim=args.hidden_dim, actor_samples=args.actor_samples, ring_samples=args.ring_samples, ring_loss_weight=args.ring_loss_weight, learning_rate=args.learning_rate, validation_every=args.validation_every, max_holdout_rgb_mae=args.max_holdout_rgb_mae, min_holdout_alpha_iou=args.min_holdout_alpha_iou, min_holdout_alpha_area_ratio=args.min_holdout_alpha_area_ratio, max_holdout_alpha_area_ratio=args.max_holdout_alpha_area_ratio, max_holdout_ring_alpha=args.max_holdout_ring_alpha)); return 0


if __name__ == "__main__": raise SystemExit(main())
