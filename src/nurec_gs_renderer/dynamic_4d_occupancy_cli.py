from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_4d_occupancy import Dynamic4DTrainOptions, train_dynamic_4d_occupancy


def _camera_ids(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("camera IDs must be unique and non-empty")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train raw-NCore actor-local 4D occupancy/transmittance field")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--reference-start", type=int, required=True)
    parser.add_argument("--reference-end", type=int, required=True)
    parser.add_argument("--camera-ids", type=_camera_ids)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--ray-samples", type=int, default=48)
    parser.add_argument("--actor-rays-per-step", type=int, default=384)
    parser.add_argument("--ring-rays-per-step", type=int, default=192)
    parser.add_argument("--anchor-points-per-step", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--ring-alpha-target", type=float, default=.08)
    parser.add_argument("--ring-hard-weight", type=float, default=2.0)
    parser.add_argument("--max-holdout-ring-alpha", type=float, default=.10)
    parser.add_argument("--max-holdout-actor-alpha-error", type=float, default=.15)
    parser.add_argument("--max-holdout-rgb-mae", type=float, default=.17)
    parser.add_argument("--validation-every", type=int, default=100)
    args = parser.parse_args(argv)
    train_dynamic_4d_occupancy(Dynamic4DTrainOptions(
        dataset_manifest=args.dataset_manifest, correspondence=args.correspondence, output=args.output,
        track_id=args.track_id, reference_start=args.reference_start, reference_end=args.reference_end,
        camera_ids=args.camera_ids, device=args.device, iterations=args.iterations, learning_rate=args.learning_rate,
        ray_samples=args.ray_samples, actor_rays_per_step=args.actor_rays_per_step, ring_rays_per_step=args.ring_rays_per_step,
        anchor_points_per_step=args.anchor_points_per_step, hidden_dim=args.hidden_dim, ring_alpha_target=args.ring_alpha_target,
        ring_hard_weight=args.ring_hard_weight, max_holdout_ring_alpha=args.max_holdout_ring_alpha,
        max_holdout_actor_alpha_error=args.max_holdout_actor_alpha_error, max_holdout_rgb_mae=args.max_holdout_rgb_mae,
        validation_every=args.validation_every,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
