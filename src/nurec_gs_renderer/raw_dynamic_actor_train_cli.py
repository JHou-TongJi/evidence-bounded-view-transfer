from __future__ import annotations

import argparse
from pathlib import Path

from .raw_dynamic_actor_train import RawDynamicActorTrainOptions, train_raw_dynamic_actor


def _track_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track IDs must be comma-separated integers") from exc
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("track IDs must be unique non-negative integers")
    return result

def _camera_ids(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result): raise argparse.ArgumentTypeError("camera IDs must be non-empty and unique")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-train a rigid raw-NCore FTheta/rolling dynamic actor without static PLY")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", type=_track_ids, required=True)
    parser.add_argument("--initial-actors", type=Path, help="raw LiDAR multi-view canonical actor asset; avoids cuboid-shell initialization")
    parser.add_argument("--reference-start", type=int)
    parser.add_argument("--reference-end", type=int)
    parser.add_argument("--camera-ids", type=_camera_ids)
    parser.add_argument("--temporal-even-train-odd-holdout", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--shell-spacing-m", type=float, default=.25)
    parser.add_argument("--max-gaussians-per-actor", type=int, default=2000)
    parser.add_argument("--validation-every", type=int, default=25)
    args = parser.parse_args(argv)
    train_raw_dynamic_actor(RawDynamicActorTrainOptions(
        trusted_manifest=args.trusted_manifest, output=args.output, track_ids=args.track_ids, initial_actors=args.initial_actors,
        reference_start=args.reference_start, reference_end=args.reference_end, camera_ids=args.camera_ids,
        temporal_even_train_odd_holdout=args.temporal_even_train_odd_holdout, device=args.device,
        width=args.width, height=args.height, iterations=args.iterations, learning_rate=args.learning_rate,
        shell_spacing_m=args.shell_spacing_m, max_gaussians_per_actor=args.max_gaussians_per_actor,
        validation_every=args.validation_every,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
