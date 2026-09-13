from __future__ import annotations

import argparse
from pathlib import Path

from .emernerf_ncore_bundle import EmerNeRFNCoreBundleOptions, build_emernerf_ncore_bundle


def _camera_ids(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("camera IDs must be unique and non-empty")
    return result


def _track_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track IDs must be integers") from exc
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("track IDs must be unique non-negative integers")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze native NCore FTheta/rolling ray records for an EmerNeRF pilot")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-ids", type=_camera_ids, required=True)
    parser.add_argument("--reference-start", type=int, required=True)
    parser.add_argument("--reference-end", type=int, required=True)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--holdout-camera-ids", type=_camera_ids, default=())
    parser.add_argument("--actor-track-ids", type=_track_ids, default=())
    parser.add_argument("--allow-holdout-background-train", action="store_true")
    parser.add_argument("--ray-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    build_emernerf_ncore_bundle(EmerNeRFNCoreBundleOptions(
        trusted_manifest=args.trusted_manifest, output=args.output, camera_ids=args.camera_ids,
        reference_start=args.reference_start, reference_end=args.reference_end, output_width=args.width, output_height=args.height,
        holdout_camera_ids=args.holdout_camera_ids, actor_track_ids=args.actor_track_ids,
        allow_holdout_background_train=args.allow_holdout_background_train, ray_smoke=args.ray_smoke, device=args.device,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
