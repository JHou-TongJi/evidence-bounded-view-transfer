from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_track_audit import DynamicTrackAuditOptions, audit_dynamic_track


def _csv_indices(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame indices must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("frame indices must be non-negative and non-empty")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one associated dynamic track before whitelisting it as a rigid vehicle")
    parser.add_argument("--dynamic-gaussians", type=Path, required=True)
    parser.add_argument("--dynamic-actor-association", type=Path, required=True)
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--frame-indices", type=_csv_indices, default=(100, 107, 157, 160))
    parser.add_argument("--camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    audit_dynamic_track(DynamicTrackAuditOptions(
        dynamic_gaussians=args.dynamic_gaussians, dynamic_actor_association=args.dynamic_actor_association,
        ncore_path=args.ncore_path, static_ply=args.static_ply, segformer_model_dir=args.segformer_model_dir,
        output_dir=args.output_dir, track_id=args.track_id, frame_indices=args.frame_indices,
        camera_id=args.camera_id, width=args.width, height=args.height, device=args.device,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
