from __future__ import annotations
import argparse
from pathlib import Path
from .raw_dynamic_actor_initialize import RawLidarActorInitializeOptions, initialize_raw_lidar_actor

def _ids(value: str) -> tuple[int, ...]:
    try: result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc: raise argparse.ArgumentTypeError("track IDs must be comma-separated integers") from exc
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result): raise argparse.ArgumentTypeError("track IDs must be unique non-negative integers")
    return result

def _camera_ids(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result): raise argparse.ArgumentTypeError("camera IDs must be non-empty and unique")
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize a rigid canonical actor from raw multi-view source-mask LiDAR")
    parser.add_argument("--trusted-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-ids", type=_ids, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size-m", type=float, default=.10); parser.add_argument("--minimum-view-support", type=int, default=2)
    parser.add_argument("--reference-start", type=int); parser.add_argument("--reference-end", type=int)
    parser.add_argument("--camera-ids", type=_camera_ids); parser.add_argument("--reference-parity", type=int, choices=(0, 1))
    args = parser.parse_args(argv)
    initialize_raw_lidar_actor(RawLidarActorInitializeOptions(trusted_manifest=args.trusted_manifest, output=args.output, track_ids=args.track_ids, device=args.device, voxel_size_m=args.voxel_size_m, minimum_view_support=args.minimum_view_support, reference_start=args.reference_start, reference_end=args.reference_end, camera_ids=args.camera_ids, reference_parity=args.reference_parity))
    return 0

if __name__ == "__main__": raise SystemExit(main())
