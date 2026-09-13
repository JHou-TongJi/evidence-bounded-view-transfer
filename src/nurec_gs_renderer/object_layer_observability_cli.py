"""CLI for the read-only rigid object-layer observability audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from .object_layer_observability import ObjectLayerObservabilityOptions, audit_object_layer_observability


def _camera_ids(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("camera IDs must be non-empty and unique")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit raw NCore LiDAR observability before rigid object-layer reconstruction")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-ids", type=_camera_ids,
                        default=("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_front_tele_30fov"))
    parser.add_argument("--reference-start", type=int, default=160)
    parser.add_argument("--reference-end", type=int, default=182)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-size-m", type=float, default=.10)
    parser.add_argument("--cuboid-margin-m", type=float, default=.15)
    args = parser.parse_args(argv)
    audit_object_layer_observability(ObjectLayerObservabilityOptions(
        trusted_manifest=args.trusted_manifest, output=args.output, track_id=args.track_id, camera_ids=args.camera_ids,
        reference_start=args.reference_start, reference_end=args.reference_end, device=args.device,
        voxel_size_m=args.voxel_size_m, cuboid_margin_m=args.cuboid_margin_m,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
