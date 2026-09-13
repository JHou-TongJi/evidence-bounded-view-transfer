from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_canonical_audit import DEFAULT_CAMERA_IDS, DynamicCanonicalAuditOptions, audit_dynamic_canonical


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected non-empty, unique comma-separated values")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target slots must be comma-separated integers") from exc
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("target slots must be non-negative and unique")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit native source-slot dynamic coverage versus a canonical actor on exact NCore FTheta frames")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--dynamic-gaussians", type=Path, required=True)
    parser.add_argument("--canonical-actors", type=Path, required=True)
    parser.add_argument("--actor-dataset-manifest", type=Path, required=True)
    parser.add_argument("--static-v3-ply", type=Path, required=True)
    parser.add_argument("--static-v4-ply", type=Path, required=True)
    parser.add_argument("--clean-static-mask-dir", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--target-slots", type=_csv_ints, default=(4, 5, 6))
    parser.add_argument("--camera-ids", type=_csv_strings, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    parser.add_argument("--vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--surrounding-ring-outer-pixels", type=int, default=20)
    parser.add_argument("--surrounding-ring-inner-pixels", type=int, default=4)
    args = parser.parse_args(argv)
    audit_dynamic_canonical(DynamicCanonicalAuditOptions(
        ncore_path=args.ncore_path, dynamic_gaussians=args.dynamic_gaussians, canonical_actors=args.canonical_actors,
        actor_dataset_manifest=args.actor_dataset_manifest, static_v3_ply=args.static_v3_ply, static_v4_ply=args.static_v4_ply,
        clean_static_mask_dir=args.clean_static_mask_dir, segformer_model_dir=args.segformer_model_dir, output_dir=args.output_dir,
        track_id=args.track_id, target_slots=args.target_slots, camera_ids=args.camera_ids, width=args.width, height=args.height,
        device=args.device, cuboid_margin_pixels=args.cuboid_margin_pixels, probability_threshold=args.vehicle_probability_threshold,
        margin_threshold=args.vehicle_margin_threshold, surrounding_ring_outer_pixels=args.surrounding_ring_outer_pixels,
        surrounding_ring_inner_pixels=args.surrounding_ring_inner_pixels,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
