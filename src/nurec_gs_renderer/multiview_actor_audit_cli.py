from __future__ import annotations

import argparse
from pathlib import Path

from .multiview_actor_audit import DEFAULT_CAMERA_IDS, MultiviewActorAuditOptions, audit_multiview_actor_observations


def _camera_ids(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("camera IDs must be a non-empty, unique comma-separated list")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit clean-static/actor visibility across original NCore FTheta source observations")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--canonical-actors", type=Path, required=True)
    parser.add_argument("--actor-dataset-manifest", type=Path, required=True)
    parser.add_argument("--clean-static-mask-dir", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--model", choices=("pa-front", "pa-multiview", "pq-front"), default="pa-front")
    parser.add_argument("--camera-ids", type=_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    parser.add_argument("--vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--depth-order-tolerance-m", type=float, default=0.30)
    parser.add_argument("--max-lidar-timestamp-delta-us", type=int, default=100_000)
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args(argv)
    audit_multiview_actor_observations(MultiviewActorAuditOptions(
        ncore_path=args.ncore_path, static_ply=args.static_ply, canonical_actors=args.canonical_actors,
        actor_dataset_manifest=args.actor_dataset_manifest, clean_static_mask_dir=args.clean_static_mask_dir,
        segformer_model_dir=args.segformer_model_dir, output_dir=args.output_dir, track_id=args.track_id,
        chunk_index=args.chunk_index, model=args.model, camera_ids=args.camera_ids, width=args.width, height=args.height,
        device=args.device, cuboid_margin_pixels=args.cuboid_margin_pixels,
        probability_threshold=args.vehicle_probability_threshold, margin_threshold=args.vehicle_margin_threshold,
        depth_order_tolerance_m=args.depth_order_tolerance_m,
        max_lidar_timestamp_delta_us=args.max_lidar_timestamp_delta_us, use_lidar=not args.no_lidar,
        write_previews=args.write_previews,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
