from __future__ import annotations

import argparse
from pathlib import Path

from .actor_audit import ActorAuditOptions, audit_actor_layers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Street actor layers on original NCore FTheta vehicle ROIs")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-render-dir", type=Path, required=True)
    parser.add_argument("--composite-render-dir", type=Path, required=True)
    parser.add_argument("--actor-render-dir", type=Path, required=True)
    parser.add_argument(
        "--actor-dataset-manifest", type=Path, required=True,
        help="ncore_street_manifest.json used to train/export this canonical actor asset",
    )
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    parser.add_argument("--no-previews", action="store_true", help="Skip large comparison strips for fast sequence-only metrics")
    args = parser.parse_args(argv)
    audit_actor_layers(ActorAuditOptions(
        ncore_path=args.ncore_path, static_render_dir=args.static_render_dir,
        composite_render_dir=args.composite_render_dir, actor_render_dir=args.actor_render_dir,
        actor_dataset_manifest=args.actor_dataset_manifest,
        segformer_model_dir=args.segformer_model_dir, output_dir=args.output_dir,
        camera_id=args.camera_id, device=args.device,
        probability_threshold=args.vehicle_probability_threshold,
        margin_threshold=args.vehicle_margin_threshold, cuboid_margin_pixels=args.cuboid_margin_pixels,
        write_previews=not args.no_previews,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
