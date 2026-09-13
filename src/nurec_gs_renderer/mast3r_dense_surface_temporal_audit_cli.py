from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_dense_surface_temporal_audit import Mast3rDenseSurfaceTemporalAuditOptions, audit_mast3r_dense_surface_temporal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit temporal residual of independently reconstructed MASt3R dense surfaces")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--static-render-dir", required=True, type=Path, action="append")
    parser.add_argument("--composite-render-dir", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--camera-id", default="camera_cross_right_120fov")
    args = parser.parse_args(argv)
    audit_mast3r_dense_surface_temporal(Mast3rDenseSurfaceTemporalAuditOptions(
        dataset_manifest=args.dataset_manifest, static_render_dirs=tuple(args.static_render_dir), composite_render_dirs=tuple(args.composite_render_dir), output=args.output, track_id=args.track_id, camera_id=args.camera_id,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
