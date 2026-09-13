from __future__ import annotations
import argparse
from pathlib import Path
from .mast3r_dense_depth import Mast3rDenseDepthAuditOptions, audit_mast3r_dense_depth

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit MASt3R metric dense depth against raw NCore LiDAR before actor reconstruction.")
    p.add_argument("--dataset-manifest", type=Path, required=True); p.add_argument("--mast3r-root", type=Path, required=True)
    p.add_argument("--mast3r-weights", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--track-id", type=int, required=True)
    p.add_argument("--camera-a", default="camera_cross_right_120fov"); p.add_argument("--camera-b", default="camera_front_wide_120fov"); p.add_argument("--reference-index", type=int, default=140)
    p.add_argument("--patch-size", type=int, default=512); p.add_argument("--patch-fov-deg", type=float, default=35.0); p.add_argument("--patch-stride", type=int, default=4); p.add_argument("--min-confidence", type=float, default=1.0)
    p.add_argument("--max-lidar-pixel-distance", type=float, default=3.0); p.add_argument("--max-median-lidar-relative-error", type=float, default=.25); p.add_argument("--min-cuboid-fraction", type=float, default=.75); p.add_argument("--device", default="cuda")
    a=p.parse_args(argv)
    audit_mast3r_dense_depth(Mast3rDenseDepthAuditOptions(a.dataset_manifest,a.mast3r_root,a.mast3r_weights,a.output,a.track_id,a.camera_a,a.camera_b,a.reference_index,a.patch_size,a.patch_fov_deg,a.patch_stride,a.min_confidence,a.max_lidar_pixel_distance,a.max_median_lidar_relative_error,a.min_cuboid_fraction,a.device))
    return 0
if __name__ == "__main__": raise SystemExit(main())
