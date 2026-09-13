from __future__ import annotations
import argparse
from pathlib import Path
from .raw_dynamic_actor_densify import RawRgbLidarDensifyOptions, densify_raw_rgb_lidar_actor
def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Densify a raw LiDAR actor using only multi-view LiDAR-anchored RGB source pixels")
    parser.add_argument("--trusted-manifest",type=Path,required=True); parser.add_argument("--initial-actors",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--track-id",type=int,required=True); parser.add_argument("--device",default="cuda"); parser.add_argument("--voxel-size-m",type=float,default=.08); parser.add_argument("--minimum-view-support",type=int,default=2); parser.add_argument("--minimum-new-gaussians",type=int,default=32); parser.add_argument("--pixel-stride",type=int,default=8); parser.add_argument("--max-anchor-distance-pixels",type=float,default=14.); parser.add_argument("--reference-start",type=int); parser.add_argument("--reference-end",type=int)
    a=parser.parse_args(argv); densify_raw_rgb_lidar_actor(RawRgbLidarDensifyOptions(trusted_manifest=a.trusted_manifest,initial_actors=a.initial_actors,output=a.output,track_id=a.track_id,device=a.device,voxel_size_m=a.voxel_size_m,minimum_view_support=a.minimum_view_support,minimum_new_gaussians=a.minimum_new_gaussians,pixel_stride=a.pixel_stride,max_anchor_distance_pixels=a.max_anchor_distance_pixels,reference_start=a.reference_start,reference_end=a.reference_end)); return 0
if __name__ == "__main__": raise SystemExit(main())
