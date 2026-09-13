from __future__ import annotations

import argparse
from pathlib import Path

from .street_dataset import StreetDatasetOptions, build_street_dataset


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert an NCore chunk into the Street Gaussians pinhole proxy dataset")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--horizontal-fov-deg", type=float, default=100.0)
    parser.add_argument("--rear-horizontal-fov-deg", type=float, default=60.0)
    parser.add_argument("--proxy-layout", choices=("single", "tiles"), default="single")
    parser.add_argument("--tile-horizontal-fov-deg", type=float, default=55.0)
    parser.add_argument("--rear-tile-horizontal-fov-deg", type=float, default=55.0)
    parser.add_argument("--tile-yaws-deg", type=_comma_floats, default=(-40.0, 0.0, 40.0),
                        help="front/cross tile yaw angles, e.g. -40,0,40")
    parser.add_argument("--rear-tile-yaws-deg", type=_comma_floats, default=(0.0,),
                        help="rear tile yaw angles, e.g. 0")
    parser.add_argument("--validation-stride", type=int, default=20,
                        help="hold out every Nth synchronized frame for validation")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-background-points", type=int, default=300_000)
    parser.add_argument("--voxel-size-m", type=float, default=0.20)
    parser.add_argument("--rectify-device", default="cuda")
    parser.add_argument("--no-actors", action="store_true")
    parser.add_argument("--write-actor-masks", action="store_true",
                        help="write conservative projected vehicle-cuboid masks for actor-only supervision")
    parser.add_argument("--actor-mask-margin-pixels", type=int, default=8)
    parser.add_argument("--actor-mask-segformer-model-dir", type=Path,
                        help="optional local SegFormer directory; intersect cuboids with high-confidence car/bus/truck/van pixels")
    parser.add_argument("--actor-mask-segformer-device", default="cuda")
    parser.add_argument("--actor-mask-vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--actor-mask-vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--actor-mask-vehicle-erosion-pixels", type=int, default=0)
    parser.add_argument("--write-lidar-depth", action="store_true",
                        help="project nearest lidar_top_360fov frame into each proxy image")
    parser.add_argument("--max-lidar-timestamp-delta-us", type=int, default=100_000)
    parser.add_argument("--include-sky-initialization", action="store_true",
                        help="keep PLY points marked sky_mask=1 in the initial Street background")
    args = parser.parse_args(argv)
    build_street_dataset(StreetDatasetOptions(
        ncore_path=args.ncore_path, static_ply=args.static_ply, output=args.output, chunk_index=args.chunk_index,
        width=args.width, height=args.height, horizontal_fov_deg=args.horizontal_fov_deg,
        rear_horizontal_fov_deg=args.rear_horizontal_fov_deg, proxy_layout=args.proxy_layout,
        tile_horizontal_fov_deg=args.tile_horizontal_fov_deg,
        rear_tile_horizontal_fov_deg=args.rear_tile_horizontal_fov_deg,
        tile_yaws_deg=args.tile_yaws_deg, rear_tile_yaws_deg=args.rear_tile_yaws_deg,
        validation_stride=args.validation_stride,
        frame_start=args.frame_start, frame_end=args.frame_end, stride=args.stride,
        max_background_points=args.max_background_points, voxel_size_m=args.voxel_size_m,
        rectify_device=args.rectify_device, include_actors=not args.no_actors,
        exclude_sky_initialization=not args.include_sky_initialization,
        write_actor_masks=args.write_actor_masks, actor_mask_margin_pixels=args.actor_mask_margin_pixels,
        actor_mask_segformer_model_dir=args.actor_mask_segformer_model_dir,
        actor_mask_segformer_device=args.actor_mask_segformer_device,
        actor_mask_vehicle_probability_threshold=args.actor_mask_vehicle_probability_threshold,
        actor_mask_vehicle_margin_threshold=args.actor_mask_vehicle_margin_threshold,
        actor_mask_vehicle_erosion_pixels=args.actor_mask_vehicle_erosion_pixels,
        write_lidar_depth=args.write_lidar_depth, max_lidar_timestamp_delta_us=args.max_lidar_timestamp_delta_us,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
