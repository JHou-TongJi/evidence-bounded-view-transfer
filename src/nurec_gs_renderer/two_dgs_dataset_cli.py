"""CLI for a bounded NCore-to-official-2DGS pinhole-proxy conversion."""
from __future__ import annotations

import argparse
from pathlib import Path

from .two_dgs_dataset import TwoDgsDatasetOptions, build_two_dgs_dataset


def _comma_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a COLMAP-style pinhole proxy for the official 2DGS baseline")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--camera-ids", type=_comma_strings, default=None)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--horizontal-fov-deg", type=float, default=55.0)
    parser.add_argument("--tile-yaws-deg", type=_comma_floats, default=(0.0,))
    parser.add_argument("--rear-tile-yaws-deg", type=_comma_floats, default=(0.0,))
    parser.add_argument("--near-field-camera-ids", type=_comma_strings, default=(),
                        help="source cameras with additional roadward-pitched virtual tiles")
    parser.add_argument("--near-field-tile-yaws-deg", type=_comma_floats, default=())
    parser.add_argument("--near-field-tile-pitch-deg", type=float, default=-12.0,
                        help="negative OpenCV pitch looks roadward; used only with --near-field-camera-ids")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--max-initial-points", type=int, default=100_000)
    parser.add_argument("--rectify-device", default="cuda")
    parser.add_argument("--dynamic-mask-manifest", type=Path,
                        help="accepted source-priority masks to exclude from static RGB supervision")
    parser.add_argument("--filter-dynamic-initialization", action="store_true",
                        help="conservatively remove initial PLY points seen in accepted dynamic masks across multiple source observations")
    parser.add_argument("--dynamic-initialization-min-observations", type=int, default=2,
                        help="minimum independent source camera/reference observations required to remove one initial point")
    parser.add_argument("--road-lidar-seed", type=Path,
                        help="optional ncore-2dgs-road-lidar-seed .npz appended to initial points3D.ply")
    parser.add_argument("--replace-road-initialization", action="store_true",
                        help="remove source PLY road_mask points before appending --road-lidar-seed")
    parser.add_argument("--road-depth-supervision", action="store_true",
                        help="write sparse static road LiDAR depth/normal targets for proxy-training supervision")
    parser.add_argument("--road-depth-stride", type=int, default=5,
                        help="write a road depth target every N selected reference frames")
    parser.add_argument("--road-depth-holdout-every", type=int, default=5,
                        help="reserve one in this many depth target slots for depth-only holdout")
    parser.add_argument("--road-depth-holdout-offset", type=int, default=4,
                        help="zero-based reserved depth slot within --road-depth-holdout-every")
    parser.add_argument("--road-depth-min-range-m", type=float, default=3.0)
    parser.add_argument("--road-depth-max-range-m", type=float, default=60.0)
    parser.add_argument("--road-depth-ground-local-z-min-m", type=float, default=-3.2)
    parser.add_argument("--road-depth-ground-local-z-max-m", type=float, default=-1.4)
    parser.add_argument("--road-depth-lidar-window", type=int, default=2,
                        help="number of neighbouring LiDAR scans on either side of the closest scan")
    parser.add_argument("--road-depth-max-lidar-delta-us", type=int, default=250_000)
    parser.add_argument("--road-depth-lidar-decay-us", type=int, default=100_000)
    parser.add_argument("--road-depth-ego-exclusion-radius-m", type=float, default=2.5)
    args = parser.parse_args(argv)
    kwargs = {} if args.camera_ids is None else {"camera_ids": args.camera_ids}
    build_two_dgs_dataset(TwoDgsDatasetOptions(ncore_path=args.ncore_path, static_ply=args.static_ply, output=args.output,
        chunk_index=args.chunk_index, width=args.width, height=args.height, horizontal_fov_deg=args.horizontal_fov_deg,
        tile_yaws_deg=args.tile_yaws_deg, rear_tile_yaws_deg=args.rear_tile_yaws_deg,
        near_field_camera_ids=args.near_field_camera_ids,
        near_field_tile_yaws_deg=args.near_field_tile_yaws_deg,
        near_field_tile_pitch_deg=args.near_field_tile_pitch_deg,
        frame_start=args.frame_start, frame_end=args.frame_end, stride=args.stride,
        holdout_stride=args.holdout_stride, max_initial_points=args.max_initial_points, rectify_device=args.rectify_device,
        dynamic_mask_manifest=args.dynamic_mask_manifest, road_lidar_seed=args.road_lidar_seed,
        filter_dynamic_initialization=args.filter_dynamic_initialization,
        dynamic_initialization_min_observations=args.dynamic_initialization_min_observations,
        replace_road_initialization=args.replace_road_initialization,
        road_depth_supervision=args.road_depth_supervision, road_depth_stride=args.road_depth_stride,
        road_depth_holdout_every=args.road_depth_holdout_every,
        road_depth_holdout_offset=args.road_depth_holdout_offset,
        road_depth_min_range_m=args.road_depth_min_range_m, road_depth_max_range_m=args.road_depth_max_range_m,
        road_depth_ground_local_z_min_m=args.road_depth_ground_local_z_min_m,
        road_depth_ground_local_z_max_m=args.road_depth_ground_local_z_max_m,
        road_depth_lidar_window=args.road_depth_lidar_window,
        road_depth_max_lidar_delta_us=args.road_depth_max_lidar_delta_us,
        road_depth_lidar_decay_us=args.road_depth_lidar_decay_us,
        road_depth_ego_exclusion_radius_m=args.road_depth_ego_exclusion_radius_m, **kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
