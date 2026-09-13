from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .color_correction import GaussianColorCorrection, file_sha256
from .dynamic import DYNAMIC_TIME_MODES, infer_dynamic_source_slots, load_dynamic_gaussians
from .dynamic_association import DynamicActorAssociation
from .canonical_actor import CanonicalActorAsset
from .opacity_correction import GaussianOpacityCorrection
from .ply_io import load_gaussian_ply
from .pose_sources import CameraPathSelection, load_camera_path_set
from .render import (
    GsplatRenderer,
    RenderOptions,
    SEQUENCE_OUTPUT_COMPONENTS,
    SequenceRenderOptions,
    render_camera_path,
)
from .sky import TemporalSkyCubemap, load_sky_asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-render",
        description="Render InstantNuRec/Graphdeco Gaussian PLY scenes with gsplat.",
    )
    parser.add_argument("--ply", required=True, type=Path, help="InstantNuRec or Graphdeco Gaussian PLY")
    parser.add_argument("--camera-config", required=True, type=Path, help="World or NCore camera path JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--camera-id",
        action="append",
        help="Render only this named camera from a multi-camera rig; may be repeated",
    )
    parser.add_argument("--width", type=int, help="Output width; defaults to camera-config width")
    parser.add_argument("--height", type=int, help="Output height; defaults to camera-config height")
    parser.add_argument(
        "--chunk-index",
        type=int,
        help="Select every source-camera frame inside this InstantNuRec chunk",
    )
    parser.add_argument("--frame-start", type=int, help="Start offset inside the resolved sequence")
    parser.add_argument("--frame-end", type=int, help="Exclusive end offset inside the resolved sequence")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--sample-count",
        type=int,
        help="Evenly sample this many frames after chunk/time/offset/stride filtering",
    )
    parser.add_argument("--timestamp-start-us", type=int)
    parser.add_argument("--timestamp-end-us", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--background", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("R", "G", "B"))
    parser.add_argument(
        "--sky-cubemap",
        type=Path,
        help="Optional world-direction sky cubemap .npz; invalid directions use --background",
    )
    parser.add_argument(
        "--gaussian-color-correction",
        type=Path,
        help="Optional hash-bound .npz sidecar that replaces selected Gaussian DC colors",
    )
    parser.add_argument(
        "--gaussian-opacity-correction",
        type=Path,
        help="Optional hash-bound .npz sidecar that replaces selected Gaussian opacities",
    )
    parser.add_argument(
        "--dynamic-gaussians",
        type=Path,
        help="Optional InstantNuRec .dynamic.npz asset evaluated at each camera timestamp",
    )
    parser.add_argument(
        "--dynamic-time-mode",
        choices=DYNAMIC_TIME_MODES,
        default="blend",
        help=(
            "Dynamic source-time selection: triangular blend, one global nearest slot, "
            "or one nearest slot per associated actor"
        ),
    )
    parser.add_argument(
        "--dynamic-actor-association",
        type=Path,
        help="Hash-bound NCore actor/source-slot sidecar required by actor-nearest mode",
    )
    parser.add_argument(
        "--dynamic-source-camera-index",
        type=int,
        help=(
            "Diagnostic-only v2 provenance filter: retain one exported source-camera index. "
            "The index must be mapped to a physical camera by the asset producer."
        ),
    )
    parser.add_argument(
        "--dynamic-source-nearest-slot",
        action="store_true",
        help=(
            "With --dynamic-source-camera-index, retain only that camera's nearest available "
            "source slot at each output timestamp; diagnostic only."
        ),
    )
    parser.add_argument(
        "--canonical-actors",
        type=Path,
        help="Optional rigid vehicle canonical-actor .npz; replaces --dynamic-gaussians",
    )
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=10000.0)
    parser.add_argument(
        "--static-opacity-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic-only multiplier for static PLY opacity.  Set to 0 with "
            "--canonical-actors or --dynamic-gaussians to render the actor layer alone; "
            "the input PLY is never modified."
        ),
    )
    parser.add_argument(
        "--canonical-actor-opacity-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic-only multiplier for canonical actor opacity.  It is useful for "
            "testing whether static/actor overlap, rather than actor geometry, causes a regression."
        ),
    )
    parser.add_argument("--depth-alpha-threshold", type=float, default=0.01)
    parser.add_argument(
        "--sky-edge-padding-pixels",
        type=int,
        default=2,
        help="Extend sampled sky colors over a narrow invalid boundary band",
    )
    parser.add_argument("--sky-edge-feather-pixels", type=int, default=0)
    parser.add_argument("--sky-edge-smoothing-iterations", type=int, default=0)
    parser.add_argument("--sky-valid-threshold", type=float, default=0.5)
    parser.add_argument("--sky-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sky-min-sample-count", type=int, default=1)
    parser.add_argument("--sky-geometry-guard-pixels", type=int, default=0)
    parser.add_argument(
        "--sky-foreground-decontaminate-pixels",
        type=int,
        default=0,
        help="Replace a narrow sky-facing foreground boundary from an eroded high-alpha core",
    )
    parser.add_argument("--sky-foreground-core-alpha", type=float, default=0.9)
    parser.add_argument("--sky-foreground-core-erosion-pixels", type=int, default=0)
    parser.add_argument("--sky-foreground-boundary-min-alpha", type=float, default=0.02)
    parser.add_argument("--sky-foreground-depth-absolute-tolerance", type=float, default=2.0)
    parser.add_argument("--sky-foreground-depth-relative-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--sky-foreground-alpha-gamma",
        type=float,
        default=1.0,
        help="Optional RGB-only boundary matte shrink; 1 keeps the original compositing alpha",
    )
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--unpacked", action="store_true", help="Use more memory for potentially faster rendering")
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=sorted(SEQUENCE_OUTPUT_COMPONENTS),
        default=["rgb", "alpha", "depth", "depth-preview", "sky-debug"],
        help="Output components; omit depth entries for a faster RGB-only rolling-shutter preview",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a matching partial sequence run")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Atomically overwrite expected files in a non-empty output directory",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_gaussian_ply(args.ply)
    correction_metadata = None
    if args.gaussian_color_correction is not None:
        correction = GaussianColorCorrection.load(args.gaussian_color_correction)
        correction.validate_source(args.ply, scene.count)
        scene = correction.apply_to_scene(scene)
        correction_metadata = {
            "path": str(args.gaussian_color_correction),
            "ply_sha256": correction.ply_sha256,
            "corrected_gaussians": correction.count,
            "details": correction.metadata,
        }
    opacity_correction_metadata = None
    if args.gaussian_opacity_correction is not None:
        opacity_correction = GaussianOpacityCorrection.load(args.gaussian_opacity_correction)
        opacity_correction.validate_source(args.ply, scene.count)
        scene = opacity_correction.apply_to_scene(scene)
        opacity_correction_metadata = {
            "path": str(args.gaussian_opacity_correction),
            "ply_sha256": opacity_correction.ply_sha256,
            "corrected_gaussians": opacity_correction.count,
            "details": opacity_correction.metadata,
        }
    selection = CameraPathSelection(
        chunk_index=args.chunk_index,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        stride=args.stride,
        sample_count=args.sample_count,
        timestamp_start_us=args.timestamp_start_us,
        timestamp_end_us=args.timestamp_end_us,
    )
    camera_set = load_camera_path_set(
        args.camera_config,
        selection=selection,
        camera_ids=None if args.camera_id is None else set(args.camera_id),
    )
    camera_paths = {
        camera_id: camera_path.resized(args.width, args.height)
        if args.width is not None or args.height is not None
        else camera_path
        for camera_id, camera_path in camera_set.paths.items()
    }
    options = RenderOptions(
        device=args.device,
        background=tuple(args.background),
        near_plane=args.near,
        far_plane=args.far,
        packed=not args.unpacked,
        rasterize_mode=args.rasterize_mode,
        depth_alpha_threshold=args.depth_alpha_threshold,
        sky_edge_padding_pixels=args.sky_edge_padding_pixels,
        sky_edge_feather_pixels=args.sky_edge_feather_pixels,
        sky_edge_smoothing_iterations=args.sky_edge_smoothing_iterations,
        sky_valid_threshold=args.sky_valid_threshold,
        sky_confidence_threshold=args.sky_confidence_threshold,
        sky_min_sample_count=args.sky_min_sample_count,
        sky_geometry_guard_pixels=args.sky_geometry_guard_pixels,
        sky_foreground_decontaminate_pixels=args.sky_foreground_decontaminate_pixels,
        sky_foreground_core_alpha=args.sky_foreground_core_alpha,
        sky_foreground_core_erosion_pixels=args.sky_foreground_core_erosion_pixels,
        sky_foreground_boundary_min_alpha=args.sky_foreground_boundary_min_alpha,
        sky_foreground_depth_absolute_tolerance=args.sky_foreground_depth_absolute_tolerance,
        sky_foreground_depth_relative_tolerance=args.sky_foreground_depth_relative_tolerance,
        sky_foreground_alpha_gamma=args.sky_foreground_alpha_gamma,
        dynamic_time_mode=args.dynamic_time_mode,
        dynamic_source_camera_index=args.dynamic_source_camera_index,
        dynamic_source_nearest_slot=args.dynamic_source_nearest_slot,
        static_opacity_scale=args.static_opacity_scale,
        canonical_actor_opacity_scale=args.canonical_actor_opacity_scale,
    )
    sky_cubemap = None if args.sky_cubemap is None else load_sky_asset(args.sky_cubemap)
    dynamic_scene = None if args.dynamic_gaussians is None else load_dynamic_gaussians(
        args.dynamic_gaussians
    )
    canonical_actor_asset = (
        None if args.canonical_actors is None else CanonicalActorAsset.load(args.canonical_actors)
    )
    if canonical_actor_asset is not None and dynamic_scene is not None:
        raise ValueError("--canonical-actors and --dynamic-gaussians are mutually exclusive")
    if args.dynamic_time_mode == "actor-nearest" and args.dynamic_actor_association is None:
        raise ValueError("--dynamic-time-mode actor-nearest requires --dynamic-actor-association")
    if args.dynamic_actor_association is not None and dynamic_scene is None:
        raise ValueError("--dynamic-actor-association requires --dynamic-gaussians")
    if args.dynamic_source_camera_index is not None and dynamic_scene is None:
        raise ValueError("--dynamic-source-camera-index requires --dynamic-gaussians")
    if args.dynamic_source_nearest_slot and args.dynamic_source_camera_index is None:
        raise ValueError("--dynamic-source-nearest-slot requires --dynamic-source-camera-index")
    dynamic_actor_association = (
        None
        if args.dynamic_actor_association is None
        else DynamicActorAssociation.load(args.dynamic_actor_association)
    )
    if dynamic_actor_association is not None:
        dynamic_actor_association.validate_source(args.dynamic_gaussians, dynamic_scene)
    print(f"Loaded {scene.count:,} Gaussians (SH degree {scene.sh_degree})")
    if dynamic_scene is not None:
        _, dynamic_slot_timestamps_us = infer_dynamic_source_slots(dynamic_scene)
        print(
            f"Loaded {dynamic_scene.count:,} time-varying Gaussians "
            f"in {len(dynamic_slot_timestamps_us)} source-time slots "
            f"(mode={args.dynamic_time_mode}; target exposure midpoint approximation)",
            flush=True,
        )
        if dynamic_actor_association is not None:
            print(
                f"Loaded actor labels for {dynamic_actor_association.associated_count:,}/"
                f"{dynamic_actor_association.gaussian_count:,} dynamic Gaussians "
                f"across {len(dynamic_actor_association.tracks)} candidate tracks",
                flush=True,
            )
    if canonical_actor_asset is not None:
        print(
            f"Loaded {canonical_actor_asset.count:,} canonical Gaussians for "
            f"{len(canonical_actor_asset.actor_ids)} rigid actor(s) (initialization-only asset)",
            flush=True,
        )
    total_frames = sum(len(camera_path.frames) for camera_path in camera_paths.values())
    if camera_set.is_rig:
        print(
            f"Rendering target rig {camera_set.rig_id}: {len(camera_paths)} camera(s), "
            f"{total_frames} total view-frame(s)"
        )
    else:
        camera_path = next(iter(camera_paths.values()))
        print(f"Rendering {len(camera_path.frames)} frame(s) from {camera_path.source} poses")
    if sky_cubemap is not None:
        coverage = (
            sky_cubemap.total_coverage
            if isinstance(sky_cubemap, TemporalSkyCubemap)
            else float(np.asarray(sky_cubemap.valid, dtype=bool).mean())
        ) * 100.0
        kind = f"{len(sky_cubemap.timestamps_us)}-slot temporal" if isinstance(
            sky_cubemap, TemporalSkyCubemap
        ) else "static"
        print(
            f"Loaded {sky_cubemap.face_size}px {kind} sky cubemap "
            f"({coverage:.1f}% mean valid directions)"
        )
    first_camera_path = next(iter(camera_paths.values()))
    if first_camera_path.metadata is not None and first_camera_path.metadata.get("chunk") is not None:
        chunk = first_camera_path.metadata["chunk"]
        print(
            f"Selected chunk {chunk['index']}/{chunk['count'] - 1} "
            f"time=[{chunk['start_us']}, {chunk['end_us']}] us"
        )
    sky_input_metadata: dict[str, object] | None = None
    if args.sky_cubemap is not None:
        sky_input_metadata = {
            "path": str(args.sky_cubemap.resolve()),
            "sha256": file_sha256(args.sky_cubemap),
        }
        if isinstance(sky_cubemap, TemporalSkyCubemap):
            sky_input_metadata["slots"] = [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in sky_cubemap.asset_paths
            ]
    common_run_metadata: dict[str, object] = {
        "static_opacity_scale": args.static_opacity_scale,
        "canonical_actor_opacity_scale": args.canonical_actor_opacity_scale,
        "dynamic_source_camera_index": args.dynamic_source_camera_index,
        "dynamic_source_nearest_slot": args.dynamic_source_nearest_slot,
        "ply": {"path": str(args.ply.resolve()), "sha256": file_sha256(args.ply)},
        "camera_config": {
            "path": str(args.camera_config.resolve()),
            "sha256": file_sha256(args.camera_config),
        },
        "sky_cubemap": sky_input_metadata,
        "gaussian_color_correction": None
        if args.gaussian_color_correction is None
        else {
            "path": str(args.gaussian_color_correction.resolve()),
            "sha256": file_sha256(args.gaussian_color_correction),
        },
        "gaussian_opacity_correction": None
        if args.gaussian_opacity_correction is None
        else {
            "path": str(args.gaussian_opacity_correction.resolve()),
            "sha256": file_sha256(args.gaussian_opacity_correction),
        },
        "dynamic_gaussians": None
        if args.dynamic_gaussians is None
        else {
            "path": str(args.dynamic_gaussians.resolve()),
            "sha256": file_sha256(args.dynamic_gaussians),
            "gaussian_count": dynamic_scene.count,
            "time_mode": args.dynamic_time_mode,
            "inferred_source_slot_timestamps_us": dynamic_slot_timestamps_us.tolist(),
            "details": dynamic_scene.metadata,
        },
        "dynamic_actor_association": None
        if args.dynamic_actor_association is None
        else {
            "path": str(args.dynamic_actor_association.resolve()),
            "sha256": file_sha256(args.dynamic_actor_association),
            "associated_gaussian_count": dynamic_actor_association.associated_count,
            "candidate_track_count": len(dynamic_actor_association.tracks),
            "details": dynamic_actor_association.metadata,
        },
        "canonical_actors": None
        if args.canonical_actors is None
        else {
            "path": str(args.canonical_actors.resolve()),
            "sha256": file_sha256(args.canonical_actors),
            "gaussian_count": canonical_actor_asset.count,
            "actor_ids": list(canonical_actor_asset.actor_ids),
            "details": canonical_actor_asset.metadata,
        },
    }
    renderer = GsplatRenderer(
        scene,
        options,
        sky_cubemap=sky_cubemap,
        color_correction_metadata=correction_metadata,
        opacity_correction_metadata=opacity_correction_metadata,
        dynamic_scene=dynamic_scene,
        dynamic_metadata=common_run_metadata["dynamic_gaussians"],
        dynamic_actor_association=dynamic_actor_association,
        canonical_actor_asset=canonical_actor_asset,
        canonical_actor_metadata=common_run_metadata["canonical_actors"],
    )
    for camera_id, camera_path in camera_paths.items():
        camera_output_dir = args.output_dir / camera_id if camera_set.is_rig else args.output_dir
        run_metadata = dict(common_run_metadata)
        if camera_set.is_rig:
            run_metadata["target_rig"] = {
                "rig_id": camera_set.rig_id,
                "camera_id": camera_id,
            }
            print(f"[{camera_id}] Rendering {len(camera_path.frames)} frame(s)")
        render_camera_path(
            renderer,
            camera_path,
            camera_output_dir,
            SequenceRenderOptions(
                components=tuple(args.outputs),
                resume=args.resume,
                overwrite_existing=args.overwrite_existing,
                progress_every=args.progress_every,
                run_metadata=run_metadata,
            ),
        )
        print(f"[{camera_id}] Wrote outputs to {camera_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
