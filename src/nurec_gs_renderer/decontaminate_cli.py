from __future__ import annotations

import argparse
from pathlib import Path

from .gaussian_decontaminate import GaussianDecontaminationOptions, decontaminate_gaussian_colors
from .ply_io import load_gaussian_ply
from .render import GsplatRenderer, RenderOptions
from .sky_build import DEFAULT_CAMERA_IDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-decontaminate",
        description="Discover and optimize sky-contaminated Gaussian DC colors from NCore views.",
    )
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--chunk-index", required=True, type=int)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sky-cubemap", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        default=["camera_front_wide_120fov"],
        help="NCore camera IDs, or the single value 'all'",
    )
    parser.add_argument(
        "--slots",
        nargs="+",
        default=["0"],
        help="Chunk slots in [0,17], or the single value 'all'",
    )
    parser.add_argument(
        "--validation-slots",
        nargs="+",
        default=[],
        help="Held-out chunk slots used only in the final report, or 'all'",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--boundary-pixels", type=int, default=5)
    parser.add_argument("--sky-probability-threshold", type=float, default=0.50)
    parser.add_argument("--sky-margin-threshold", type=float, default=0.0)
    parser.add_argument("--geometry-alpha-threshold", type=float, default=0.05)
    parser.add_argument("--halo-min-luminance-excess", type=float, default=0.03)
    parser.add_argument("--halo-max-sky-color-distance", type=float, default=0.30)
    parser.add_argument("--candidate-fraction", type=float, default=0.01)
    parser.add_argument("--per-view-candidate-fraction", type=float, default=0.02)
    parser.add_argument("--min-view-support", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-color-delta", type=float, default=0.25)
    parser.add_argument("--color-prior-weight", type=float, default=0.05)
    parser.add_argument("--static-preservation-weight", type=float, default=0.20)
    parser.add_argument("--sky-edge-padding-pixels", type=int, default=8)
    parser.add_argument("--sky-edge-smoothing-iterations", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preview-views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_cameras(values: list[str]) -> tuple[str, ...]:
    if values == ["all"]:
        return DEFAULT_CAMERA_IDS
    if "all" in values:
        raise ValueError("'all' cannot be combined with explicit camera IDs")
    return tuple(values)


def _resolve_slots(values: list[str]) -> tuple[int, ...]:
    if values == ["all"]:
        return tuple(range(18))
    if "all" in values:
        raise ValueError("'all' cannot be combined with explicit slots")
    return tuple(int(value) for value in values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = GaussianDecontaminationOptions(
        ply_path=args.ply,
        ncore_path=args.ncore_path,
        chunk_index=args.chunk_index,
        model_dir=args.model_dir,
        sky_cubemap_path=args.sky_cubemap,
        output=args.output,
        camera_ids=_resolve_cameras(args.camera_ids),
        slots=_resolve_slots(args.slots),
        validation_slots=() if not args.validation_slots else _resolve_slots(args.validation_slots),
        device=args.device,
        width=args.width,
        height=args.height,
        boundary_pixels=args.boundary_pixels,
        sky_probability_threshold=args.sky_probability_threshold,
        sky_margin_threshold=args.sky_margin_threshold,
        geometry_alpha_threshold=args.geometry_alpha_threshold,
        halo_min_luminance_excess=args.halo_min_luminance_excess,
        halo_max_sky_color_distance=args.halo_max_sky_color_distance,
        candidate_fraction=args.candidate_fraction,
        per_view_candidate_fraction=args.per_view_candidate_fraction,
        min_view_support=args.min_view_support,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        max_color_delta=args.max_color_delta,
        color_prior_weight=args.color_prior_weight,
        static_preservation_weight=args.static_preservation_weight,
        sky_edge_padding_pixels=args.sky_edge_padding_pixels,
        sky_edge_smoothing_iterations=args.sky_edge_smoothing_iterations,
        cache_dir=args.cache_dir,
        preview_views=args.preview_views,
        seed=args.seed,
    )
    scene = load_gaussian_ply(options.ply_path)
    renderer = GsplatRenderer(
        scene,
        RenderOptions(
            device=options.device,
            sky_edge_padding_pixels=options.sky_edge_padding_pixels,
            sky_edge_smoothing_iterations=options.sky_edge_smoothing_iterations,
        ),
    )
    print(f"Loaded {scene.count:,} Gaussians (SH degree {scene.sh_degree})", flush=True)
    correction = decontaminate_gaussian_colors(scene, renderer, options)
    print(
        f"Wrote correction for {correction.count:,} Gaussians to {options.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
