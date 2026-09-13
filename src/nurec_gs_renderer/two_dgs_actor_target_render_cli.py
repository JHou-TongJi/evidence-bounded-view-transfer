from __future__ import annotations
import argparse
from pathlib import Path

from .two_dgs_actor_target_render import ActorViewExpert, TwoDgsActorTargetRenderOptions, accepted_experts_from_batch_manifest, experts_from_registry, render_two_dgs_actor_target_sequence


def _expert(value: str) -> ActorViewExpert:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expert must be NAME,ACTOR_DATASET,MODEL_PATH,ITERATION")
    try:
        return ActorViewExpert(parts[0], Path(parts[1]), Path(parts[2]), int(parts[3]))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render camera-gated canonical 2DGS actor experts over a static RGB sequence")
    parser.add_argument("--two-dgs-root", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", type=_expert, action="append", help="NAME,ACTOR_DATASET,MODEL_PATH,ITERATION; may be repeated")
    parser.add_argument("--accepted-batch-manifest", type=Path, help="Use every independently accepted expert from nurec-gs-2dgs-actor-batch")
    parser.add_argument("--expert-registry", type=Path, help="Use a reusable nurec-gs-actor-expert-registry")
    parser.add_argument("--background-rgb-dir", type=Path)
    parser.add_argument("--background-depth-dir", type=Path,
                        help="optional uint16 static 2DGS camera-z depth directory; actors behind it are rejected")
    parser.add_argument("--background-depth-quantization-m", type=float, default=.01)
    parser.add_argument("--static-occlusion-margin-m", type=float, default=.50)
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-view-angle-deg", type=float, default=30.0)
    parser.add_argument("--min-distance-ratio", type=float, default=.55)
    parser.add_argument("--max-distance-ratio", type=float, default=1.85)
    parser.add_argument("--temporal-fade-us", type=int, default=100_000)
    args = parser.parse_args(argv)
    sources = sum(bool(value) for value in (args.expert, args.accepted_batch_manifest, args.expert_registry))
    if sources != 1:
        parser.error("supply exactly one of --expert, --accepted-batch-manifest or --expert-registry")
    if args.expert:
        experts = tuple(args.expert)
    elif args.accepted_batch_manifest:
        experts = accepted_experts_from_batch_manifest(args.accepted_batch_manifest)
    else:
        experts = experts_from_registry(args.expert_registry)
    render_two_dgs_actor_target_sequence(TwoDgsActorTargetRenderOptions(
        two_dgs_root=args.two_dgs_root, camera_config=args.camera_config, output=args.output,
        experts=experts, background_rgb_dir=args.background_rgb_dir,
        background_depth_dir=args.background_depth_dir,
        background_depth_quantization_m=args.background_depth_quantization_m,
        static_occlusion_margin_m=args.static_occlusion_margin_m,
        camera_ids=tuple(args.camera_id or ()), chunk_index=args.chunk_index,
        frame_start=args.frame_start, frame_end=args.frame_end, stride=args.stride,
        width=args.width, height=args.height, device=args.device,
        max_view_angle_deg=args.max_view_angle_deg, min_distance_ratio=args.min_distance_ratio,
        max_distance_ratio=args.max_distance_ratio, temporal_fade_us=args.temporal_fade_us,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
