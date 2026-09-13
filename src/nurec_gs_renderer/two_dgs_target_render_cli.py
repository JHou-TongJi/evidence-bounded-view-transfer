"""CLI for diagnostic target-rig rendering with an official static 2DGS model."""
from __future__ import annotations

import argparse
from pathlib import Path

from .two_dgs_target_render import TwoDgsTargetRenderOptions, render_two_dgs_target_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a static official-2DGS checkpoint through a target rig (diagnostic only)")
    parser.add_argument("--two-dgs-root", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sky-cubemap", type=Path,
                        help="optional static .npz or temporal .json world-direction sky asset")
    parser.add_argument("--sky-confidence-threshold", type=float, default=.5)
    parser.add_argument("--sky-min-sample-count", type=int, default=1)
    parser.add_argument("--write-depth", action="store_true", help="write uint16 camera-z surf_depth PNGs for actor/static diagnostic gating")
    parser.add_argument("--depth-quantization-m", type=float, default=.01)
    args = parser.parse_args(argv)
    render_two_dgs_target_sequence(TwoDgsTargetRenderOptions(
        two_dgs_root=args.two_dgs_root, source_path=args.source_path, model_path=args.model_path,
        camera_config=args.camera_config, output=args.output, iteration=args.iteration,
        camera_ids=tuple(args.camera_id or ()), chunk_index=args.chunk_index,
        frame_start=args.frame_start, frame_end=args.frame_end, stride=args.stride,
        width=args.width, height=args.height, device=args.device, sky_cubemap=args.sky_cubemap,
        sky_confidence_threshold=args.sky_confidence_threshold, sky_min_sample_count=args.sky_min_sample_count,
        write_depth=args.write_depth, depth_quantization_m=args.depth_quantization_m,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
