"""CLI for a display-only ProPainter L4 target-rig experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from .propainter_demo import (
    ProPainterBackgroundPrepareOptions,
    ProPainterPrepareOptions,
    audit_propainter_display_demo,
    prepare_propainter_background_demo,
    prepare_propainter_display_demo,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare/audit a display-only ProPainter dynamic-removal experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write ordered target RGB frames and projected dynamic-removal masks")
    prepare.add_argument("--render-dir", type=Path, required=True)
    prepare.add_argument("--dynamic-dataset-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--track-id", type=int, action="append", required=True)
    prepare.add_argument("--frame-start", type=int, required=True)
    prepare.add_argument("--frame-end", type=int, required=True)
    prepare.add_argument("--margin-pixels", type=int, default=4)
    prepare.add_argument("--actor-mask-render-dir", type=Path,
                         help="Optional actor-only target render: its alpha replaces coarse cuboid rectangles as the display mask.")
    prepare.add_argument("--actor-alpha-threshold", type=float, default=.02)
    prepare.add_argument("--actor-mask-dilation-pixels", type=int, default=2)
    prepare.add_argument("--actor-mask-allow-missing", action="store_true",
                         help="Use transparent context frames when the optional actor render does not cover the full requested window.")
    prepare.add_argument("--max-mask-fraction", type=float, default=.20)
    background = sub.add_parser("prepare-background", help="write only alpha-low plus sky-invalid background-hole masks")
    background.add_argument("--render-dir", type=Path, required=True)
    background.add_argument("--output-dir", type=Path, required=True)
    background.add_argument("--frame-start", type=int, required=True)
    background.add_argument("--frame-end", type=int, required=True)
    background.add_argument("--alpha-threshold", type=float, default=.02)
    background.add_argument("--sky-valid-threshold", type=float, default=.50)
    background.add_argument("--dilation-pixels", type=int, default=1)
    background.add_argument("--max-mask-fraction", type=float, default=.08)
    audit = sub.add_parser("audit", help="verify ProPainter output only changed requested mask pixels")
    audit.add_argument("--prepared-dir", type=Path, required=True)
    audit.add_argument("--result-frames", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_propainter_display_demo(ProPainterPrepareOptions(
            render_dir=args.render_dir, dynamic_dataset_manifest=args.dynamic_dataset_manifest,
            output=args.output_dir, track_ids=tuple(args.track_id), frame_start=args.frame_start,
            frame_end=args.frame_end, actor_mask_render_dir=args.actor_mask_render_dir,
            actor_alpha_threshold=args.actor_alpha_threshold, actor_mask_dilation_pixels=args.actor_mask_dilation_pixels,
            actor_mask_allow_missing=args.actor_mask_allow_missing,
            margin_pixels=args.margin_pixels, max_mask_fraction=args.max_mask_fraction,
        ))
    elif args.command == "prepare-background":
        prepare_propainter_background_demo(ProPainterBackgroundPrepareOptions(
            render_dir=args.render_dir, output=args.output_dir, frame_start=args.frame_start, frame_end=args.frame_end,
            alpha_threshold=args.alpha_threshold, sky_valid_threshold=args.sky_valid_threshold,
            dilation_pixels=args.dilation_pixels, max_mask_fraction=args.max_mask_fraction,
        ))
    else:
        audit_propainter_display_demo(args.prepared_dir, args.result_frames, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
