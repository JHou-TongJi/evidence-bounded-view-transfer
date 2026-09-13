from __future__ import annotations

import argparse
from pathlib import Path

from .source_proxy_composite import SourceProxyStitchOptions, stitch_source_proxy_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stitch completed source-proxy raw shards with global display fades.")
    parser.add_argument("--static-render-dir", required=True, type=Path)
    parser.add_argument("--chunk-dir", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--temporal-fade-frames", type=int, default=5)
    parser.add_argument("--minimum-raw-proxy-fraction", type=float, default=0.0,
                        help="Display-only completeness gate; smaller raw proxy fragments fall back to static RGB.")
    parser.add_argument("--minimum-contiguous-frames", type=int, default=1,
                        help="Suppress shorter accepted proxy fragments before applying temporal fades.")
    args = parser.parse_args(argv)
    result = stitch_source_proxy_chunks(SourceProxyStitchOptions(
        static_render_dir=args.static_render_dir, chunk_dirs=tuple(args.chunk_dir), output_dir=args.output_dir,
        temporal_fade_frames=args.temporal_fade_frames, minimum_raw_proxy_fraction=args.minimum_raw_proxy_fraction,
        minimum_contiguous_frames=args.minimum_contiguous_frames,
    ))
    print(f"Stitched source-conditioned 2.5-D sequence: {args.output_dir} ({result['frame_count']} frame(s))", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
