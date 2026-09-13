from __future__ import annotations

import argparse
from pathlib import Path

from .submission_v1 import SubmissionV1Options, build_submission_v1, verify_submission_v1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the evidence-first L4 7V V1 submission")
    parser.add_argument("--verify", type=Path, help="verify an existing submission and exit")
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--background-root", type=Path)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--expert-registry", type=Path)
    parser.add_argument("--road-depth-report", type=Path)
    parser.add_argument("--ncore-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--metric-stride", type=int, default=10, help="decode every Nth frame for sequence-quality proxy metrics")
    args = parser.parse_args(argv)
    if args.verify:
        return 0 if verify_submission_v1(args.verify)["status"] == "complete" else 1
    required = ("sequence_root", "background_root", "camera_config", "expert_registry", "road_depth_report", "ncore_path", "output")
    missing = [item for item in required if getattr(args, item) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + item.replace("_", "-") for item in missing))
    output = build_submission_v1(SubmissionV1Options(sequence_root=args.sequence_root, background_root=args.background_root, camera_config=args.camera_config, expert_registry=args.expert_registry, road_depth_report=args.road_depth_report, output=args.output, ncore_path=args.ncore_path, fps=args.fps, metric_stride=args.metric_stride))
    print(f"Wrote submission V1: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
