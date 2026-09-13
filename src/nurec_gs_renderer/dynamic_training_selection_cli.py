from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_training_selection import DynamicTrainingSelectionOptions, select_dynamic_training_supervision


def _track_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track IDs must be comma-separated integers") from exc
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("track IDs must be unique non-negative integers")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select trusted raw dynamic-layer supervision and cross-view SAM2 holdouts")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--refinement-manifest", type=Path, required=True)
    parser.add_argument("--crossview-report", type=Path, action="append", required=True,
                        help="exact-FTheta cross-view audit JSON; may be repeated")
    parser.add_argument("--track-ids", type=_track_ids, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true", help="only for synthetic/unit-test manifests")
    args = parser.parse_args(argv)
    select_dynamic_training_supervision(DynamicTrainingSelectionOptions(
        source_manifest=args.source_manifest,
        refinement_manifest=args.refinement_manifest,
        crossview_reports=tuple(args.crossview_report),
        output=args.output,
        track_ids=args.track_ids,
        verify_referenced_files=not args.skip_file_check,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
