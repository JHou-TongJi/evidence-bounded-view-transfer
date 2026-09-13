from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_feature_correspondence import DynamicFeatureCorrespondenceOptions, audit_dynamic_feature_correspondence


def _camera_ids(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("camera IDs must be unique and non-empty")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local SAM2 temporal feature correspondences for trusted dynamic source masks")
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-ids", type=_camera_ids)
    parser.add_argument("--reference-start", type=int)
    parser.add_argument("--reference-end", type=int)
    parser.add_argument("--max-samples-per-mask", type=int, default=512)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--feature-level", choices=("high", "mid", "deep"), default="deep")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    audit_dynamic_feature_correspondence(DynamicFeatureCorrespondenceOptions(
        trusted_manifest=args.trusted_manifest, model_dir=args.model_dir, output=args.output, track_id=args.track_id,
        camera_ids=args.camera_ids, reference_start=args.reference_start, reference_end=args.reference_end,
        max_samples_per_mask=args.max_samples_per_mask, max_pairs=args.max_pairs, device=args.device,
        feature_level=args.feature_level,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
