from __future__ import annotations

import argparse
from pathlib import Path

from .mast3r_correspondence import Mast3rCorrespondenceOptions, build_mast3r_correspondence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build locally loaded MASt3R dynamic-instance correspondences with native FTheta gates")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--mast3r-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report; adjacent NPZ holds accepted points")
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-a", default="camera_cross_right_120fov")
    parser.add_argument("--camera-b", default="camera_front_wide_120fov")
    parser.add_argument("--reference-start", type=int)
    parser.add_argument("--reference-end", type=int)
    parser.add_argument("--reference-stride", type=int, default=1)
    parser.add_argument("--max-references", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-fov-deg", type=float, default=35.0)
    parser.add_argument("--descriptor-subsample", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    build_mast3r_correspondence(Mast3rCorrespondenceOptions(
        dataset_manifest=args.dataset_manifest, mast3r_root=args.mast3r_root, mast3r_weights=args.mast3r_weights,
        output=args.output, track_id=args.track_id, camera_a=args.camera_a, camera_b=args.camera_b,
        reference_start=args.reference_start, reference_end=args.reference_end, reference_stride=args.reference_stride, max_references=args.max_references,
        patch_size=args.patch_size, patch_fov_deg=args.patch_fov_deg, descriptor_subsample=args.descriptor_subsample,
        device=args.device,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
