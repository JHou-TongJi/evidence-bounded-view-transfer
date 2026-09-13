from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic_mask_refinement_audit import DynamicMaskRefinementAuditOptions, audit_dynamic_mask_refinement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SAM2-refined dynamic mask supervision")
    parser.add_argument("--refinement-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rigid-only", action="store_true")
    parser.add_argument("--minimum-new-masks", type=int, default=5)
    args = parser.parse_args(argv)
    audit_dynamic_mask_refinement(DynamicMaskRefinementAuditOptions(
        refinement_manifest=args.refinement_manifest, output=args.output, rigid_only=args.rigid_only,
        minimum_new_masks=args.minimum_new_masks,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
