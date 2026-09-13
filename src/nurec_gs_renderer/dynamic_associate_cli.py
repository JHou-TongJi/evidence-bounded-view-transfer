from __future__ import annotations

import argparse
from pathlib import Path

from .dynamic import load_dynamic_gaussians
from .dynamic_associate import DynamicAssociateOptions, build_dynamic_actor_association


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-dynamic-associate",
        description="Associate exported dynamic Gaussians with NCore actor tracks and source slots.",
    )
    parser.add_argument("--dynamic-gaussians", required=True, type=Path)
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-margin-us", type=int, default=1_000_000)
    parser.add_argument("--minimum-rig-distance-m", type=float, default=3.0)
    parser.add_argument("--minimum-travel-distance-m", type=float, default=1.5)
    parser.add_argument("--cuboid-padding-m", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_dynamic_gaussians(args.dynamic_gaussians)
    print(f"Loaded {scene.count:,} dynamic Gaussians", flush=True)
    association = build_dynamic_actor_association(
        scene,
        DynamicAssociateOptions(
            dynamic_path=args.dynamic_gaussians,
            ncore_path=args.ncore_path,
            output=args.output,
            query_margin_us=args.query_margin_us,
            minimum_rig_distance_m=args.minimum_rig_distance_m,
            minimum_travel_distance_m=args.minimum_travel_distance_m,
            cuboid_padding_m=tuple(args.cuboid_padding_m),
        ),
    )
    association.save(args.output)
    active_tracks = sum(
        int(track.get("associated_gaussian_count", 0)) > 0 for track in association.tracks
    )
    print(
        f"Associated {association.associated_count:,}/{association.gaussian_count:,} Gaussians "
        f"({association.associated_count / association.gaussian_count:.1%}) to "
        f"{active_tracks}/{len(association.tracks)} dynamic tracks",
        flush=True,
    )
    for index, track in enumerate(association.tracks):
        count = int(track.get("associated_gaussian_count", 0))
        if count:
            print(
                f"  track[{index}] id={track['track_id']} class={track['class_id']} "
                f"gaussians={count:,} slots={track['source_slots']}",
                flush=True,
            )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

