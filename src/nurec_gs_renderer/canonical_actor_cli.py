from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .canonical_actor import ActorTrajectory, build_canonical_actor_asset, recover_v1_actor_provenance
from .dynamic import load_dynamic_gaussians
from .dynamic_association import DynamicActorAssociation
from .sky_build import _create_ncore_loader


def _ncore_vehicle_tracks(ncore_path: Path, timestamps_us: np.ndarray, labels: set[str]) -> dict[int, dict[str, object]]:
    try:
        from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
        from instant_nurec.utils.types import HalfClosedInterval
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("canonical actor build requires InstantNuRec and nvidia-ncore") from exc
    loader = _create_ncore_loader(ncore_path)
    start, end = int(timestamps_us.min()) - 1_000_000, int(timestamps_us.max()) + 1_000_001
    tracks = consolidate_cuboid_tracks(
        compute_cuboid_df(loader, HalfClosedInterval(start, end)), loader, ["AUTOLABEL"], 0.0, np.eye(4, dtype=np.float64)
    )
    output: dict[int, dict[str, object]] = {}
    for raw_id, track in tracks.items():
        if str(track["label_class"]) not in labels:
            continue
        try:
            track_id = int(raw_id)
        except ValueError:
            continue
        pose = np.asarray(track["poses"], dtype=np.float32)
        times = np.asarray(track["timestamps_us"], dtype=np.int64)
        if len(times) < 2:
            continue
        # Repeat endpoint motion intervals so render timestamps immediately
        # around the source observations remain defined.
        first_gap, last_gap = times[1] - times[0], times[-1] - times[-2]
        poses = np.concatenate((pose[:1], pose, pose[-1:]), axis=0)
        full_times = np.concatenate(([times[0] - first_gap], times, [times[-1] + last_gap])).astype(np.int64)
        dimensions = np.asarray(track.get("dimension", track.get("size", [4.8, 2.0, 1.8])), dtype=np.float32).reshape(-1)
        if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
            raise ValueError(f"NCore track {track_id} has invalid length/width/height")
        output[track_id] = {
            "track_id": track_id,
            "label": str(track["label_class"]),
            "timestamps_us": full_times.astype(np.int64).tolist(),
            "actor_to_world": poses.astype(np.float32).tolist(),
            "length_width_height": dimensions.astype(float).tolist(),
        }
    return output


def _trajectories(tracks: dict[int, dict[str, object]]) -> dict[int, ActorTrajectory]:
    return {
        track_id: ActorTrajectory(
            np.asarray(record["timestamps_us"], dtype=np.int64),
            np.asarray(record["actor_to_world"], dtype=np.float32),
        )
        for track_id, record in tracks.items()
    }


def _write_track_manifest(path: Path, tracks: dict[int, dict[str, object]], actor_ids: tuple[int, ...], *, dynamic_path: Path) -> None:
    selected = [tracks[track_id] for track_id in actor_ids]
    document = {
        "schema": "nurec-gs-canonical-actor-track-manifest",
        "version": 1,
        "source_dynamic_path": str(dynamic_path.resolve()),
        "tracks": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse native-v2 or association-recovered-v1 dynamic Gaussians into rigid vehicle-local canonical actors")
    parser.add_argument("--dynamic-gaussians", required=True, type=Path)
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dynamic-actor-association", type=Path,
                        help="Required to recover vehicle provenance from a v1 pa-front dynamic asset")
    parser.add_argument("--track-manifest-output", type=Path,
                        help="Frozen actor tracks for exact-FTheta optimisation/audit; defaults beside --output")
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--track-id", type=int, action="append",
                        help="Only fuse this numeric NCore track ID; repeat to select several tracks.")
    parser.add_argument("--vehicle-class", action="append",
                        help="Eligible rigid class; repeat to select several. Defaults to automobile and heavy_truck when omitted.")
    args = parser.parse_args(argv)
    scene = load_dynamic_gaussians(args.dynamic_gaussians)
    eligible_classes = set(args.vehicle_class or ("automobile", "heavy_truck"))
    tracks = _ncore_vehicle_tracks(args.ncore_path, scene.keyframe_timestamps_us[:, 1], eligible_classes)
    if args.track_id:
        requested_ids = set(args.track_id)
        missing = sorted(requested_ids.difference(tracks))
        if missing:
            raise ValueError(f"requested track IDs are not eligible or absent from the NCore trajectory: {missing}")
        tracks = {track_id: track for track_id, track in tracks.items() if track_id in requested_ids}
    trajectories = _trajectories(tracks)
    if scene.source_track_ids is None:
        if args.dynamic_actor_association is None:
            raise ValueError(
                "canonical fusion of a v1 dynamic asset requires --dynamic-actor-association; "
                "run nurec-gs-dynamic-associate first"
            )
        association = DynamicActorAssociation.load(args.dynamic_actor_association)
        association.validate_source(args.dynamic_gaussians, scene)
        scene = recover_v1_actor_provenance(scene, association, trajectories)
    elif args.dynamic_actor_association is not None:
        raise ValueError("--dynamic-actor-association is only valid for a v1 dynamic asset")
    asset = build_canonical_actor_asset(
        scene, trajectories, dynamic_path=args.dynamic_gaussians, eligible_track_ids=set(trajectories), voxel_size_m=args.voxel_size_m
    )
    asset.save(args.output)
    track_manifest = args.track_manifest_output or args.output.with_suffix(".tracks.json")
    _write_track_manifest(track_manifest, tracks, asset.actor_ids, dynamic_path=args.dynamic_gaussians)
    print(
        f"Wrote {asset.count:,} canonical Gaussians for {len(asset.actor_ids)} vehicle actor(s): {args.output}",
        flush=True,
    )
    print(f"Wrote frozen canonical actor tracks: {track_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
