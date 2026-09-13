"""Source-provenance audit for InstantNuRec dynamic Gaussian v2 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamic import DynamicGaussianScene, infer_dynamic_source_slots, load_dynamic_gaussians
from .render import _write_json_atomic


def build_dynamic_forensics(scene: DynamicGaussianScene, *, voxel_size_m: float = 0.10) -> dict[str, object]:
    """Summarise track×source-slot evidence without inferring missing labels.

    Pixel-level MAE/alpha/depth and LiDAR overlap are deliberately reported as
    ``pending_render_audit`` here: they require the exact target camera and
    renderer outputs and must not be fabricated from point counts.
    """
    if scene.source_track_ids is None or scene.association_sources is None or scene.actor_local_positions is None:
        raise ValueError("dynamic forensics requires v2 provenance fields")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    slots, slot_times = infer_dynamic_source_slots(scene)
    valid = scene.source_track_ids >= 0
    rows: list[dict[str, object]] = []
    by_track_slot: dict[tuple[int, int], np.ndarray] = {}
    for track_id in np.unique(scene.source_track_ids[valid]):
        track = scene.source_track_ids == track_id
        for slot in np.unique(slots[track]):
            member = track & (slots == slot)
            by_track_slot[(int(track_id), int(slot))] = np.flatnonzero(member)
            provenance = scene.association_sources[member]
            rows.append({
                "track_id": int(track_id), "source_slot": int(slot),
                "source_timestamp_us": int(slot_times[slot]), "gaussian_count": int(member.sum()),
                "point_in_cuboid_count": int((provenance == 1).sum()),
                "ray_cuboid_fallback_count": int((provenance == 2).sum()),
                "source_camera_indices": [int(value) for value in np.unique(scene.source_camera_indices[member])] if scene.source_camera_indices is not None else [],
                "rendered_pixel_count": None, "alpha_contribution": None, "source_image_mae": None,
                "static_overlap_alpha": None, "static_depth_difference_m": None, "color_residual": None,
                "pixel_audit_status": "pending_render_audit",
            })
    tracks: list[dict[str, object]] = []
    for track_id in np.unique(scene.source_track_ids[valid]):
        member = np.flatnonzero(scene.source_track_ids == track_id)
        local = scene.actor_local_positions[member]
        voxels = np.floor(local / voxel_size_m).astype(np.int32)
        _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
        slot_count_per_voxel = np.asarray([len(np.unique(slots[member][inverse == index])) for index in range(len(counts))])
        duplicate_excess = int(np.maximum(counts - 1, 0).sum())
        fallback = float((scene.association_sources[member] == 2).mean())
        duplicate_ratio = duplicate_excess / max(len(member), 1)
        tags: list[str] = []
        if duplicate_ratio >= 0.25 and int((slot_count_per_voxel >= 2).sum()) > 0:
            tags.append("dynamic_cross_slot_replicas_likely")
        if fallback >= 0.20:
            tags.append("track_box_or_depth_association_uncertain")
        if len(np.unique(slots[member])) <= 1:
            tags.append("single_slot_appearance_or_depth_limited")
        tracks.append({
            "track_id": int(track_id), "gaussian_count": int(len(member)),
            "source_slots": [int(value) for value in np.unique(slots[member])],
            "source_cameras": [int(value) for value in np.unique(scene.source_camera_indices[member])] if scene.source_camera_indices is not None else [],
            "point_in_cuboid_fraction": float((scene.association_sources[member] == 1).mean()),
            "ray_cuboid_fallback_fraction": fallback,
            "canonical_voxel_size_m": voxel_size_m, "canonical_voxel_count": int(len(counts)),
            "cross_slot_duplicate_voxels": int((slot_count_per_voxel >= 2).sum()),
            "cross_slot_duplicate_excess_gaussians": duplicate_excess,
            "cross_slot_duplicate_ratio": float(duplicate_ratio), "classification": tags,
        })
    track_102 = next((row for row in tracks if row["track_id"] == 102), None)
    return {
        "schema_version": 1, "dynamic_asset_version": int(scene.metadata["version"]),
        "gaussian_count": scene.count, "source_slot_count": len(slot_times),
        "source_slot_timestamps_us": slot_times.astype(int).tolist(),
        "association": {
            "point_in_cuboid": int((scene.association_sources == 1).sum()),
            "ray_cuboid_fallback": int((scene.association_sources == 2).sum()),
            "unassociated": int((scene.association_sources == 0).sum()),
        },
        "track_slot_rows": rows, "tracks": tracks,
        "protruding_object_track_102": {
            "audit_required": True,
            "optimization_default": "excluded from rigid vehicle fusion",
            "source_provenance": track_102,
        },
        "required_followup": [
            "render each source slot and actor separately at representative target frames",
            "intersect projected cuboids with high-confidence local vehicle segmentation before scoring",
            "project lidar_top_360fov depth and compute static/dynamic alpha-depth overlap only inside that conservative mask",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v2 dynamic Gaussian source provenance and cross-slot duplication")
    parser.add_argument("--dynamic-gaussians", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size-m", type=float, default=0.10)
    args = parser.parse_args(argv)
    report = build_dynamic_forensics(load_dynamic_gaussians(args.dynamic_gaussians), voxel_size_m=args.voxel_size_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ("track_slot_rows", "tracks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
