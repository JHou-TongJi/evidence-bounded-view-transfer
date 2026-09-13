"""Build one short-window rigid canonical surface from per-frame dense assets."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, CanonicalActorAsset
from .mast3r_dense_surface_fuse import secondary_missing_primary_voxels
from .mast3r_surface_layer import fuse_observed_surface


SCHEMA = "ncore-mast3r-metric-dense-surface-temporal-priority-fusion"
VERSION = 1


def ordered_assets_by_reference(assets: list[CanonicalActorAsset], priority_reference: int) -> list[CanonicalActorAsset]:
    """Place the requested source time first while retaining deterministic order."""
    values = [(int(asset.metadata.get("reference_index", -1)), asset) for asset in assets]
    if any(reference < 0 for reference, _ in values):
        raise ValueError("all surface assets need a scalar reference_index metadata field")
    if len({reference for reference, _ in values}) != len(values):
        raise ValueError("temporal fusion needs one surface asset per unique reference")
    if priority_reference not in {reference for reference, _ in values}:
        raise ValueError("priority reference is absent from the supplied assets")
    return [asset for reference, asset in values if reference == priority_reference] + [asset for reference, asset in sorted(values) if reference != priority_reference]


@dataclass(frozen=True)
class Mast3rDenseSurfaceTemporalFuseOptions:
    actors: tuple[Path, ...]
    output: Path
    track_id: int
    priority_reference: int
    voxel_size_m: float = .025
    gaussian_scale_m: float = .018
    opacity: float = .72

    def __post_init__(self) -> None:
        if len(self.actors) < 2 or self.track_id < 0 or self.voxel_size_m <= 0 or self.gaussian_scale_m <= 0:
            raise ValueError("actors/track/voxel/scale parameters are invalid")
        if self.output.suffix != ".npz" or not 0 < self.opacity <= 1:
            raise ValueError("output/opacity parameters are invalid")


def fuse_mast3r_dense_surfaces_temporally(options: Mast3rDenseSurfaceTemporalFuseOptions) -> Path:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    paths = tuple(path.resolve() for path in options.actors)
    loaded = [CanonicalActorAsset.load(path) for path in paths]
    for asset in loaded:
        if asset.actor_ids != (options.track_id,):
            raise ValueError("each temporal input must contain exactly the requested actor track")
        if asset.metadata.get("unknown_surface_not_rendered") is not True:
            raise ValueError("temporal fusion accepts observed-surface-only actors only")
    ordered = ordered_assets_by_reference(loaded, options.priority_reference)
    canonical = ordered[0]
    if any(not np.array_equal(asset.trajectories[0].timestamps_us, canonical.trajectories[0].timestamps_us) or not np.allclose(asset.trajectories[0].actor_to_world, canonical.trajectories[0].actor_to_world, atol=1e-5) for asset in ordered[1:]):
        raise ValueError("temporal surface assets must share the same frozen track trajectory")
    positions = canonical.positions.copy()
    colors = canonical.rgb.copy()
    confidence = np.maximum(canonical.source_counts.astype(np.float32), 1.)
    references = np.full(len(positions), int(canonical.metadata["reference_index"]), np.int32)
    additions: list[dict[str, int]] = []
    for asset in ordered[1:]:
        keep = secondary_missing_primary_voxels(positions, asset.positions, options.voxel_size_m)
        additions.append({"reference_index": int(asset.metadata["reference_index"]), "input_gaussians": int(asset.count), "primary_absent_voxel_gaussians": int(keep.sum())})
        positions = np.concatenate((positions, asset.positions[keep]), axis=0)
        colors = np.concatenate((colors, asset.rgb[keep]), axis=0)
        confidence = np.concatenate((confidence, np.maximum(asset.source_counts[keep].astype(np.float32), 1.)), axis=0)
        references = np.concatenate((references, np.full(int(keep.sum()), int(asset.metadata["reference_index"]), np.int32)), axis=0)
    local, rgb, visibility, source_count = fuse_observed_surface(positions, colors, references, confidence, voxel_size_m=options.voxel_size_m)
    reference_indices = [int(asset.metadata["reference_index"]) for asset in ordered]
    result = CanonicalActorAsset(
        positions=local,
        rotations_wxyz=np.tile(np.asarray((1., 0., 0., 0.), np.float32), (len(local), 1)),
        scales=np.full((len(local), 3), options.gaussian_scale_m, np.float32), rgb=rgb,
        opacities=np.full(len(local), options.opacity, np.float32), actor_indices=np.zeros(len(local), np.int32),
        visibility_counts=visibility, source_counts=source_count, actor_ids=(options.track_id,), trajectories=canonical.trajectories,
        metadata={
            "schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION,
            "surface_layer_schema": SCHEMA, "surface_layer_version": VERSION,
            "coordinate_frame": "actor_local", "initialization": "short_window_mast3r_metric_dense_rigid_canonical_priority_fusion",
            "track_id": options.track_id, "reference_index": options.priority_reference,
            "visibility_reference_indices": sorted(reference_indices), "priority_reference": options.priority_reference,
            "input_assets": [str(path) for path in paths], "input_references": reference_indices,
            "colour_policy": "priority reference owns occupied voxel colour; other times only fill priority-absent voxels",
            "unknown_surface_not_rendered": True, "static_ply_used": False, "not_target_l4_production": True,
            "voxel_size_m": options.voxel_size_m, "gaussian_scale_m": options.gaussian_scale_m, "opacity": options.opacity,
        },
    )
    result.save(output)
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "asset": str(output), "track_id": options.track_id, "references": sorted(reference_indices), "priority_reference": options.priority_reference, "counts": {"priority_input_gaussians": int(canonical.count), "later_additions": additions, "fused_surface_voxels": int(result.count)}, "limitations": ["Only the supplied reference indices are visible; intermediate/other frames are fail-closed transparent.", "This is rigid-actor temporal consistency experiment, not a complete dynamic reconstruction or L4 sequence asset."]}
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote temporal priority-fused MASt3R dense actor: {output} ({result.count:,} voxels; references={sorted(reference_indices)})", flush=True)
    return output
