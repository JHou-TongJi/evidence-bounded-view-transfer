"""Fuse independently audited metric-dense surface observations conservatively.

The primary source camera owns every actor-local voxel it observes.  Secondary
views only add voxels absent from the primary grid; their colours are never
averaged into primary surfaces.  This avoids reintroducing cross-camera colour
smear while still recovering genuinely complementary vehicle faces.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, ActorTrajectory, CanonicalActorAsset
from .mast3r_dense_depth import DATASET_SCHEMA, SCHEMA as DENSE_AUDIT_SCHEMA
from .mast3r_surface_layer import fuse_observed_surface


SCHEMA = "ncore-mast3r-metric-dense-surface-priority-fusion"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def secondary_missing_primary_voxels(primary: np.ndarray, secondary: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Mask secondary points that occupy an actor-local voxel absent in primary."""
    first = np.asarray(primary, np.float32)
    second = np.asarray(secondary, np.float32)
    if first.ndim != 2 or first.shape[1] != 3 or second.ndim != 2 or second.shape[1] != 3:
        raise ValueError("primary and secondary positions must both be [N,3]")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    first_keys = {tuple(key) for key in np.floor(first / voxel_size_m).astype(np.int32)}
    return np.asarray([tuple(key) not in first_keys for key in np.floor(second / voxel_size_m).astype(np.int32)], bool)


@dataclass(frozen=True)
class Mast3rDenseSurfaceFuseOptions:
    dataset_manifest: Path
    primary_dense_audit: Path
    secondary_dense_audit: Path
    output: Path
    track_id: int
    voxel_size_m: float = .025
    gaussian_scale_m: float = .018
    opacity: float = .72

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.voxel_size_m <= 0 or self.gaussian_scale_m <= 0:
            raise ValueError("track/voxel/scale parameters are invalid")
        if not 0 < self.opacity <= 1 or self.output.suffix != ".npz":
            raise ValueError("opacity/output parameters are invalid")


def _audit_points(path: Path, track_id: int, reference: int | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, int, str]:
    audit = _read_json(path)
    if audit.get("schema") != DENSE_AUDIT_SCHEMA or audit.get("status") != "complete" or not audit.get("accepted", False):
        raise ValueError(f"dense audit must be completed and accepted: {path}")
    if int(audit.get("track_id", -1)) != track_id:
        raise ValueError(f"dense audit track mismatch: {path}")
    audit_reference = int(audit.get("reference_index", -1))
    if reference is not None and audit_reference != reference:
        raise ValueError("all dense audits must use the same reference index")
    cameras = tuple(str(value) for value in audit.get("camera_ids", ()))
    if len(cameras) != 2:
        raise ValueError("dense audit must record two camera IDs")
    asset = Path(str(audit.get("asset", ""))).resolve()
    with np.load(asset, allow_pickle=False) as data:
        points = np.asarray(data["points_actor_local"], np.float32)
        colors = np.asarray(data["colors"], np.float32) / 255.0
        confidence = np.asarray(data["confidence"], np.float32)
        asset_reference = np.asarray(data["reference_index"], np.int32).reshape(-1)
    if len(asset_reference) != 1 or int(asset_reference[0]) != audit_reference:
        raise ValueError("dense audit JSON/NPZ reference indices disagree")
    if not len(points) or points.shape[1:] != (3,) or colors.shape != points.shape or confidence.shape != (len(points),):
        raise ValueError("dense audit asset has invalid point/color/confidence fields")
    return audit, points, np.clip(colors, 0., 1.), confidence, audit_reference, cameras[0]


def fuse_mast3r_dense_surfaces(options: Mast3rDenseSurfaceFuseOptions) -> Path:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest_path = options.dataset_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset manifest is not a completed dynamic reconstruction dataset")
    tracks = {int(item["track_id"]): item for item in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("priority fusion requires one listed rigid track")
    primary_path, secondary_path = options.primary_dense_audit.resolve(), options.secondary_dense_audit.resolve()
    primary_audit, primary_points, primary_rgb, primary_conf, reference, primary_camera = _audit_points(primary_path, options.track_id, None)
    secondary_audit, secondary_points, secondary_rgb, secondary_conf, secondary_reference, secondary_camera = _audit_points(secondary_path, options.track_id, reference)
    if primary_camera == secondary_camera:
        raise ValueError("primary and secondary dense audits must have different source cameras")
    keep_secondary = secondary_missing_primary_voxels(primary_points, secondary_points, options.voxel_size_m)
    positions = np.concatenate((primary_points, secondary_points[keep_secondary]), axis=0)
    colors = np.concatenate((primary_rgb, secondary_rgb[keep_secondary]), axis=0)
    confidence = np.concatenate((primary_conf, secondary_conf[keep_secondary]), axis=0)
    references = np.full(len(positions), reference, np.int32)
    local, rgb, visibility, source_count = fuse_observed_surface(positions, colors, references, confidence, voxel_size_m=options.voxel_size_m)
    trajectory = ActorTrajectory(np.asarray(track["timestamps_us"], np.int64), np.asarray(track["actor_to_world"], np.float32))
    asset = CanonicalActorAsset(
        positions=local,
        rotations_wxyz=np.tile(np.asarray((1., 0., 0., 0.), np.float32), (len(local), 1)),
        scales=np.full((len(local), 3), options.gaussian_scale_m, np.float32), rgb=rgb,
        opacities=np.full(len(local), options.opacity, np.float32), actor_indices=np.zeros(len(local), np.int32),
        visibility_counts=visibility, source_counts=source_count, actor_ids=(options.track_id,), trajectories=(trajectory,),
        metadata={
            "schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION,
            "surface_layer_schema": SCHEMA, "surface_layer_version": VERSION,
            "coordinate_frame": "actor_local", "initialization": "mast3r_metric_dense_depth_priority_source_fusion",
            "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": _sha256(manifest_path),
            "primary_dense_audit": str(primary_path), "primary_dense_audit_sha256": _sha256(primary_path),
            "secondary_dense_audit": str(secondary_path), "secondary_dense_audit_sha256": _sha256(secondary_path),
            "primary_source_camera": primary_camera, "secondary_source_camera": secondary_camera,
            "reference_index": reference, "visibility_reference_indices": [reference],
            "colour_policy": "primary voxel colour retained; secondary colour only fills primary-absent actor-local voxels",
            "unknown_surface_not_rendered": True, "static_ply_used": False, "not_target_l4_production": True,
            "voxel_size_m": options.voxel_size_m, "gaussian_scale_m": options.gaussian_scale_m, "opacity": options.opacity,
        },
    )
    asset.save(output)
    report = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "asset": str(output),
        "inputs": {"dataset_manifest": str(manifest_path), "primary_dense_audit": str(primary_path), "secondary_dense_audit": str(secondary_path), "track_id": options.track_id},
        "source_policy": {"primary_source_camera": primary_camera, "secondary_source_camera": secondary_camera, "reference_index": reference, "visibility_reference_indices": [reference]},
        "counts": {"primary_input_points": int(len(primary_points)), "secondary_input_points": int(len(secondary_points)), "secondary_points_in_primary_absent_voxels": int(keep_secondary.sum()), "fused_surface_voxels": int(len(local))},
        "limitations": ["Secondary observations never modify a primary-observed voxel colour.", "Only reference-time 140 is rendered; unobserved vehicle surfaces remain absent.", "This is a source-view fusion audit asset, not a dynamic sequence or L4 production actor."],
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote priority-fused MASt3R dense surface actor: {output} ({len(local):,} voxels; secondary additions={int(keep_secondary.sum()):,})", flush=True)
    return output
