"""Convert one LiDAR-audited MASt3R metric-depth patch into a safe actor layer.

The dense audit produces many more points than descriptor triangulation, but
only for one source camera and one reference time.  This bridge deliberately
keeps that limitation in the output metadata and lets the renderer fail closed
outside that reference frame.  It is therefore an A/B diagnostic asset, not a
complete dynamic actor or an L4 sequence input.
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


SCHEMA = "ncore-mast3r-metric-dense-surface-layer"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Mast3rDenseSurfaceLayerOptions:
    dataset_manifest: Path
    dense_audit: Path
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


def build_mast3r_dense_surface_layer(options: Mast3rDenseSurfaceLayerOptions) -> Path:
    """Fuse a single passing metric-depth audit into a visibility-bound actor."""
    manifest_path = options.dataset_manifest.resolve()
    audit_path = options.dense_audit.resolve()
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest, audit = _read_json(manifest_path), _read_json(audit_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset manifest is not a completed dynamic reconstruction dataset")
    if audit.get("schema") != DENSE_AUDIT_SCHEMA or audit.get("status") != "complete" or not audit.get("accepted", False):
        raise ValueError("dense audit must be a completed, accepted MASt3R metric-depth audit")
    if int(audit.get("track_id", -1)) != options.track_id:
        raise ValueError("dense audit track does not match --track-id")
    reference_index = int(audit.get("reference_index", -1))
    if reference_index < 0:
        raise ValueError("dense audit has no valid reference index")
    tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("metric dense surface currently requires one listed rigid track")
    source_cameras = tuple(str(value) for value in audit.get("camera_ids", ()))
    if len(source_cameras) != 2:
        raise ValueError("dense audit must record exactly two camera IDs")
    asset_path = Path(str(audit.get("asset", ""))).resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"dense audit asset does not exist: {asset_path}")
    with np.load(asset_path, allow_pickle=False) as data:
        required = {"points_actor_local", "colors", "confidence", "reference_index"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dense audit asset missing: {', '.join(sorted(missing))}")
        positions = np.asarray(data["points_actor_local"], np.float32)
        colors = np.asarray(data["colors"], np.float32)
        confidence = np.asarray(data["confidence"], np.float32)
        asset_reference = np.asarray(data["reference_index"], np.int32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or colors.shape != positions.shape or confidence.shape != (len(positions),):
        raise ValueError("dense audit arrays must contain matching points/colors/confidence")
    if not len(positions) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(confidence)):
        raise ValueError("dense audit has no finite accepted surface points")
    if len(asset_reference) != 1 or int(asset_reference[0]) != reference_index:
        raise ValueError("dense audit report and NPZ reference indices disagree")
    # The audit samples colour from camera_a.  Preserve that provenance: it is
    # not a synthetic multi-view texture average.
    colors = np.clip(colors / 255.0, 0.0, 1.0)
    references = np.full(len(positions), reference_index, np.int32)
    local, rgb, visibility, source_count = fuse_observed_surface(
        positions, colors, references, confidence, voxel_size_m=options.voxel_size_m,
    )
    if not len(local):
        raise RuntimeError("metric dense voxel fusion produced no surface")
    trajectory = ActorTrajectory(
        timestamps_us=np.asarray(track["timestamps_us"], np.int64),
        actor_to_world=np.asarray(track["actor_to_world"], np.float32),
    )
    asset = CanonicalActorAsset(
        positions=local,
        rotations_wxyz=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0), np.float32), (len(local), 1)),
        scales=np.full((len(local), 3), options.gaussian_scale_m, np.float32),
        rgb=rgb,
        opacities=np.full(len(local), options.opacity, np.float32),
        actor_indices=np.zeros(len(local), np.int32),
        visibility_counts=visibility,
        source_counts=source_count,
        actor_ids=(options.track_id,),
        trajectories=(trajectory,),
        metadata={
            "schema": CANONICAL_SCHEMA,
            "version": CANONICAL_VERSION,
            "surface_layer_schema": SCHEMA,
            "surface_layer_version": VERSION,
            "coordinate_frame": "actor_local",
            "initialization": "mast3r_metric_dense_depth_lidar_cuboid_audited_surface_only",
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _sha256(manifest_path),
            "dense_audit": str(audit_path),
            "dense_audit_sha256": _sha256(audit_path),
            "dense_audit_asset": str(asset_path),
            "source_camera": source_cameras[0],
            "support_camera": source_cameras[1],
            "reference_index": reference_index,
            # Enforced by GsplatRenderer using the NCore-source index encoded
            # in CameraFrame.frame_id.  This must remain a singleton until a
            # separate observation has independently passed the audit.
            "visibility_reference_indices": [reference_index],
            "unknown_surface_not_rendered": True,
            "static_ply_used": False,
            "not_target_l4_production": True,
            "voxel_size_m": options.voxel_size_m,
            "gaussian_scale_m": options.gaussian_scale_m,
            "opacity": options.opacity,
        },
    )
    asset.save(output)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "complete",
        "asset": str(output),
        "input": {"dataset_manifest": str(manifest_path), "dense_audit": str(audit_path), "track_id": options.track_id},
        "visibility": {"source_camera": source_cameras[0], "support_camera": source_cameras[1], "reference_indices": [reference_index], "renderer_enforced": True},
        "counts": {"input_dense_points": int(len(positions)), "fused_surface_voxels": int(len(local))},
        "limitations": [
            "Only one source-observed reference time is enabled; all other frames are transparent by renderer policy.",
            "Unknown vehicle surfaces remain absent. This asset is for source-view A/B auditing, not L4 or sequence production.",
        ],
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote MASt3R metric dense observed-surface actor: {output} ({len(local):,} voxels)", flush=True)
    return output
