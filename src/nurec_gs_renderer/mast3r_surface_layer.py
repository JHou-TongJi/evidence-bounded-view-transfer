"""Build a conservative, observed-surface-only Gaussian actor from MASt3R anchors.

This is deliberately a diagnostic bridge, not a dynamic-object reconstructor.
It turns only FTheta/rolling-shutter validated MASt3R points into compact
canonical Gaussians, samples their colour from the two original source images,
and leaves every unobserved surface absent.  The resulting asset is suitable
only for short source-view compositing audits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .canonical_actor import (
    CANONICAL_SCHEMA,
    CANONICAL_VERSION,
    ActorTrajectory,
    CanonicalActorAsset,
)
from .dynamic_mask_crossview_audit import project_world_points_native_rolling
from .mast3r_correspondence import DATASET_SCHEMA, SCHEMA as CORRESPONDENCE_SCHEMA
from .pose_sources import _create_ncore_loader
from .roma_multiview_audit import pixels_in_mask
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-mast3r-observed-surface-layer"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((item for item in record["instances"] if int(item["track_id"]) == track_id), None)


def _bilinear_rgb(image: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample an RGB image at subpixel positions without extrapolating its rim."""
    image = np.asarray(image, np.uint8)
    xy = np.asarray(pixels, np.float32)
    height, width = image.shape[:2]
    x0, y0 = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    output = np.zeros((len(xy), 3), np.float32)
    if valid.any():
        x, y = xy[valid, 0] - x0[valid], xy[valid, 1] - y0[valid]
        c00, c10 = image[y0[valid], x0[valid]], image[y0[valid], x1[valid]]
        c01, c11 = image[y1[valid], x0[valid]], image[y1[valid], x1[valid]]
        output[valid] = (
            c00 * ((1 - x) * (1 - y))[:, None] + c10 * (x * (1 - y))[:, None]
            + c01 * ((1 - x) * y)[:, None] + c11 * (x * y)[:, None]
        )
    return output / 255.0, valid


def fuse_observed_surface(
    positions: np.ndarray,
    colors: np.ndarray,
    references: np.ndarray,
    confidence: np.ndarray,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-weighted voxel fusion, retaining only source-observed points."""
    positions = np.asarray(positions, np.float32)
    colors = np.asarray(colors, np.float32)
    references = np.asarray(references, np.int32)
    confidence = np.asarray(confidence, np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or colors.shape != positions.shape:
        raise ValueError("positions/colors must have matching [N,3] shape")
    if references.shape != (len(positions),) or confidence.shape != (len(positions),):
        raise ValueError("references/confidence must have one entry per point")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    keys = np.floor(positions / voxel_size_m).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    weight = np.clip(confidence, 1e-3, None)
    summed = np.bincount(inverse, weights=weight, minlength=count).astype(np.float32)
    fused_position = np.stack([
        np.bincount(inverse, weights=positions[:, axis] * weight, minlength=count) / summed
        for axis in range(3)
    ], axis=1).astype(np.float32)
    fused_color = np.stack([
        np.bincount(inverse, weights=colors[:, axis] * weight, minlength=count) / summed
        for axis in range(3)
    ], axis=1).astype(np.float32)
    # A visibility count is a genuine temporal source-observation count, not
    # an inferred novel-view visibility label.
    visibility = np.zeros(count, np.uint16)
    source_count = np.bincount(inverse, minlength=count).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    for cell in range(count):
        visibility[cell] = min(len(np.unique(references[inverse == cell])), np.iinfo(np.uint16).max)
    return fused_position, fused_color.clip(0.0, 1.0), visibility, source_count


@dataclass(frozen=True)
class Mast3rSurfaceLayerOptions:
    dataset_manifest: Path
    correspondence: Path
    output: Path
    track_id: int
    voxel_size_m: float = .05
    gaussian_scale_m: float = .035
    opacity: float = .65
    source_camera: str | None = None
    reference_index: int | None = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.voxel_size_m <= 0 or self.gaussian_scale_m <= 0:
            raise ValueError("track/voxel/scale parameters are invalid")
        if not 0 < self.opacity <= 1 or self.output.suffix != ".npz":
            raise ValueError("opacity/output parameters are invalid")
        if self.reference_index is not None and self.reference_index < 0:
            raise ValueError("reference_index must be non-negative")


def build_mast3r_surface_layer(options: Mast3rSurfaceLayerOptions) -> Path:
    """Write a canonical actor containing only coloured, observed MASt3R surface voxels."""
    manifest_path, correspondence_path, output = (
        options.dataset_manifest.resolve(), options.correspondence.resolve(), options.output.resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest, correspondence = _read_json(manifest_path), _read_json(correspondence_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset manifest is not a completed dynamic reconstruction dataset")
    if correspondence.get("schema") != CORRESPONDENCE_SCHEMA or correspondence.get("status") != "complete":
        raise ValueError("correspondence must be a completed MASt3R FTheta asset")
    if int(correspondence.get("track_id", -1)) != options.track_id:
        raise ValueError("correspondence track does not match --track-id")
    if correspondence.get("dataset_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("correspondence does not bind this dataset manifest")
    tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("MASt3R observed surface currently requires one listed rigid track")
    asset_path = Path(str(correspondence["asset"])).resolve()
    with np.load(asset_path, allow_pickle=False) as data:
        local = np.asarray(data["points_actor_local"], np.float32)
        references = np.asarray(data["reference_frame_indices"], np.int32)
        confidence = np.asarray(data["confidence"], np.float32)
    if not len(local):
        raise ValueError("MASt3R asset contains no accepted points")

    cameras_by_reference = {int(frame["reference_frame_index"]): {str(record["camera_id"]): record for record in frame["cameras"]} for frame in manifest["frames"]}
    camera_ids = tuple(str(value) for value in correspondence["camera_ids"])
    if len(camera_ids) != 2:
        raise ValueError("surface layer expects exactly the two correspondence source cameras")
    if options.source_camera is not None and options.source_camera not in camera_ids:
        raise ValueError("--source-camera must be one of the correspondence source cameras")
    if options.reference_index is not None:
        keep_reference = references == options.reference_index
        if not keep_reference.any():
            raise ValueError("--reference-index has no accepted MASt3R anchors")
        local, references, confidence = local[keep_reference], references[keep_reference], confidence[keep_reference]
    active_cameras = camera_ids if options.source_camera is None else (options.source_camera,)
    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    from ncore.impl.sensors.camera import FThetaCameraModel
    models: dict[str, Any] = {}
    colored = np.zeros((len(local), 3), np.float32)
    color_weight = np.zeros(len(local), np.float32)
    accepted_rgb = np.zeros(len(local), bool)

    for reference in np.unique(references).tolist():
        selected = np.flatnonzero(references == reference)
        records = cameras_by_reference.get(int(reference))
        if records is None:
            raise ValueError(f"reference {reference} is absent from the dataset manifest")
        first, second = (records.get(camera_id) for camera_id in camera_ids)
        if first is None or second is None:
            raise ValueError(f"reference {reference} lacks a correspondence source camera")
        timestamp = (int(first["timestamp_midpoint_us"]) + int(second["timestamp_midpoint_us"])) // 2
        pose = _interpolate_track_pose(track, timestamp)
        if pose is None:
            raise ValueError(f"track {options.track_id} has no pose at reference {reference}")
        world = local[selected] @ pose[:3, :3].T + pose[:3, 3]
        for camera_id in active_cameras:
            record = records[camera_id]
            sensor = loader.get_camera_sensor(camera_id)
            model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            valid_index, pixels, _ = project_world_points_native_rolling(model, world, record)
            if not len(valid_index):
                continue
            instance = _instance(record, options.track_id)
            if instance is None:
                continue
            from PIL import Image
            image = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), np.uint8)
            mask = np.asarray(Image.open(manifest_path.parent / str(instance["mask"])).convert("L"), np.uint8) > 0
            inside = pixels_in_mask(mask, pixels)
            colors, sample_valid = _bilinear_rgb(image, pixels.astype(np.float32))
            kept = inside & sample_valid
            indices = selected[valid_index[kept]]
            colored[indices] += colors[kept]
            color_weight[indices] += 1.0
            accepted_rgb[indices] = True

    if not accepted_rgb.any():
        raise RuntimeError("no MASt3R anchors reprojected into their original masked source images")
    local, colored, visibility, source_count = fuse_observed_surface(
        local[accepted_rgb], colored[accepted_rgb] / color_weight[accepted_rgb, None],
        references[accepted_rgb], confidence[accepted_rgb], voxel_size_m=options.voxel_size_m,
    )
    trajectory = ActorTrajectory(
        timestamps_us=np.asarray(track["timestamps_us"], np.int64),
        actor_to_world=np.asarray(track["actor_to_world"], np.float32),
    )
    asset = CanonicalActorAsset(
        positions=local, rotations_wxyz=np.tile(np.asarray((1., 0., 0., 0.), np.float32), (len(local), 1)),
        scales=np.full((len(local), 3), options.gaussian_scale_m, np.float32), rgb=colored,
        opacities=np.full(len(local), options.opacity, np.float32), actor_indices=np.zeros(len(local), np.int32),
        visibility_counts=visibility, source_counts=source_count, actor_ids=(options.track_id,), trajectories=(trajectory,),
        metadata={
            "schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION,
            "surface_layer_schema": SCHEMA, "surface_layer_version": VERSION,
            "coordinate_frame": "actor_local", "initialization": "mast3r_ftheta_rolling_observed_surface_only",
            "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": _sha256(manifest_path),
            "correspondence": str(correspondence_path), "correspondence_sha256": _sha256(correspondence_path),
            "accepted_anchor_count": int(len(accepted_rgb.nonzero()[0])), "fused_surface_voxel_count": int(len(local)),
            "voxel_size_m": options.voxel_size_m, "gaussian_scale_m": options.gaussian_scale_m,
            "opacity": options.opacity, "source_cameras": list(active_cameras),
            "reference_index": options.reference_index, "unknown_surface_not_rendered": True,
            "static_ply_used": False,
        },
    )
    asset.save(output)
    report = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "asset": str(output),
        "input": {"dataset_manifest": str(manifest_path), "correspondence": str(correspondence_path), "track_id": options.track_id},
        "counts": {"input_anchors": int(len(references)), "anchors_with_source_colour": int(accepted_rgb.sum()), "fused_surface_voxels": int(len(local))},
        "limitations": [
            "Only MASt3R anchors reprojecting inside their original source instance masks are rendered.",
            "Unobserved vehicle surfaces remain absent; this is not a complete actor or a target-L4 production asset.",
            "The asset must first pass source-view alpha/depth/ROI and coverage audits before any target-camera use.",
        ],
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote MASt3R observed-surface canonical actor: {output} ({len(local):,} voxels from {accepted_rgb.sum():,} anchors)", flush=True)
    return output
