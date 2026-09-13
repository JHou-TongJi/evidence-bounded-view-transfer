"""Read-only raw-LiDAR observability audit for a rigid NCore actor.

This is deliberately a gate before another actor optimiser.  It establishes
whether a local rigid surface can be supported by *different cameras* and by
an even/odd temporal split, without consulting a static PLY or creating a
Gaussian asset.  A 2-D instance mask alone is insufficient: every retained
LiDAR return is projected through the native FTheta camera and must also lie
inside the track cuboid at the LiDAR timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import interpolate_w2c
from .pose_sources import _create_ncore_loader
from .static_leakage import _project_world_points_ftheta
from .street_dataset import _interpolate_track_pose


TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"
SCHEMA = "ncore-rigid-object-layer-observability-audit"
VERSION = 1


@dataclass(frozen=True)
class ObjectLayerObservabilityOptions:
    trusted_manifest: Path
    output: Path
    track_id: int
    camera_ids: tuple[str, ...]
    reference_start: int
    reference_end: int
    device: str = "cuda"
    voxel_size_m: float = .10
    cuboid_margin_m: float = .15

    def __post_init__(self) -> None:
        if self.output.suffix != ".json" or self.track_id < 0:
            raise ValueError("output must be a .json path and track_id must be non-negative")
        if not self.camera_ids or len(self.camera_ids) != len(set(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique")
        if self.reference_start < 0 or self.reference_end <= self.reference_start:
            raise ValueError("reference range is invalid")
        if self.voxel_size_m <= 0 or self.cuboid_margin_m < 0:
            raise ValueError("voxel size and cuboid margin are invalid")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _midpoint_w2c(record: dict[str, Any]) -> np.ndarray:
    start = np.linalg.inv(np.asarray(record["camera_to_world_start"], np.float64))
    end = np.linalg.inv(np.asarray(record["camera_to_world_end"], np.float64))
    rotations, translations = interpolate_w2c(start, end, np.asarray([.5], np.float64))
    midpoint = np.eye(4, dtype=np.float64)
    midpoint[:3, :3], midpoint[:3, 3] = rotations[0], translations[0]
    return midpoint


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    values = np.floor(np.asarray(points, np.float32) / float(voxel_size_m)).astype(np.int64)
    return {tuple(int(component) for component in row) for row in values}


def _surface_probes(size_lwh: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic cuboid surface grid and its closest face labels."""
    half = np.asarray(size_lwh, np.float32) * .5
    axes = [np.arange(-extent, extent + voxel_size_m * .5, voxel_size_m, dtype=np.float32) for extent in half]
    points: list[np.ndarray] = []
    labels: list[str] = []
    for axis, name in enumerate(("x", "y", "z")):
        other = [index for index in range(3) if index != axis]
        first, second = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign, sign_name in ((-1., "-"), (1., "+")):
            surface = np.zeros((first.size, 3), np.float32)
            surface[:, axis] = sign * half[axis]
            surface[:, other[0]] = first.ravel()
            surface[:, other[1]] = second.ravel()
            points.append(surface)
            labels.extend([f"{sign_name}{name}"] * len(surface))
    return np.concatenate(points), np.asarray(labels, dtype=object)


def _surface_coverage(
    positions: np.ndarray, size_lwh: np.ndarray, voxel_size_m: float,
) -> dict[str, Any]:
    probes, labels = _surface_probes(size_lwh, voxel_size_m)
    if not len(positions):
        return {"probe_count": int(len(probes)), "observed_probe_count": 0, "fraction": 0., "by_face": {}}
    observed = np.zeros(len(probes), bool)
    threshold2 = float((voxel_size_m * 1.5) ** 2)
    # Keep the temporary array bounded if future actors use much denser LiDAR.
    for begin in range(0, len(probes), 1024):
        block = probes[begin:begin + 1024]
        distances2 = ((block[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1)
        observed[begin:begin + len(block)] = distances2.min(axis=1) <= threshold2
    by_face: dict[str, dict[str, float | int]] = {}
    for face in sorted(set(labels.tolist())):
        members = labels == face
        by_face[face] = {"probe_count": int(members.sum()), "observed_probe_count": int((observed & members).sum()),
                         "fraction": float(observed[members].mean())}
    return {"probe_count": int(len(probes)), "observed_probe_count": int(observed.sum()),
            "fraction": float(observed.mean()), "by_face": by_face}


def _voxel_representatives(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """One median local position per audit voxel, keeping coverage bounded."""
    values = np.asarray(points, np.float32)
    keys = np.floor(values / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    representatives = [np.median(values[inverse == index], axis=0) for index in range(int(inverse.max()) + 1)]
    return np.asarray(representatives, np.float32)


def summarize_voxel_support(
    points: np.ndarray, references: np.ndarray, cameras: np.ndarray, *, voxel_size_m: float,
) -> dict[str, Any]:
    """Summarize canonical voxel support, including the proposed even/odd split."""
    local = np.asarray(points, np.float32)
    reference = np.asarray(references, np.int64)
    camera = np.asarray(cameras, dtype=object)
    if local.ndim != 2 or local.shape[1] != 3 or reference.shape != (len(local),) or camera.shape != (len(local),):
        raise ValueError("point/reference/camera arrays are inconsistent")
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(np.floor(local / voxel_size_m).astype(np.int64)):
        groups.setdefault(tuple(int(value) for value in key), []).append(index)
    multi_observation = multi_camera = multi_reference = 0
    even_keys: set[tuple[int, int, int]] = set(); odd_keys: set[tuple[int, int, int]] = set()
    supports: list[dict[str, Any]] = []
    for key, members in groups.items():
        ids = np.asarray(members, np.int64)
        cameras_here = sorted(set(str(value) for value in camera[ids]))
        references_here = sorted(set(int(value) for value in reference[ids]))
        if len(ids) >= 2: multi_observation += 1
        if len(cameras_here) >= 2: multi_camera += 1
        if len(references_here) >= 2: multi_reference += 1
        if any(value % 2 == 0 for value in references_here): even_keys.add(key)
        if any(value % 2 == 1 for value in references_here): odd_keys.add(key)
        supports.append({"key": list(key), "associations": int(len(ids)), "camera_count": int(len(cameras_here)),
                         "reference_count": int(len(references_here)), "cameras": cameras_here,
                         "references": references_here})
    support_counts = np.asarray([entry["associations"] for entry in supports], np.int64)
    return {
        "voxel_size_m": float(voxel_size_m), "voxel_count": int(len(groups)),
        "multi_association_voxels": int(multi_observation), "independent_camera_voxels": int(multi_camera),
        "independent_reference_voxels": int(multi_reference),
        "even_reference_voxels": int(len(even_keys)), "odd_reference_voxels": int(len(odd_keys)),
        "even_odd_overlap_voxels": int(len(even_keys & odd_keys)),
        "support_per_voxel": {"min": int(support_counts.min()) if len(support_counts) else 0,
                              "median": float(np.median(support_counts)) if len(support_counts) else 0.,
                              "max": int(support_counts.max()) if len(support_counts) else 0.},
        # This list makes an unexpected positive gate auditable rather than a
        # black-box aggregate.  It contains only discrete local coordinates,
        # never source images or point clouds.
        "per_voxel_support": supports,
    }


def audit_object_layer_observability(options: ObjectLayerObservabilityOptions) -> Path:
    """Write a source-mask/raw-LiDAR observability report; never writes assets."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite observability report: {options.output}")
    manifest_path = options.trusted_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted_manifest must be a complete trusted raw supervision manifest")
    tracks = {int(track["track_id"]): track for track in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("track must be a rigid actor in the trusted manifest")
    available_cameras = set(str(value) for value in manifest.get("camera_ids", ()))
    if set(options.camera_ids).difference(available_cameras):
        raise ValueError("requested camera is not in trusted manifest")

    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    lidar = loader.get_lidar_sensor("lidar_top_360fov")
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    models: dict[str, Any] = {}
    clouds: dict[int, np.ndarray] = {}
    local_parts: list[np.ndarray] = []; reference_parts: list[np.ndarray] = []; camera_parts: list[np.ndarray] = []
    association_ids: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    raw_seen = mask_associations = inside_associations = 0
    selected_frames = 0

    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if not options.reference_start <= reference < options.reference_end:
            continue
        selected_frames += 1
        for record in frame["cameras"]:
            camera_id = str(record["camera_id"])
            if camera_id not in options.camera_ids:
                continue
            instance = next((value for value in record["instances"]
                             if int(value["track_id"]) == options.track_id and value.get("mask_origin") == "source"), None)
            if instance is None:
                continue
            lidar_index, lidar_time = record.get("lidar_frame_index"), record.get("lidar_timestamp_us")
            if lidar_index is None or lidar_time is None:
                continue
            lidar_index = int(lidar_index); lidar_time = int(lidar_time)
            sensor = loader.get_camera_sensor(camera_id)
            model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            source_index = int(record["source_frame_index"])
            source = np.asarray(sensor.get_frame_image_array(source_index), np.uint8)
            if lidar_index not in clouds:
                cloud = lidar.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                transform = np.asarray(lidar.get_frames_T_sensor_target("world", lidar_index), np.float64)
                clouds[lidar_index] = np.asarray(cloud, np.float32) @ transform[:3, :3].T + transform[:3, 3]
            world = clouds[lidar_index]
            raw_seen += len(world)
            mask = np.asarray(Image.open(roots[str(instance["mask_root"])] / str(instance["mask"])).convert("L"), np.uint8) > 0
            if mask.shape != source.shape[:2]:
                raise ValueError(f"mask/image shape mismatch: {camera_id} ref={reference}")
            pixels, _, valid = _project_world_points_ftheta(
                world, _midpoint_w2c(record), model, source_width=source.shape[1], source_height=source.shape[0],
                width=source.shape[1], height=source.shape[0], device=options.device,
            )
            xx = np.where(valid, np.rint(pixels[:, 0]), -1).astype(np.int64)
            yy = np.where(valid, np.rint(pixels[:, 1]), -1).astype(np.int64)
            valid &= (xx >= 0) & (xx < source.shape[1]) & (yy >= 0) & (yy < source.shape[0])
            chosen = np.flatnonzero(valid)
            chosen = chosen[mask[yy[chosen], xx[chosen]]]
            mask_associations += len(chosen)
            pose = _interpolate_track_pose(track, lidar_time)
            if pose is None:
                continue
            local = (world[chosen] - pose[:3, 3].astype(np.float32)) @ pose[:3, :3].astype(np.float32)
            half = np.asarray(track["length_width_height"], np.float32) * .5 + options.cuboid_margin_m
            inside = np.all(np.abs(local) <= half[None], axis=1)
            chosen, local = chosen[inside], local[inside]
            inside_associations += len(chosen)
            association_ids.update((lidar_index, int(index)) for index in chosen)
            if len(chosen):
                local_parts.append(local.astype(np.float32))
                reference_parts.append(np.full(len(local), reference, np.int64))
                camera_parts.append(np.full(len(local), camera_id, dtype=object))
            records.append({"reference_frame_index": reference, "camera_id": camera_id, "source_frame_index": source_index,
                            "lidar_frame_index": lidar_index, "mask_lidar_associations": int(len(chosen) + int((~inside).sum())),
                            "inside_cuboid_associations": int(len(chosen)), "rejected_2d_mask_background_associations": int((~inside).sum())})
            print(f"Observability ref={reference:03d} camera={camera_id}: "
                  f"mask={int(len(chosen) + int((~inside).sum()))} inside={len(chosen)}", flush=True)

    if not local_parts:
        raise RuntimeError("no source-mask LiDAR association survives the strict 3-D cuboid gate")
    local = np.concatenate(local_parts); references = np.concatenate(reference_parts); cameras = np.concatenate(camera_parts)
    voxel_summary = summarize_voxel_support(local, references, cameras, voxel_size_m=options.voxel_size_m)
    # Surface coverage is explicitly a voxel-scale observability measure.  Do
    # not form an N-probe × N-association matrix: a return seen by many cameras
    # represents one local surface location, and direct association count can
    # reach hundreds of thousands in this 22-frame audit.
    local_all_voxels = _voxel_representatives(local, options.voxel_size_m)
    local_even_voxels = _voxel_representatives(local[references % 2 == 0], options.voxel_size_m)
    local_odd_voxels = _voxel_representatives(local[references % 2 == 1], options.voxel_size_m)
    surface_all = _surface_coverage(local_all_voxels, np.asarray(track["length_width_height"], np.float32), options.voxel_size_m)
    surface_even = _surface_coverage(local_even_voxels, np.asarray(track["length_width_height"], np.float32), options.voxel_size_m)
    surface_odd = _surface_coverage(local_odd_voxels, np.asarray(track["length_width_height"], np.float32), options.voxel_size_m)
    report = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "trusted_manifest": str(manifest_path),
        "track_id": int(options.track_id), "track_type": str(track.get("type")), "actor_family": str(track.get("actor_family")),
        "camera_ids": list(options.camera_ids), "reference_range": [options.reference_start, options.reference_end],
        "parameters": {"voxel_size_m": options.voxel_size_m, "cuboid_margin_m": options.cuboid_margin_m,
                       "static_ply_used": False, "sam2_only_used": False, "projection": "native_FTheta_midpoint_for_LiDAR_association"},
        "association_counts": {"selected_references": selected_frames, "raw_lidar_points_seen_with_repetition": raw_seen,
                               "source_mask_associations": mask_associations, "inside_cuboid_associations": inside_associations,
                               "unique_lidar_returns_inside_cuboid": len(association_ids),
                               "rejected_2d_mask_background_associations": mask_associations - inside_associations,
                               "records_with_source_mask": len(records)},
        "canonical_local_bounds_m": {"min": local.min(axis=0).astype(float).tolist(), "max": local.max(axis=0).astype(float).tolist()},
        "canonical_voxel_representative_counts": {"all": int(len(local_all_voxels)), "even": int(len(local_even_voxels)), "odd": int(len(local_odd_voxels))},
        "voxel_support": voxel_summary,
        "surface_probe_coverage": {"all_references": surface_all, "even_references": surface_even, "odd_references": surface_odd},
        "per_observation": records,
        "interpretation": [
            "This is a read-only raw-LiDAR observability audit, not a reconstruction or a quality result.",
            "An object-layer train/holdout split must use all cameras on even references for training and all cameras on odd references for holdout; the even_odd_overlap_voxels and surface coverage quantify its geometric support.",
            "Surface probes not supported by strict 3-D-gated raw LiDAR remain unknown. They must not be filled from static PLY, SAM2-only masks, or unconstrained RGB rays.",
        ],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote rigid object-layer observability audit: {options.output}", flush=True)
    return options.output
