"""Fuse raw NCore LiDAR returns into a conservative canonical rigid actor."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import interpolate_w2c
from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, ActorTrajectory, CanonicalActorAsset
from .color_correction import file_sha256
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .static_leakage import _project_world_points_ftheta
from .street_dataset import _interpolate_track_pose


TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"
SCHEMA = "ncore-raw-lidar-multiview-actor-initialization"
VERSION = 1


@dataclass(frozen=True)
class RawLidarActorInitializeOptions:
    trusted_manifest: Path
    output: Path
    track_ids: tuple[int, ...]
    device: str = "cuda"
    voxel_size_m: float = .10
    minimum_view_support: int = 2
    reference_start: int | None = None
    reference_end: int | None = None
    camera_ids: tuple[str, ...] | None = None
    reference_parity: int | None = None

    def __post_init__(self) -> None:
        if self.output.suffix != ".npz" or not self.track_ids or len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("output must be .npz and track IDs must be non-empty/unique")
        if any(value < 0 for value in self.track_ids) or self.voxel_size_m <= 0 or self.minimum_view_support <= 0:
            raise ValueError("track IDs, voxel size and view support are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")
        if self.camera_ids is not None and (not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique when specified")
        if self.reference_parity not in {None, 0, 1}:
            raise ValueError("reference_parity must be 0, 1 or omitted")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def fuse_actor_local_lidar(
    local_positions: np.ndarray,
    rgb: np.ndarray,
    actor_indices: np.ndarray,
    observations: list[tuple[str, int]],
    *, voxel_size_m: float,
    minimum_view_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median-fuse local points; require independent camera×reference support."""
    local = np.asarray(local_positions, np.float32)
    color = np.asarray(rgb, np.float32)
    actors = np.asarray(actor_indices, np.int32)
    if local.ndim != 2 or local.shape[1] != 3 or color.shape != local.shape or actors.shape != (len(local),) or len(observations) != len(local):
        raise ValueError("raw LiDAR arrays/observations are inconsistent")
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index, point in enumerate(local):
        voxel = tuple(np.floor(point / voxel_size_m).astype(np.int64))
        groups.setdefault((int(actors[index]), *voxel), []).append(index)
    positions: list[np.ndarray] = []; colors: list[np.ndarray] = []; support: list[int] = []; counts: list[int] = []
    for _, members in sorted(groups.items()):
        views = set(observations[index] for index in members)
        if len(views) < minimum_view_support:
            continue
        member = np.asarray(members, np.int64)
        positions.append(np.median(local[member], axis=0))
        colors.append(np.median(color[member], axis=0))
        support.append(len(views)); counts.append(len(member))
    if not positions:
        raise RuntimeError("no raw LiDAR voxel has the requested independent multi-view support")
    return (np.asarray(positions, np.float32), np.clip(np.asarray(colors, np.float32), 0., 1.),
            np.asarray(support, np.uint16), np.asarray(counts, np.uint16))


def initialize_raw_lidar_actor(options: RawLidarActorInitializeOptions) -> CanonicalActorAsset:
    """Use raw LiDAR points that project into original source masks only."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite actor asset: {options.output}")
    manifest_path = options.trusted_manifest.resolve()
    manifest = _read(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted_manifest must be complete trusted raw supervision")
    tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    if set(options.track_ids).difference(tracks) or any(tracks[value].get("actor_family") != "rigid" for value in options.track_ids):
        raise ValueError("requested tracks must be rigid tracks in trusted manifest")
    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    lidar = loader.get_lidar_sensor("lidar_top_360fov")
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    actor_index = {track_id: index for index, track_id in enumerate(options.track_ids)}
    models: dict[str, Any] = {}
    clouds: dict[int, np.ndarray] = {}
    local_points: list[np.ndarray] = []; colors: list[np.ndarray] = []; indices: list[int] = []; views: list[tuple[str, int]] = []
    raw_points = 0; mask_points = 0; used_observations = 0
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if options.reference_start is not None and reference < options.reference_start:
            continue
        if options.reference_end is not None and reference >= options.reference_end:
            continue
        if options.reference_parity is not None and reference % 2 != options.reference_parity:
            continue
        for record in frame["cameras"]:
            camera_id = str(record["camera_id"])
            if options.camera_ids is not None and camera_id not in options.camera_ids:
                continue
            source_index = int(record["source_frame_index"])
            sensor = loader.get_camera_sensor(camera_id)
            model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            source = np.asarray(sensor.get_frame_image_array(source_index), np.uint8)
            start = np.linalg.inv(np.asarray(record["camera_to_world_start"], np.float64))
            end = np.linalg.inv(np.asarray(record["camera_to_world_end"], np.float64))
            midpoint_w2c = np.eye(4, dtype=np.float64)
            rotation, translation = interpolate_w2c(start, end, np.asarray([.5], np.float64))
            midpoint_w2c[:3, :3], midpoint_w2c[:3, 3] = rotation[0], translation[0]
            for instance in record["instances"]:
                track_id = int(instance["track_id"])
                if track_id not in actor_index or instance.get("mask_origin") != "source":
                    continue
                lidar_index = record.get("lidar_frame_index")
                lidar_time = record.get("lidar_timestamp_us")
                if lidar_index is None or lidar_time is None:
                    continue
                lidar_index = int(lidar_index)
                if lidar_index not in clouds:
                    cloud = lidar.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                    T_lidar_world = np.asarray(lidar.get_frames_T_sensor_target("world", lidar_index), np.float64)
                    clouds[lidar_index] = np.asarray(cloud, np.float32) @ T_lidar_world[:3, :3].T + T_lidar_world[:3, 3]
                world = clouds[lidar_index]
                raw_points += len(world)
                mask = np.asarray(Image.open(roots[str(instance["mask_root"])] / str(instance["mask"])).convert("L"), np.uint8) > 0
                if mask.shape != source.shape[:2]:
                    raise ValueError(f"source mask/image shape mismatch: {camera_id} ref={reference}")
                pixels, _, valid = _project_world_points_ftheta(
                    world, midpoint_w2c, model, source_width=source.shape[1], source_height=source.shape[0],
                    width=source.shape[1], height=source.shape[0], device=options.device,
                )
                xx = np.where(valid, np.rint(pixels[:, 0]), -1).astype(np.int64)
                yy = np.where(valid, np.rint(pixels[:, 1]), -1).astype(np.int64)
                valid &= (xx >= 0) & (xx < source.shape[1]) & (yy >= 0) & (yy < source.shape[0])
                selected = np.flatnonzero(valid)
                selected = selected[mask[yy[selected], xx[selected]]]
                if not len(selected):
                    continue
                pose = _interpolate_track_pose(tracks[track_id], int(lidar_time))
                if pose is None:
                    continue
                # NCore row-vector convention: world = local @ R.T + t.
                local = (world[selected] - pose[:3, 3].astype(np.float32)) @ pose[:3, :3].astype(np.float32)
                # A 2D semantic∩cuboid mask can still contain a background
                # LiDAR return along the same ray.  Require the return to be
                # physically inside the tracked actor at the LiDAR timestamp;
                # otherwise the fused actor would grow distant road/building
                # sheets that later look like smearing.
                half = np.asarray(tracks[track_id]["length_width_height"], np.float32) * .5 + .15
                local_inside = np.all(np.abs(local) <= half[None], axis=1)
                selected, local = selected[local_inside], local[local_inside]
                if not len(selected):
                    continue
                local_points.append(local); colors.append(source[yy[selected], xx[selected]].astype(np.float32) / 255.)
                indices.extend([actor_index[track_id]] * len(selected)); views.extend([(camera_id, reference)] * len(selected))
                mask_points += len(selected); used_observations += 1
    if not local_points:
        raise RuntimeError("no raw LiDAR return projected into selected source masks")
    positions, rgb, support, source_counts = fuse_actor_local_lidar(
        np.concatenate(local_points), np.concatenate(colors), np.asarray(indices, np.int32), views,
        voxel_size_m=options.voxel_size_m, minimum_view_support=options.minimum_view_support,
    )
    # Recover each fused point's actor ID from its voxel key by nearest raw local point; all points are within a voxel.
    raw_local = np.concatenate(local_points); raw_indices = np.asarray(indices, np.int32)
    assigned = np.empty(len(positions), np.int32)
    for index, point in enumerate(positions):
        distances = np.linalg.norm(raw_local - point[None], axis=1)
        assigned[index] = raw_indices[int(np.argmin(distances))]
    trajectories = tuple(ActorTrajectory(np.asarray(tracks[track_id]["timestamps_us"], np.int64), np.asarray(tracks[track_id]["actor_to_world"], np.float32)) for track_id in options.track_ids)
    asset = CanonicalActorAsset(
        positions=positions, rotations_wxyz=np.tile(np.asarray([[1., 0., 0., 0.]], np.float32), (len(positions), 1)),
        scales=np.full((len(positions), 3), options.voxel_size_m * .60, np.float32), rgb=rgb,
        opacities=np.full(len(positions), .45, np.float32), actor_indices=assigned, visibility_counts=support,
        source_counts=source_counts, actor_ids=options.track_ids, trajectories=trajectories,
        metadata={"schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION, "coordinate_frame": "actor_local",
                  "initialization": "raw_lidar_projected_into_source_masks_multiview_voxel_fusion",
                  "trusted_manifest": str(manifest_path), "trusted_manifest_sha256": file_sha256(manifest_path),
                  "voxel_size_m": options.voxel_size_m, "minimum_view_support": options.minimum_view_support,
                  "camera_ids": None if options.camera_ids is None else list(options.camera_ids),
                  "reference_parity": options.reference_parity,
                  "static_ply_used": False},
    )
    asset.validate(); asset.save(options.output)
    report = {"schema": SCHEMA, "version": VERSION, "trusted_manifest": str(manifest_path), "output": str(options.output.resolve()),
              "track_ids": list(options.track_ids), "reference_range": [options.reference_start, options.reference_end],
              "camera_ids": None if options.camera_ids is None else list(options.camera_ids), "reference_parity": options.reference_parity,
              "raw_lidar_points_seen": raw_points, "source_mask_lidar_points": mask_points, "source_mask_observations": used_observations,
              "fused_gaussians": asset.count, "visibility_support": {"min": int(support.min()), "median": float(np.median(support)), "max": int(support.max())},
              "limitations": ["Only original source masks contribute; SAM2-only masks are never initialisation input.", "FTheta projection is exposure-midpoint for LiDAR association; final training/render remains rolling-shutter.", "No static PLY or static depth participates."],}
    report_path = options.output.with_suffix(".init-report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote raw LiDAR multi-view canonical actor: {options.output} ({asset.count} Gaussians)", flush=True)
    return asset
