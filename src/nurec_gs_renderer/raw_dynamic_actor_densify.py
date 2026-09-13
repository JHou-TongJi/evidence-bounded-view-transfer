"""Conservative RGB densification anchored to raw LiDAR and multi-view support."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import interpolate_w2c
from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, CanonicalActorAsset
from .raw_dynamic_actor_initialize import TRUSTED_SCHEMA, fuse_actor_local_lidar
from .raw_dynamic_actor_train import box_surface_points
from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-raw-rgb-lidar-actor-densification"
VERSION = 1


@dataclass(frozen=True)
class RawRgbLidarDensifyOptions:
    trusted_manifest: Path
    initial_actors: Path
    output: Path
    track_id: int
    device: str = "cuda"
    voxel_size_m: float = .08
    minimum_view_support: int = 2
    minimum_new_gaussians: int = 32
    pixel_stride: int = 8
    max_anchor_distance_pixels: float = 14.
    reference_start: int | None = None
    reference_end: int | None = None

    def __post_init__(self) -> None:
        if self.output.suffix != ".npz" or self.track_id < 0 or self.voxel_size_m <= 0 or self.minimum_view_support <= 0 or self.minimum_new_gaussians <= 0:
            raise ValueError("output/track/voxel/support parameters are invalid")
        if self.pixel_stride <= 0 or self.max_anchor_distance_pixels <= 0:
            raise ValueError("pixel sampling parameters are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: value = json.load(handle)
    if not isinstance(value, dict): raise ValueError(f"JSON object expected: {path}")
    return value


def anchor_mask_pixels(mask: np.ndarray, lidar_xy: np.ndarray, lidar_depth: np.ndarray, *, stride: int, max_distance_pixels: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid-sample a mask only when a nearest sparse LiDAR anchor is nearby."""
    binary = np.asarray(mask, bool); xy = np.asarray(lidar_xy, np.int64); depth = np.asarray(lidar_depth, np.float32)
    if binary.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2 or depth.shape != (len(xy),):
        raise ValueError("mask/LiDAR shapes are invalid")
    if not len(xy): return np.empty((0, 2), np.int64), np.empty(0, np.float32)
    yy, xx = np.nonzero(binary)
    selected = (xx % stride == 0) & (yy % stride == 0)
    pixels = np.stack((xx[selected], yy[selected]), axis=1).astype(np.int64)
    if not len(pixels): return pixels, np.empty(0, np.float32)
    distances = ((pixels[:, None, :].astype(np.float32) - xy[None].astype(np.float32)) ** 2).sum(axis=-1)
    nearest = distances.argmin(axis=1); keep = distances[np.arange(len(pixels)), nearest] <= max_distance_pixels ** 2
    return pixels[keep], depth[nearest[keep]]


def _midpoint_c2w(record: dict[str, Any]) -> np.ndarray:
    start = np.linalg.inv(np.asarray(record["camera_to_world_start"], np.float64)); end = np.linalg.inv(np.asarray(record["camera_to_world_end"], np.float64))
    rotations, translations = interpolate_w2c(start, end, np.asarray([.5], np.float64))
    midpoint = np.eye(4, dtype=np.float64); midpoint[:3, :3], midpoint[:3, 3] = rotations[0], translations[0]
    return np.linalg.inv(midpoint)


def densify_raw_rgb_lidar_actor(options: RawRgbLidarDensifyOptions) -> CanonicalActorAsset:
    """Back-project only LiDAR-anchored RGB mask pixels, then require multi-view fusion."""
    if options.output.exists(): raise FileExistsError(f"refusing to overwrite actor asset: {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = options.trusted_manifest.resolve(); manifest = _read(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete": raise ValueError("trusted manifest is invalid")
    tracks = {int(item["track_id"]): item for item in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid": raise ValueError("track must be a trusted rigid actor")
    initial = CanonicalActorAsset.load(options.initial_actors)
    if initial.actor_ids != (options.track_id,) or initial.metadata.get("trusted_manifest_sha256") != __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("initial actors must be a hash-bound single-track asset for this manifest")
    from ncore.impl.sensors.camera import FThetaCameraModel
    import torch
    loader = _create_ncore_loader(Path(manifest["ncore_path"])); roots = {key: Path(value) for key, value in manifest["roots"].items()}
    models: dict[str, Any] = {}; local_parts: list[np.ndarray] = []; color_parts: list[np.ndarray] = []; views: list[tuple[str, int]] = []
    candidate_count = 0; accepted_observations = 0
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if options.reference_start is not None and reference < options.reference_start: continue
        if options.reference_end is not None and reference >= options.reference_end: continue
        for record in frame["cameras"]:
            instance = next((item for item in record["instances"] if int(item["track_id"]) == options.track_id and item.get("mask_origin") == "source"), None)
            if instance is None or not isinstance(instance.get("lidar_depth"), str): continue
            camera_id = str(record["camera_id"]); sensor = loader.get_camera_sensor(camera_id)
            model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=options.device))
            source_index = int(record["source_frame_index"]); image = np.asarray(sensor.get_frame_image_array(source_index), np.uint8)
            mask = np.asarray(Image.open(roots[str(instance["mask_root"])] / str(instance["mask"])).convert("L"), np.uint8) > 0
            with np.load(roots[str(instance["lidar_depth_root"])] / str(instance["lidar_depth"]), allow_pickle=False) as data:
                lidar_xy, lidar_depth = np.asarray(data["xy"], np.int64), np.asarray(data["depth_m"], np.float32)
            pixels, depths = anchor_mask_pixels(mask, lidar_xy, lidar_depth, stride=options.pixel_stride, max_distance_pixels=options.max_anchor_distance_pixels)
            if not len(pixels): continue
            rays = model.pixels_to_camera_rays(torch.from_numpy(pixels).to(options.device)).detach().cpu().numpy().astype(np.float32)
            valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > .05) & np.isfinite(depths) & (depths > .1)
            pixels, depths, rays = pixels[valid], depths[valid], rays[valid]
            if not len(pixels): continue
            camera_to_world = _midpoint_c2w(record)
            camera_points = rays * (depths / rays[:, 2])[:, None]
            world = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
            pose = _interpolate_track_pose(track, int(record["timestamp_midpoint_us"]))
            if pose is None: continue
            local = (world - pose[:3, 3]) @ pose[:3, :3]
            half = np.asarray(track["length_width_height"], np.float32) * .5 + .10
            keep = np.all(np.abs(local) <= half[None], axis=1)
            local, pixels = local[keep].astype(np.float32), pixels[keep]
            if not len(local): continue
            local_parts.append(local); color_parts.append(image[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.)
            views.extend([(camera_id, reference)] * len(local)); candidate_count += len(local); accepted_observations += 1
    if not local_parts: raise RuntimeError("no LiDAR-anchored RGB pixels survived the 3D actor gate")
    positions, rgb, visibility, source_counts = fuse_actor_local_lidar(np.concatenate(local_parts), np.concatenate(color_parts), np.zeros(candidate_count, np.int32), views, voxel_size_m=options.voxel_size_m, minimum_view_support=options.minimum_view_support)
    # Explicit unknown: unsupported portions of a canonical cuboid surface do
    # not get a Gaussian.  They are represented in a sidecar rather than
    # hallucinated with RGB projection or rendered as opaque geometry.
    base_keys = {tuple(np.floor(point / options.voxel_size_m).astype(np.int64)) for point in initial.positions}
    new_keys = np.asarray([tuple(np.floor(point / options.voxel_size_m).astype(np.int64)) not in base_keys for point in positions], bool)
    positions, rgb, visibility, source_counts = positions[new_keys], rgb[new_keys], visibility[new_keys], source_counts[new_keys]
    if len(positions) < options.minimum_new_gaussians:
        raise RuntimeError(
            f"RGB-LiDAR densification is too sparse: {len(positions)} new Gaussian(s), "
            f"need at least {options.minimum_new_gaussians}; keep unsupported surface unknown"
        )
    combined_positions = np.concatenate((initial.positions, positions)); combined_rgb = np.concatenate((initial.rgb, rgb))
    combined_visibility = np.concatenate((initial.visibility_counts, visibility)); combined_counts = np.concatenate((initial.source_counts, source_counts))
    surface = box_surface_points(np.asarray(track["length_width_height"], np.float32), options.voxel_size_m)
    distances = ((surface[:, None] - combined_positions[None]) ** 2).sum(axis=-1)
    observed = distances.min(axis=1) <= (options.voxel_size_m * 1.5) ** 2
    visibility_path = options.output.with_suffix(".visibility.npz")
    np.savez_compressed(visibility_path, surface_positions=surface.astype(np.float32), observed=observed.astype(np.uint8), voxel_size_m=np.asarray(options.voxel_size_m, np.float32))
    asset = CanonicalActorAsset(
        positions=combined_positions.astype(np.float32), rotations_wxyz=np.tile(np.asarray([[1., 0., 0., 0.]], np.float32), (len(combined_positions), 1)),
        scales=np.concatenate((initial.scales, np.full((len(positions), 3), options.voxel_size_m * .50, np.float32))), rgb=combined_rgb.astype(np.float32),
        opacities=np.concatenate((initial.opacities, np.full(len(positions), .22, np.float32))), actor_indices=np.zeros(len(combined_positions), np.int32),
        visibility_counts=combined_visibility, source_counts=combined_counts, actor_ids=initial.actor_ids, trajectories=initial.trajectories,
        metadata={"schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION, "coordinate_frame": "actor_local", "initialization": "raw_lidar_plus_multiview_rgb_lidar_anchored_densification", "trusted_manifest": str(manifest_path), "trusted_manifest_sha256": initial.metadata["trusted_manifest_sha256"], "static_ply_used": False, "visibility_sidecar": str(visibility_path.resolve()), "unknown_surface_not_rendered": True},
    )
    asset.validate(); asset.save(options.output)
    report = {"schema": SCHEMA, "version": VERSION, "output": str(options.output.resolve()), "trusted_manifest": str(manifest_path), "initial_actors": str(options.initial_actors.resolve()), "candidate_pixels_after_3d_gate": candidate_count, "source_observations": accepted_observations, "new_multiview_gaussians": int(len(positions)), "total_gaussians": asset.count, "visibility_surface_observed_fraction": float(observed.mean()), "visibility_surface_unknown_fraction": float((~observed).mean()), "visibility_sidecar": str(visibility_path.resolve()), "limitations": ["RGB depths are locally inherited from sparse LiDAR anchors; no unconstrained mask pixel is made into geometry.", "Unsupported surface cells stay unknown and transparent; no static PLY or SAM2-only observation is used.", "This is initialization only and requires exact FTheta/rolling training plus holdout audit."]}
    options.output.with_suffix(".densify-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote RGB-LiDAR densified canonical actor: {options.output} (+{len(positions)} Gaussians)", flush=True)
    return asset
