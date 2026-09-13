"""Exact NCore FTheta/rolling-shutter validation of SAM2-added instance masks.

This is a pre-training gate.  It does not optimise a Gaussian layer: sparse
LiDAR points inside a SAM2-added source mask are selected in the tracked rigid
cuboid, then projected with NCore's own rolling-shutter FTheta API into an
independent source-mask view at the same reference time.  The target mask must
therefore not be another SAM2 proposal.  A frozen static PLY depth rendering
also vetoes points proven to be behind static foreground geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import compose_ncore_camera_pose, invert_pose
from .ply_io import load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .render import GsplatRenderer, RenderOptions
from .static_leakage import _project_world_points_ftheta
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-dynamic-mask-crossview-audit"
VERSION = 1


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match {shape}: {path}")
    return value


def select_points_in_actor_cuboid(points_world: np.ndarray, track: dict[str, Any], timestamp_us: int, *, margin_m: float) -> np.ndarray:
    """Select raw LiDAR points in one rigid actor's world-space cuboid."""
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    pose = _interpolate_track_pose(track, int(timestamp_us))
    if pose is None:
        return np.zeros(len(points_world), dtype=bool)
    points = np.asarray(points_world, dtype=np.float32)
    half = np.asarray(track["length_width_height"], dtype=np.float32) * 0.5 + float(margin_m)
    local = (points - pose[:3, 3].astype(np.float32)) @ pose[:3, :3].astype(np.float32)
    return np.all(np.abs(local) <= half[None], axis=1)


def project_world_points_native_rolling(
    model: Any,
    points_world: np.ndarray,
    camera_record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use NCore's iterative rolling-shutter projection, returning source indices/pixels/z."""
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N, 3]")
    if not len(points):
        return np.empty(0, np.int64), np.empty((0, 2), np.int64), np.empty(0, np.float32)
    start = np.asarray(camera_record["camera_to_world_start"], dtype=np.float64)
    end = np.asarray(camera_record["camera_to_world_end"], dtype=np.float64)
    projected = model.world_points_to_pixels_shutter_pose(
        points,
        invert_pose(start),
        invert_pose(end),
        int(camera_record["timestamp_start_us"]),
        int(camera_record["timestamp_end_us"]),
        return_T_world_sensors=True,
        return_valid_indices=True,
    )
    indices = projected.valid_indices.detach().cpu().numpy().astype(np.int64)
    pixels = projected.pixels.detach().cpu().numpy().astype(np.int64)
    w2c = projected.T_world_sensors.detach().cpu().numpy().astype(np.float32)
    depth = np.einsum("nij,nj->ni", w2c[:, :3, :3], points[indices])[:, 2] + w2c[:, 2, 3]
    width, height = (int(value) for value in camera_record["source_resolution"])
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (depth > 0.1)
    return indices[inside], pixels[inside], depth[inside].astype(np.float32)


def static_foreground_mask(
    static_alpha: np.ndarray,
    static_depth: np.ndarray,
    pixels: np.ndarray,
    point_depth: np.ndarray,
    *,
    depth_margin_m: float,
) -> np.ndarray:
    """Return points occluded by foreground geometry in a frozen static render."""
    if depth_margin_m < 0.0:
        raise ValueError("depth_margin_m must be non-negative")
    alpha = np.asarray(static_alpha, dtype=np.float32)
    depth = np.asarray(static_depth, dtype=np.float32)
    pixel = np.asarray(pixels, dtype=np.int64)
    points = np.asarray(point_depth, dtype=np.float32)
    sampled_alpha = alpha[pixel[:, 1], pixel[:, 0]]
    sampled_depth = depth[pixel[:, 1], pixel[:, 0]]
    return (sampled_alpha >= 0.02) & np.isfinite(sampled_depth) & (sampled_depth < points - depth_margin_m)


def resize_pixel_indices(pixels: np.ndarray, *, source_width: int, source_height: int, width: int, height: int) -> np.ndarray:
    """Apply the same half-pixel coordinate resize used by NCore camera paths."""
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    value = np.asarray(pixels, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    scaled = np.empty_like(value)
    scaled[:, 0] = np.rint((value[:, 0] + 0.5) * (width / source_width) - 0.5).astype(np.int64)
    scaled[:, 1] = np.rint((value[:, 1] + 0.5) * (height / source_height) - 0.5).astype(np.int64)
    return scaled


@dataclass(frozen=True)
class DynamicMaskCrossviewAuditOptions:
    refinement_manifest: Path
    static_ply: Path
    output: Path
    track_ids: tuple[int, ...]
    device: str = "cuda"
    max_source_observations: int = 0
    cuboid_margin_m: float = 0.25
    static_occluder_depth_margin_m: float = 0.30
    minimum_source_lidar_points: int = 8
    minimum_target_lidar_points: int = 4
    minimum_target_mask_fraction: float = 0.50
    static_width: int = 480
    static_height: int = 270

    def __post_init__(self) -> None:
        if not self.track_ids or len(set(self.track_ids)) != len(self.track_ids) or any(value < 0 for value in self.track_ids):
            raise ValueError("track_ids must be unique non-negative values")
        if self.max_source_observations < 0 or self.cuboid_margin_m < 0.0 or self.static_occluder_depth_margin_m < 0.0:
            raise ValueError("observation/margin values are invalid")
        if self.minimum_source_lidar_points <= 0 or self.minimum_target_lidar_points <= 0 or self.static_width <= 0 or self.static_height <= 0:
            raise ValueError("minimum LiDAR point counts must be positive")
        if not 0.0 <= self.minimum_target_mask_fraction <= 1.0:
            raise ValueError("minimum_target_mask_fraction must be in [0, 1]")


def _entry_key(camera_id: str, entry: dict[str, Any]) -> tuple[str, int, int]:
    return camera_id, int(entry["reference_frame_index"]), int(entry["track_id"])


def audit_dynamic_mask_crossview(options: DynamicMaskCrossviewAuditOptions) -> Path:
    refinement_path = options.refinement_manifest.resolve()
    refined = _read(refinement_path)
    if refined.get("schema") != "ncore-dynamic-reconstruction-mask-refinement" or refined.get("status") != "complete":
        raise ValueError("refinement manifest must be complete")
    source_path = Path(refined["source_manifest"]).resolve()
    source = _read(source_path)
    if source.get("schema") != "ncore-dynamic-reconstruction-dataset" or source.get("status") != "complete":
        raise ValueError("source manifest must be a complete dynamic reconstruction dataset")
    tracks = {int(value["track_id"]): value for value in source["tracks"]}
    if set(options.track_ids).difference(tracks):
        raise ValueError("a requested track is not in the source dataset")
    if any(tracks[track_id]["actor_family"] != "rigid" for track_id in options.track_ids):
        raise ValueError("cross-view LiDAR gate currently accepts rigid tracks only")
    source_root, refinement_root = source_path.parent, refinement_path.parent
    frames = {int(value["reference_frame_index"]): value for value in source["frames"]}
    records = {
        (int(frame["reference_frame_index"]), camera["camera_id"]): camera
        for frame in source["frames"] for camera in frame["cameras"]
    }
    final_entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    proposal_entries: list[tuple[str, dict[str, Any]]] = []
    for camera in refined["cameras"]:
        camera_id = str(camera["camera_id"])
        for entry in camera.get("observations", []):
            if not isinstance(entry.get("refined_mask"), str):
                continue
            key = _entry_key(camera_id, entry)
            final_entries[key] = entry
            if int(entry["track_id"]) in options.track_ids and entry.get("mask_origin") == "sam2":
                proposal_entries.append((camera_id, entry))
    def has_independent_target(camera_id: str, entry: dict[str, Any]) -> bool:
        reference, track_id = int(entry["reference_frame_index"]), int(entry["track_id"])
        return any(
            target_camera != camera_id and target_entry.get("mask_origin") == "source"
            for (target_camera, target_reference, target_track), target_entry in final_entries.items()
            if target_reference == reference and target_track == track_id
        )
    proposal_entries = [value for value in proposal_entries if has_independent_target(*value)]
    proposal_entries.sort(key=lambda value: (int(value[1]["track_id"]), int(value[1]["reference_frame_index"]), value[0]))
    if options.max_source_observations:
        proposal_entries = proposal_entries[: options.max_source_observations]
    if not proposal_entries:
        raise RuntimeError("no SAM2-added observation has an independent source-mask camera at the same reference time")

    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(Path(source["ncore_path"]))
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov")
    camera_ids = sorted({camera_id for camera_id, _ in proposal_entries} | {key[0] for key in final_entries})
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in camera_ids}
    models = {camera_id: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera_id, sensor in sensors.items()}
    renderer = GsplatRenderer(load_gaussian_ply(options.static_ply), RenderOptions(device=options.device))
    static_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    def static_depth(camera_id: str, reference_index: int) -> tuple[np.ndarray, np.ndarray]:
        key = camera_id, reference_index
        if key not in static_cache:
            camera_record = records[(reference_index, camera_id)]
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [int(camera_record["source_frame_index"])], "output": {"width": options.static_width, "height": options.static_height}},
                loader, camera_id,
            ).frames[0]
            frame = renderer.render(camera, include_depth=True)
            if frame.depth is None:
                raise RuntimeError("static render did not return depth")
            static_cache[key] = frame.alpha, frame.depth
        return static_cache[key]

    report_entries: list[dict[str, Any]] = []
    for ordinal, (source_camera, entry) in enumerate(proposal_entries, start=1):
        reference, track_id = int(entry["reference_frame_index"]), int(entry["track_id"])
        source_record = records[(reference, source_camera)]
        source_mask = _mask(refinement_root / entry["refined_mask"], (int(source_record["source_resolution"][1]), int(source_record["source_resolution"][0])))
        lidar_index = source_record.get("lidar_frame_index")
        if lidar_index is None:
            report_entries.append({"reference_frame_index": reference, "track_id": track_id, "source_camera": source_camera, "status": "no_lidar_frame"})
            continue
        cloud = lidar_sensor.get_frame_point_cloud(int(lidar_index), False, False).xyz_m_end
        lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", int(lidar_index)), dtype=np.float64)
        world = np.asarray(cloud, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        # The source dataset's LiDAR sidecar was selected in native image
        # space (semantic ∩ cuboid), not by a cuboid test at the asynchronous
        # LiDAR timestamp.  Preserve that proven source convention to avoid
        # discarding valid returns when an actor pose is sparse, then apply
        # the exact rolling-shutter source projection to the small candidate
        # set before any cross-camera test.
        actor_points = select_points_in_actor_cuboid(world, tracks[track_id], int(source_record["lidar_timestamp_us"]), margin_m=options.cuboid_margin_m)
        midpoint = int(source_record["timestamp_midpoint_us"])
        rig = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
        source_midpoint_pose = compose_ncore_camera_pose(rig, np.asarray(sensors[source_camera].T_sensor_rig))
        source_pixels_all, _, source_valid = _project_world_points_ftheta(
            world, invert_pose(source_midpoint_pose), models[source_camera],
            source_width=int(source_record["source_resolution"][0]), source_height=int(source_record["source_resolution"][1]),
            width=int(source_record["source_resolution"][0]), height=int(source_record["source_resolution"][1]), device=options.device,
        )
        source_seed = np.flatnonzero(source_valid)
        source_pixels = source_pixels_all[source_valid]
        source_seed = source_seed[source_mask[source_pixels[:, 1].astype(np.int64), source_pixels[:, 0].astype(np.int64)]]
        indices, pixels, _ = project_world_points_native_rolling(models[source_camera], world[source_seed], source_record)
        source_inside = source_mask[pixels[:, 1], pixels[:, 0]] if len(pixels) else np.zeros(0, bool)
        selected_world = world[source_seed][indices[source_inside]]
        item: dict[str, Any] = {
            "reference_frame_index": reference, "track_id": track_id, "source_camera": source_camera,
            "source_mask_origin": entry.get("mask_origin"), "actor_cuboid_lidar_points": int(actor_points.sum()),
            "source_image_lidar_seed_points": int(len(source_seed)),
            "source_mask_lidar_points": int(len(selected_world)), "target_views": [],
        }
        if len(selected_world) < options.minimum_source_lidar_points:
            item["status"] = "insufficient_source_lidar"
            report_entries.append(item)
            print(f"[{ordinal}/{len(proposal_entries)}] track={track_id} ref={reference} source={source_camera}: source LiDAR={len(selected_world)} skip", flush=True)
            continue
        independent = [
            (target_camera, target_entry)
            for (target_camera, target_reference, target_track), target_entry in final_entries.items()
            if target_reference == reference and target_track == track_id and target_camera != source_camera
            and target_entry.get("mask_origin") == "source"
        ]
        for target_camera, target_entry in sorted(independent):
            target_record = records[(reference, target_camera)]
            target_mask = _mask(refinement_root / target_entry["refined_mask"], (int(target_record["source_resolution"][1]), int(target_record["source_resolution"][0])))
            _, target_pixels, target_depth = project_world_points_native_rolling(models[target_camera], selected_world, target_record)
            view: dict[str, Any] = {"camera_id": target_camera, "projected_lidar_points": int(len(target_pixels))}
            if len(target_pixels):
                alpha, depth = static_depth(target_camera, reference)
                static_pixels = resize_pixel_indices(
                    target_pixels, source_width=int(target_record["source_resolution"][0]), source_height=int(target_record["source_resolution"][1]),
                    width=options.static_width, height=options.static_height,
                )
                occluded = static_foreground_mask(alpha, depth, static_pixels, target_depth, depth_margin_m=options.static_occluder_depth_margin_m)
                eligible = ~occluded
                inside = target_mask[target_pixels[:, 1], target_pixels[:, 0]]
                view.update({
                    "static_foreground_occluded": int(occluded.sum()), "eligible_lidar_points": int(eligible.sum()),
                    "mask_supported_lidar_points": int((inside & eligible).sum()),
                    "mask_support_fraction": None if not eligible.any() else float((inside & eligible).sum() / eligible.sum()),
                })
                view["accepted"] = bool(
                    int(eligible.sum()) >= options.minimum_target_lidar_points
                    and float(view["mask_support_fraction"] or 0.0) >= options.minimum_target_mask_fraction
                )
            else:
                view.update({"static_foreground_occluded": 0, "eligible_lidar_points": 0, "mask_supported_lidar_points": 0, "mask_support_fraction": None, "accepted": False})
            item["target_views"].append(view)
        observable = [view for view in item["target_views"] if view["eligible_lidar_points"] >= options.minimum_target_lidar_points]
        if not observable:
            item["status"] = "no_crossview_visibility"
        elif any(view["accepted"] for view in observable):
            item["status"] = "accepted"
        else:
            item["status"] = "crossview_rejected"
        report_entries.append(item)
        support = [view["mask_support_fraction"] for view in item["target_views"] if view["mask_support_fraction"] is not None]
        print(f"[{ordinal}/{len(proposal_entries)}] track={track_id} ref={reference} source={source_camera}: source LiDAR={len(selected_world)} target_support={support} status={item['status']}", flush=True)

    tracks_report = []
    for track_id in options.track_ids:
        values = [value for value in report_entries if value["track_id"] == track_id]
        usable = [value for value in values if value.get("status") != "no_lidar_frame"]
        observable = [value for value in usable if value.get("status") in {"accepted", "crossview_rejected"}]
        accepted = [value for value in observable if value.get("status") == "accepted"]
        support = [
            view["mask_support_fraction"] for value in usable for view in value.get("target_views", [])
            if view.get("mask_support_fraction") is not None
        ]
        tracks_report.append({
            "track_id": track_id, "label": tracks[track_id]["label"], "sam2_observations_checked": len(values),
            "crossview_observable_observations": len(observable), "crossview_accepted_observations": len(accepted),
            "crossview_acceptance_fraction": None if not observable else float(len(accepted) / len(observable)),
            "no_crossview_visibility_observations": int(sum(value.get("status") == "no_crossview_visibility" for value in usable)),
            "target_mask_support_fraction_median": None if not support else float(np.median(support)),
            "target_mask_support_fraction_p10": None if not support else float(np.percentile(support, 10)),
        })
    output = options.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parameters = {}
    for key, value in vars(options).items():
        if isinstance(value, Path):
            parameters[key] = str(value.resolve())
        elif isinstance(value, tuple):
            parameters[key] = list(value)
        else:
            parameters[key] = value
    report = {
        "schema": SCHEMA, "version": VERSION, "refinement_manifest": str(refinement_path), "source_manifest": str(source_path),
        "static_ply": str(options.static_ply.resolve()), "parameters": parameters,
        "track_reports": tracks_report, "observations": report_entries,
        "limitations": [
            "This is a sparse LiDAR/cross-camera supervision gate, not RGB reconstruction quality or a dynamic Gaussian asset.",
            "Only target masks originating from the independent source semantic/cuboid observation count as holdout support.",
            "Static depth is rendered through the production exact FTheta/rolling-shutter path at the configured audit resolution; LiDAR itself remains temporally sparse.",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote exact-FTheta cross-view dynamic mask audit: {output}", flush=True)
    return output
