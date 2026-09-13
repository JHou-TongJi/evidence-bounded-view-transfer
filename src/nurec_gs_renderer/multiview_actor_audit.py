"""Audit whether a canonical actor has usable multi-view observations.

This is intentionally a *diagnostic* between clean-static reconstruction and
another object optimisation.  It renders the frozen clean-static layer and one
canonical actor independently through the original NCore FTheta/rolling
shutter cameras, then records the evidence needed to distinguish three cases:

* the clean-static input mask did not cover the tracked object;
* static geometry is in front of, coplanar with, or behind the actor;
* the actor has (or lacks) image and sparse-LiDAR support in each source view.

An alpha overlap alone is not labelled as static leakage: a correct static
road/background is expected to overlap a foreground actor in image space.  The
report keeps the depth ordering separate so a later trainer can use only
observations with defensible visibility evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask, _read_actor_tracks, masked_mae, vehicle_roi
from .canonical_actor import CANONICAL_SCHEMA, CANONICAL_VERSION, CanonicalActorAsset
from .camera import compose_ncore_camera_pose
from .ply_io import GaussianScene, load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path, _ncore_exposure_timestamps
from .render import GsplatRenderer, RenderOptions
from .static_leakage import _project_lidar_depth_ftheta
from .street_dataset import SegformerVehicleSegmenter, _interpolate_track_pose, _nearest_index


DEFAULT_CAMERA_IDS = (
    "camera_cross_left_120fov",
    "camera_rear_left_70fov",
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
)


@dataclass(frozen=True)
class MultiviewActorAuditOptions:
    ncore_path: Path
    static_ply: Path
    canonical_actors: Path
    actor_dataset_manifest: Path
    clean_static_mask_dir: Path
    segformer_model_dir: Path
    output_dir: Path
    track_id: int
    chunk_index: int = 0
    model: str = "pa-front"
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    width: int = 480
    height: int = 270
    device: str = "cuda"
    cuboid_margin_pixels: int = 8
    probability_threshold: float = 0.60
    margin_threshold: float = 0.05
    depth_order_tolerance_m: float = 0.30
    max_lidar_timestamp_delta_us: int = 100_000
    use_lidar: bool = True
    write_previews: bool = False

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.chunk_index < 0 or not self.camera_ids:
            raise ValueError("track ID, chunk index and camera IDs are required")
        if self.width <= 0 or self.height <= 0 or self.cuboid_margin_pixels < 0:
            raise ValueError("image dimensions and cuboid margin are invalid")
        if not 0.0 <= self.probability_threshold <= 1.0 or not -1.0 <= self.margin_threshold <= 1.0:
            raise ValueError("semantic thresholds are invalid")
        if self.depth_order_tolerance_m <= 0.0 or self.max_lidar_timestamp_delta_us <= 0:
            raise ValueError("depth and LiDAR timestamp tolerances must be positive")


def isolate_canonical_track(asset: CanonicalActorAsset, track_id: int) -> CanonicalActorAsset:
    """Return a one-track canonical asset without changing its source file."""
    matches = [index for index, value in enumerate(asset.actor_ids) if int(value) == int(track_id)]
    if len(matches) != 1:
        raise ValueError(f"canonical asset has {len(matches)} actor IDs matching track {track_id}; expected one")
    actor_index = matches[0]
    members = np.asarray(asset.actor_indices == actor_index, dtype=bool)
    if not members.any():
        raise ValueError(f"canonical track {track_id} has no Gaussians")
    metadata = dict(asset.metadata)
    metadata.update({
        "schema": CANONICAL_SCHEMA,
        "version": CANONICAL_VERSION,
        "multiview_audit_isolated_track_id": int(track_id),
        "multiview_audit_isolated_gaussian_count": int(members.sum()),
    })
    isolated = CanonicalActorAsset(
        positions=asset.positions[members].copy(), rotations_wxyz=asset.rotations_wxyz[members].copy(),
        scales=asset.scales[members].copy(), rgb=asset.rgb[members].copy(),
        opacities=asset.opacities[members].copy(),
        actor_indices=np.zeros(int(members.sum()), dtype=np.int32),
        visibility_counts=asset.visibility_counts[members].copy(), source_counts=asset.source_counts[members].copy(),
        actor_ids=(int(track_id),), trajectories=(asset.trajectories[actor_index],), metadata=metadata,
    )
    isolated.validate()
    return isolated


def depth_order_masks(
    static_alpha: np.ndarray,
    static_depth: np.ndarray,
    actor_alpha: np.ndarray,
    actor_depth: np.ndarray,
    roi: np.ndarray,
    *,
    alpha_threshold: float = 0.02,
    tolerance_m: float = 0.30,
) -> dict[str, np.ndarray]:
    """Partition visible static/actor overlap by expected-depth ordering."""
    static_alpha = np.asarray(static_alpha, dtype=np.float32)
    actor_alpha = np.asarray(actor_alpha, dtype=np.float32)
    static_depth = np.asarray(static_depth, dtype=np.float32)
    actor_depth = np.asarray(actor_depth, dtype=np.float32)
    roi = np.asarray(roi, dtype=bool)
    if any(value.shape != roi.shape for value in (static_alpha, actor_alpha, static_depth, actor_depth)):
        raise ValueError("depth-order inputs must share one HxW shape")
    if alpha_threshold < 0.0 or tolerance_m <= 0.0:
        raise ValueError("alpha threshold and depth tolerance are invalid")
    both = roi & (static_alpha >= alpha_threshold) & (actor_alpha >= alpha_threshold)
    both &= np.isfinite(static_depth) & np.isfinite(actor_depth)
    difference = static_depth - actor_depth
    return {
        "both": both,
        "static_front": both & (difference < -tolerance_m),
        "coplanar": both & (np.abs(difference) <= tolerance_m),
        "static_behind": both & (difference > tolerance_m),
    }


def _fraction(mask: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator = np.asarray(denominator, dtype=bool)
    return None if not denominator.any() else float(np.asarray(mask, dtype=bool)[denominator].mean())


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def _load_clean_mask_entries(path: Path) -> tuple[dict[tuple[str, int], Path], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "nurec-clean-static-mask" or int(manifest.get("version", -1)) != 1:
        raise ValueError(f"unsupported clean-static mask manifest: {manifest_path}")
    entries: dict[tuple[str, int], Path] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        camera_id, frame_index, relative = entry.get("camera_id"), entry.get("frame_index"), entry.get("path")
        if not isinstance(camera_id, str) or not isinstance(frame_index, int) or not isinstance(relative, str):
            raise ValueError(f"malformed clean-static mask entry in {manifest_path}")
        target = path / relative
        if not target.is_file():
            raise FileNotFoundError(f"clean-static mask listed but missing: {target}")
        entries[(camera_id, frame_index)] = target
    if not entries:
        raise ValueError(f"clean-static mask manifest has no entries: {manifest_path}")
    return entries, manifest


def source_slots_from_clean_mask_manifest(
    manifest: dict[str, Any], sensors: dict[str, object], camera_ids: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[tuple[str, int], Path | None]]:
    """Use the original mask slots as temporal ground truth for all cameras.

    A profile such as ``pa-front`` has masks only for its one context camera.
    Other audit cameras must be synchronised to those exact reference times,
    not resampled according to a different camera subset.  They deliberately
    receive a ``None`` source-mask path, which is evidence that the upstream
    clean-static run never consumed pixels from that sensor.
    """
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("clean-static mask manifest has no entries")
    reference_by_slot: dict[int, int] = {}
    lookup: dict[tuple[str, int], tuple[int, Path]] = {}
    root = Path(manifest["_root"])
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("clean-static mask manifest has malformed entry")
        camera_id, slot, frame_index, reference, relative = (
            entry.get("camera_id"), entry.get("slot"), entry.get("frame_index"),
            entry.get("reference_timestamp_us"), entry.get("path"),
        )
        if not isinstance(camera_id, str) or not isinstance(slot, int) or not isinstance(frame_index, int):
            raise ValueError("clean-static mask manifest has malformed slot fields")
        if not isinstance(reference, int) or not isinstance(relative, str):
            raise ValueError("clean-static mask manifest has malformed timestamp/path")
        if slot in reference_by_slot and reference_by_slot[slot] != reference:
            raise ValueError("clean-static mask manifest has inconsistent slot reference timestamps")
        reference_by_slot[slot] = reference
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"clean-static mask listed but missing: {target}")
        lookup[(camera_id, slot)] = (frame_index, target)
    slots = tuple(sorted(reference_by_slot))
    if slots != tuple(range(len(slots))):
        raise ValueError("clean-static mask slots must be contiguous from zero")
    references = np.asarray([reference_by_slot[slot] for slot in slots], dtype=np.int64)
    selected: dict[str, np.ndarray] = {}
    masks: dict[tuple[str, int], Path | None] = {}
    for camera_id in camera_ids:
        if camera_id not in sensors:
            raise ValueError(f"unknown audit camera: {camera_id}")
        exposure_ends = _ncore_exposure_timestamps(sensors[camera_id])[:, 1].astype(np.int64)
        values: list[int] = []
        for slot, reference in enumerate(references):
            entry = lookup.get((camera_id, slot))
            frame_index = int(entry[0]) if entry is not None else _nearest_index(exposure_ends, int(reference))
            values.append(frame_index)
            masks[(camera_id, slot)] = None if entry is None else entry[1]
        selected[camera_id] = np.asarray(values, dtype=np.int64)
    return selected, references, masks


def _track_from_manifest(path: Path, track_id: int) -> dict[str, Any]:
    matches = [track for track in _read_actor_tracks(path) if int(track["track_id"]) == int(track_id)]
    if len(matches) != 1:
        raise ValueError(f"actor dataset manifest has {len(matches)} tracks matching {track_id}; expected one")
    return matches[0]


def match_frozen_track_geometry(target: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Match separately queried cuboid tracks by geometry, never by ID alone.

    ``consolidate_cuboid_tracks`` can renumber IDs when its time interval
    changes.  Clean-static masks and actor manifests are therefore compared in
    their shared world-time domain before their numeric IDs are interpreted.
    """
    target_times = np.asarray(target["timestamps_us"], dtype=np.int64)
    target_dimensions = np.asarray(target["length_width_height"], dtype=np.float64)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_times = np.asarray(candidate.get("timestamps_us", []), dtype=np.int64)
        if len(candidate_times) < 2:
            continue
        start, end = max(int(target_times.min()), int(candidate_times.min())), min(int(target_times.max()), int(candidate_times.max()))
        if end <= start:
            continue
        timestamps = np.linspace(start, end, num=5, dtype=np.int64)
        centres: list[float] = []
        for timestamp in timestamps:
            target_pose = _interpolate_track_pose(target, int(timestamp))
            candidate_pose = _interpolate_track_pose(candidate, int(timestamp))
            if target_pose is not None and candidate_pose is not None:
                centres.append(float(np.linalg.norm(target_pose[:3, 3] - candidate_pose[:3, 3])))
        if not centres:
            continue
        candidate_dimensions = np.asarray(candidate.get("length_width_height", []), dtype=np.float64)
        dimension_distance = float(np.linalg.norm(target_dimensions - candidate_dimensions)) if candidate_dimensions.shape == (3,) else float("inf")
        centre_distance = float(np.mean(centres))
        # Position is the primary identity feature; dimensions only break
        # otherwise-near parallel tracks such as adjacent parked vehicles.
        scored.append({
            "track_id": candidate.get("track_id"), "label": candidate.get("label"),
            "mean_center_distance_m": centre_distance, "dimension_distance_m": dimension_distance,
            "score": centre_distance + 0.25 * dimension_distance,
            "shared_timestamp_start_us": start, "shared_timestamp_end_us": end,
        })
    if not scored:
        return {"status": "no_overlapping_clean_mask_track", "geometry_match_confirmed": False}
    best = min(scored, key=lambda value: float(value["score"]))
    best["status"] = "matched_by_world_trajectory"
    best["geometry_match_confirmed"] = bool(
        best["mean_center_distance_m"] <= 0.50 and best["dimension_distance_m"] <= 0.50
    )
    best["candidate_count_with_temporal_overlap"] = len(scored)
    return best


def _write_preview(
    path: Path,
    source: np.ndarray,
    static_rgb: np.ndarray,
    actor_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    cuboid: np.ndarray,
    source_exclusion: np.ndarray,
) -> None:
    """Write a compact source/static/actor/composite evidence strip."""
    panels = [source, static_rgb, actor_rgb, composite_rgb]
    edge = np.asarray(cuboid, dtype=bool) & ~(
        np.roll(cuboid, 1, 0) & np.roll(cuboid, -1, 0) & np.roll(cuboid, 1, 1) & np.roll(cuboid, -1, 1)
    )
    canvas = Image.new("RGB", (source.shape[1] * len(panels), source.shape[0]))
    for index, panel in enumerate(panels):
        image = np.clip(np.rint(panel * 255.0), 0, 255).astype(np.uint8).copy()
        image[edge] = (255, 32, 32)
        image[source_exclusion] = (0.55 * image[source_exclusion] + 0.45 * np.asarray([0, 255, 255])).astype(np.uint8)
        canvas.paste(Image.fromarray(image, mode="RGB"), (index * source.shape[1], 0))
    canvas.save(path)


def audit_multiview_actor_observations(options: MultiviewActorAuditOptions) -> Path:
    """Write source-observation, occlusion and clean-mask evidence for one actor."""
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty multiview audit output: {output}")
    mask_entries, mask_manifest = _load_clean_mask_entries(options.clean_static_mask_dir.resolve())
    asset = isolate_canonical_track(CanonicalActorAsset.load(options.canonical_actors), options.track_id)
    track = _track_from_manifest(options.actor_dataset_manifest, options.track_id)
    clean_track_match = match_frozen_track_geometry(track, list(mask_manifest.get("tracks", [])))
    scene: GaussianScene = load_gaussian_ply(options.static_ply)
    loader = _create_ncore_loader(options.ncore_path)
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in options.camera_ids}
    # The loaded mapping is retained for strict manifest/file validation.  Slot
    # selection itself must come from the manifest's original references.
    _ = mask_entries
    mask_manifest["_root"] = str(options.clean_static_mask_dir.resolve())
    selected, references, source_mask_paths = source_slots_from_clean_mask_manifest(mask_manifest, sensors, options.camera_ids)
    chunk_count = int(mask_manifest.get("chunk_count", 0))
    if int(mask_manifest.get("chunk_index", -1)) != options.chunk_index:
        raise ValueError("clean-static mask chunk index does not match requested audit chunk")
    if str(mask_manifest.get("profile")) != options.model:
        raise ValueError("clean-static mask profile does not match requested audit model")

    from ncore.impl.sensors.camera import FThetaCameraModel

    models = {camera_id: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera_id, sensor in sensors.items()}
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    renderer = GsplatRenderer(scene, RenderOptions(device=options.device), canonical_actor_asset=asset)
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov") if options.use_lidar else None
    output.mkdir(parents=True)
    if options.write_previews:
        (output / "previews").mkdir()
    records: list[dict[str, Any]] = []
    print(
        f"Auditing track {options.track_id}: {len(options.camera_ids)} cameras x {len(references)} slots "
        f"at {options.width}x{options.height}",
        flush=True,
    )
    ordinal = 0
    for camera_id in options.camera_ids:
        sensor = sensors[camera_id]
        exposures = _ncore_exposure_timestamps(sensor)
        for slot, frame_index in enumerate(selected[camera_id]):
            ordinal += 1
            frame_index = int(frame_index)
            source_native = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
            source = np.asarray(
                Image.fromarray(source_native, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            source_float = source.astype(np.float32) / 255.0
            source_mask_path = source_mask_paths[(camera_id, slot)]
            clean_mask_available = source_mask_path is not None
            if clean_mask_available:
                source_mask = np.asarray(Image.open(source_mask_path).convert("L"), dtype=np.uint8) > 0
                if source_mask.shape != source_native.shape[:2]:
                    raise ValueError(f"clean-static mask shape mismatch for {camera_id} frame {frame_index}")
                source_mask = np.asarray(
                    Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").resize((options.width, options.height), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                ) > 0
            else:
                source_mask = np.zeros(source_native.shape[:2], dtype=bool)
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [frame_index], "output": {"width": options.width, "height": options.height}}, loader, camera_id
            ).frames[0]
            midpoint = int((int(exposures[frame_index, 0]) + int(exposures[frame_index, 1])) // 2)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            cuboid = _ftheta_cuboid_mask(
                [track], midpoint, camera_to_world, models[camera_id], height=options.height, width=options.width,
                source_height=int(source_native.shape[0]), source_width=int(source_native.shape[1]),
                margin_pixels=options.cuboid_margin_pixels, device=options.device,
            ).astype(bool)
            probability, margin = segmenter.vehicle_scores(Image.fromarray(source, mode="RGB"))
            semantic = vehicle_roi(
                cuboid, np.ones_like(cuboid, dtype=bool), probability, margin,
                probability_threshold=options.probability_threshold, margin_threshold=options.margin_threshold,
            )
            # A protruding/work-equipment label has no guaranteed ADE vehicle
            # class.  Its cuboid remains evidence, but the report makes the
            # fallback explicit rather than pretending semantic confirmation.
            roi = semantic if semantic.any() else cuboid
            roi_kind = "vehicle_semantic_intersection" if semantic.any() else "cuboid_fallback_no_vehicle_semantic"

            renderer.options = replace(renderer.options, static_opacity_scale=1.0, canonical_actor_opacity_scale=1.0)
            static = renderer.render(camera, include_depth=True)
            renderer.options = replace(renderer.options, static_opacity_scale=0.0, canonical_actor_opacity_scale=1.0)
            actor = renderer.render(camera, include_depth=True)
            composite = np.clip(actor.rgb + (1.0 - actor.alpha[..., None]) * static.rgb, 0.0, 1.0).astype(np.float32)
            ordering = depth_order_masks(static.alpha, static.depth, actor.alpha, actor.depth, roi, tolerance_m=options.depth_order_tolerance_m)
            static_actor_pixels = ordering["both"]
            static_better = static_actor_pixels & (
                np.abs(static.rgb - source_float).mean(axis=-1) < np.abs(actor.rgb - source_float).mean(axis=-1)
            )

            lidar_depth = None
            lidar_time = None
            if lidar_sensor is not None:
                lidar_index = int(lidar_sensor.get_closest_frame_index(midpoint))
                lidar_time = int(lidar_sensor.get_frame_timestamp_us(lidar_index))
                if abs(lidar_time - midpoint) <= options.max_lidar_timestamp_delta_us:
                    lidar_points = lidar_sensor.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                    lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
                    lidar_world = np.asarray(lidar_points, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
                    lidar_depth = _project_lidar_depth_ftheta(
                        lidar_world, np.linalg.inv(camera_to_world), models[camera_id],
                        source_width=int(source_native.shape[1]), source_height=int(source_native.shape[0]),
                        width=options.width, height=options.height, device=options.device,
                    )
            lidar_roi = roi & (lidar_depth > 0.0) if lidar_depth is not None else np.zeros_like(roi)
            static_lidar = lidar_roi & (static.alpha >= 0.02) & np.isfinite(static.depth)
            actor_lidar = lidar_roi & (actor.alpha >= 0.02) & np.isfinite(actor.depth)
            stem = f"{ordinal - 1:03d}_{camera_id}_{slot:02d}_{frame_index:06d}"
            if options.write_previews:
                _write_preview(output / "previews" / f"{stem}.png", source, static.rgb, actor.rgb, composite, cuboid, source_mask)
            records.append({
                "camera_id": camera_id, "slot": int(slot), "source_frame_index": frame_index,
                "reference_timestamp_us": int(references[slot]), "timestamp_midpoint_us": midpoint,
                "roi_kind": roi_kind, "cuboid_pixels": int(cuboid.sum()), "semantic_vehicle_pixels": int(semantic.sum()),
                "roi_pixels": int(roi.sum()), "clean_static_mask_available_for_camera_slot": clean_mask_available,
                "clean_static_exclusion_fraction_in_cuboid": _fraction(source_mask, cuboid) if clean_mask_available else None,
                "clean_static_exclusion_fraction_in_roi": _fraction(source_mask, roi) if clean_mask_available else None,
                "actor_alpha_fraction_in_roi": _fraction(actor.alpha >= 0.02, roi),
                "static_alpha_fraction_in_roi": _fraction(static.alpha >= 0.02, roi),
                "static_actor_overlap_fraction_in_roi": _fraction(static_actor_pixels, roi),
                "static_front_of_actor_fraction_in_roi": _fraction(ordering["static_front"], roi),
                "static_coplanar_with_actor_fraction_in_roi": _fraction(ordering["coplanar"], roi),
                "static_behind_actor_fraction_in_roi": _fraction(ordering["static_behind"], roi),
                "static_better_than_actor_fraction_in_overlap": _fraction(static_better, static_actor_pixels),
                "mae": {
                    "static": masked_mae(static.rgb, source_float, roi),
                    "actor": masked_mae(actor.rgb, source_float, roi),
                    "actor_over_static": masked_mae(composite, source_float, roi),
                },
                "lidar": {
                    "timestamp_us": lidar_time, "pixels_in_roi": int(lidar_roi.sum()),
                    "static_depth_mae_m": None if not static_lidar.any() else float(np.abs(static.depth[static_lidar] - lidar_depth[static_lidar]).mean()),
                    "actor_depth_mae_m": None if not actor_lidar.any() else float(np.abs(actor.depth[actor_lidar] - lidar_depth[actor_lidar]).mean()),
                },
                "preview": None if not options.write_previews else f"previews/{stem}.png",
            })
            print(
                f"[{ordinal:03d}/{len(options.camera_ids) * len(references)}] {camera_id} slot={slot:02d} frame={frame_index} "
                f"roi={int(roi.sum())} mask={_fraction(source_mask, roi) if clean_mask_available else 'not-input'} actor={_fraction(actor.alpha >= 0.02, roi)} "
                f"front={_fraction(ordering['static_front'], roi)}",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "purpose": "Pre-training multi-view actor visibility, clean-static-mask and depth-order audit.",
        "interpretation": {
            "alpha_overlap": "Not sufficient evidence of static leakage: static geometry behind a foreground actor is expected.",
            "static_front_or_coplanar": "Evidence requiring inspection before actor optimisation; it can indicate static leakage or a cuboid/actor depth error.",
            "cuboid_fallback": "The target had no high-confidence ADE vehicle silhouette in that observation; it is not a semantic confirmation.",
        },
        "inputs": {
            "ncore_path": str(options.ncore_path.resolve()), "static_ply": str(options.static_ply.resolve()),
            "canonical_actors": str(options.canonical_actors.resolve()), "actor_dataset_manifest": str(options.actor_dataset_manifest.resolve()),
            "clean_static_mask_dir": str(options.clean_static_mask_dir.resolve()), "clean_static_mask_profile": mask_manifest.get("profile"),
            "clean_static_mask_camera_ids": mask_manifest.get("camera_ids"),
        },
        "track": {
            "track_id": int(options.track_id), "label": track.get("label"), "canonical_gaussian_count": asset.count,
            "clean_static_mask_geometry_match": clean_track_match,
        },
        "selection": {"model": options.model, "chunk_index": options.chunk_index, "chunk_count": chunk_count,
                      "camera_ids": list(options.camera_ids), "slot_count": len(references), "reference_timestamps_us": [int(value) for value in references],
                      "non_context_cameras_use_nearest_frame_to_clean_mask_reference": True},
        "parameters": {"resolution": [options.width, options.height], "cuboid_margin_pixels": options.cuboid_margin_pixels,
                       "semantic_probability_threshold": options.probability_threshold, "semantic_margin_threshold": options.margin_threshold,
                       "depth_order_tolerance_m": options.depth_order_tolerance_m, "use_lidar": options.use_lidar},
        "mean": {
            "clean_static_exclusion_fraction_in_roi": _mean([record["clean_static_exclusion_fraction_in_roi"] for record in records]),
            "actor_alpha_fraction_in_roi": _mean([record["actor_alpha_fraction_in_roi"] for record in records]),
            "static_actor_overlap_fraction_in_roi": _mean([record["static_actor_overlap_fraction_in_roi"] for record in records]),
            "static_front_of_actor_fraction_in_roi": _mean([record["static_front_of_actor_fraction_in_roi"] for record in records]),
            "static_coplanar_with_actor_fraction_in_roi": _mean([record["static_coplanar_with_actor_fraction_in_roi"] for record in records]),
            "static_behind_actor_fraction_in_roi": _mean([record["static_behind_actor_fraction_in_roi"] for record in records]),
            "static_mae": _mean([record["mae"]["static"] for record in records]),
            "actor_mae": _mean([record["mae"]["actor"] for record in records]),
            "actor_over_static_mae": _mean([record["mae"]["actor_over_static"] for record in records]),
        },
        "observations": records,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote multiview actor audit: {report_path}", flush=True)
    return report_path
