"""Exact-FTheta attribution of native source-slot dynamics versus a canonical actor.

This module is deliberately an audit, not another optimiser.  A canonical
actor can fail for two materially different reasons: voxel fusion may have
removed necessary visible surfaces, or the source-slot dynamic Gaussians may
already be incomplete/contradictory.  Rendering both representations through
the same original camera makes that distinction measurable.  The audit also
compares the *surrounding*, source-mask-excluded static background of v3 and
v4 clean-static reconstructions so an actor is not blamed for a changed
background reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .actor_audit import _ftheta_cuboid_mask, masked_mae, vehicle_roi
from .camera import compose_ncore_camera_pose
from .canonical_actor import CanonicalActorAsset
from .dynamic import DynamicGaussianScene, load_dynamic_gaussians
from .multiview_actor_audit import (
    _load_clean_mask_entries,
    _track_from_manifest,
    isolate_canonical_track,
    source_slots_from_clean_mask_manifest,
)
from .ply_io import load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .render import GsplatRenderer, RenderOptions
from .street_dataset import SegformerVehicleSegmenter


DEFAULT_CAMERA_IDS = ("camera_front_wide_120fov", "camera_cross_right_120fov")


@dataclass(frozen=True)
class DynamicCanonicalAuditOptions:
    ncore_path: Path
    dynamic_gaussians: Path
    canonical_actors: Path
    actor_dataset_manifest: Path
    static_v3_ply: Path
    static_v4_ply: Path
    clean_static_mask_dir: Path
    segformer_model_dir: Path
    output_dir: Path
    track_id: int
    target_slots: tuple[int, ...] = (4, 5, 6)
    camera_ids: tuple[str, ...] = DEFAULT_CAMERA_IDS
    width: int = 480
    height: int = 270
    device: str = "cuda"
    cuboid_margin_pixels: int = 8
    probability_threshold: float = 0.60
    margin_threshold: float = 0.05
    surrounding_ring_outer_pixels: int = 20
    surrounding_ring_inner_pixels: int = 4

    def __post_init__(self) -> None:
        if self.track_id < 0 or not self.camera_ids or not self.target_slots:
            raise ValueError("track ID, camera IDs and target slots are required")
        if len(set(self.camera_ids)) != len(self.camera_ids) or len(set(self.target_slots)) != len(self.target_slots):
            raise ValueError("camera IDs and target slots must be unique")
        if min(self.target_slots) < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("target slots and image dimensions are invalid")
        if self.surrounding_ring_outer_pixels <= self.surrounding_ring_inner_pixels:
            raise ValueError("surrounding ring outer radius must exceed inner radius")


def subset_dynamic_track(
    scene: DynamicGaussianScene, track_id: int, source_slot: int | None = None,
    source_camera_index: int | None = None,
) -> DynamicGaussianScene:
    """Return an independently renderable track/source-slot dynamic layer.

    The renderer validates v2 source slots as contiguous.  A filtered source
    layer does not retain the original complete slot set, so labels are
    remapped locally; interpolation still uses each Gaussian's original three
    timestamps and is therefore unchanged.
    """
    if scene.source_track_ids is None or scene.source_frame_slots is None:
        raise ValueError("source-slot audit requires a v2 dynamic asset")
    selected = scene.source_track_ids == int(track_id)
    if source_slot is not None:
        selected &= scene.source_frame_slots == int(source_slot)
    if source_camera_index is not None:
        if scene.source_camera_indices is None:
            raise ValueError("source-camera selection requires v2 camera provenance")
        selected &= scene.source_camera_indices == int(source_camera_index)
    if not selected.any():
        raise ValueError(f"track {track_id} has no dynamic Gaussians for source slot {source_slot}")
    indices = np.flatnonzero(selected)
    metadata = dict(scene.metadata)
    metadata.update({"audit_track_id": int(track_id), "audit_source_slot": source_slot,
                     "audit_source_camera_index": source_camera_index})
    original_slots = scene.source_frame_slots[indices]
    unique_slots = np.unique(original_slots)
    remap = np.searchsorted(unique_slots, original_slots).astype(np.int16)
    return DynamicGaussianScene(
        keyframe_positions=scene.keyframe_positions[indices], keyframe_timestamps_us=scene.keyframe_timestamps_us[indices],
        rotations_wxyz=scene.rotations_wxyz[indices], scales=scene.scales[indices], rgb=scene.rgb[indices],
        max_opacities=scene.max_opacities[indices], metadata=metadata,
        source_track_ids=scene.source_track_ids[indices],
        source_camera_indices=None if scene.source_camera_indices is None else scene.source_camera_indices[indices],
        source_frame_slots=remap,
        association_sources=None if scene.association_sources is None else scene.association_sources[indices],
        actor_local_positions=None if scene.actor_local_positions is None else scene.actor_local_positions[indices],
    )


def nearest_source_slot_for_camera(
    scene: DynamicGaussianScene, track_id: int, source_camera_index: int, timestamp_us: int
) -> int:
    """Choose one observed source slot for one provenance camera by time.

    This is intentionally a deterministic *single-observation* selector. It
    is not the existing blend/nearest mode: only Gaussian points emitted by
    the requested source camera and selected slot remain active afterwards.
    """
    if scene.source_track_ids is None or scene.source_frame_slots is None or scene.source_camera_indices is None:
        raise ValueError("source-camera/slot selection requires native v2 provenance")
    members = (scene.source_track_ids == int(track_id)) & (scene.source_camera_indices == int(source_camera_index))
    if not members.any():
        raise ValueError(f"track {track_id} has no source-camera index {source_camera_index}")
    slots = np.unique(scene.source_frame_slots[members])
    centers = np.asarray([
        np.median(scene.keyframe_timestamps_us[members & (scene.source_frame_slots == slot), 1])
        for slot in slots
    ], dtype=np.int64)
    return int(slots[np.argmin(np.abs(centers.astype(np.int64) - int(timestamp_us)))])


def surrounding_ring(mask: np.ndarray, *, outer_pixels: int, inner_pixels: int) -> np.ndarray:
    """Return a finite-width ring outside a projected actor/cuboid mask."""
    if outer_pixels <= inner_pixels or inner_pixels < 0:
        raise ValueError("invalid surrounding ring radii")
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    outer = np.asarray(image.filter(ImageFilter.MaxFilter(2 * outer_pixels + 1)), dtype=np.uint8) > 0
    inner = np.asarray(image.filter(ImageFilter.MaxFilter(2 * inner_pixels + 1)), dtype=np.uint8) > 0
    return outer & ~inner


def alpha_iou(first: np.ndarray, second: np.ndarray, roi: np.ndarray, *, threshold: float = 0.02) -> float | None:
    first_mask = np.asarray(first >= threshold, dtype=bool) & np.asarray(roi, dtype=bool)
    second_mask = np.asarray(second >= threshold, dtype=bool) & np.asarray(roi, dtype=bool)
    union = first_mask | second_mask
    return None if not union.any() else float((first_mask & second_mask).sum() / union.sum())


def _fraction(mask: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator = np.asarray(denominator, dtype=bool)
    return None if not denominator.any() else float(np.asarray(mask, dtype=bool)[denominator].mean())


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not valid else float(np.mean(valid))


def _layer_record(rendered: Any, source: np.ndarray, roi: np.ndarray) -> dict[str, float | None]:
    visible = (rendered.alpha >= 0.02) & roi
    return {
        "alpha_fraction_in_roi": _fraction(visible, roi),
        "actor_only_mae": masked_mae(rendered.rgb, source, roi),
        "depth_valid_fraction_in_roi": _fraction(np.isfinite(rendered.depth), roi),
    }


def _rgb_depth_difference(first: Any, second: Any, mask: np.ndarray) -> dict[str, float | None]:
    shared_depth = np.asarray(mask, dtype=bool) & np.isfinite(first.depth) & np.isfinite(second.depth)
    return {
        "rgb_mae": masked_mae(first.rgb, second.rgb, mask),
        "alpha_mae": None if not np.asarray(mask, dtype=bool).any() else float(np.abs(first.alpha[mask] - second.alpha[mask]).mean()),
        "depth_mae_m": None if not shared_depth.any() else float(np.abs(first.depth[shared_depth] - second.depth[shared_depth]).mean()),
    }


def audit_dynamic_canonical(options: DynamicCanonicalAuditOptions) -> Path:
    """Run a short exact-camera source-slot/canonical/static attribution audit."""
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dynamic/canonical audit output: {output}")
    dynamic = load_dynamic_gaussians(options.dynamic_gaussians)
    if dynamic.source_track_ids is None or dynamic.source_frame_slots is None:
        raise ValueError("dynamic/canonical audit needs native v2 source provenance")
    track_slots = tuple(int(value) for value in np.unique(dynamic.source_frame_slots[dynamic.source_track_ids == options.track_id]))
    if not track_slots:
        raise ValueError(f"dynamic asset has no track {options.track_id}")
    all_track_dynamic = subset_dynamic_track(dynamic, options.track_id)
    if dynamic.source_camera_indices is None:
        raise ValueError("dynamic/canonical audit needs v2 source camera provenance")
    track_camera_indices = tuple(int(value) for value in np.unique(
        dynamic.source_camera_indices[dynamic.source_track_ids == options.track_id]
    ))
    # ``subset_dynamic_track`` makes retained slots contiguous for renderer
    # validation. Keep this explicit mapping so the report always labels the
    # original exported source-slot number.
    local_slot_by_original = {slot: index for index, slot in enumerate(track_slots)}
    per_slot_dynamic = {slot: subset_dynamic_track(dynamic, options.track_id, slot) for slot in track_slots}
    canonical = isolate_canonical_track(CanonicalActorAsset.load(options.canonical_actors), options.track_id)
    _, mask_manifest = _load_clean_mask_entries(options.clean_static_mask_dir.resolve())
    mask_manifest["_root"] = str(options.clean_static_mask_dir.resolve())
    loader = _create_ncore_loader(options.ncore_path)
    sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in options.camera_ids}
    selected, references, mask_paths = source_slots_from_clean_mask_manifest(mask_manifest, sensors, options.camera_ids)
    if max(options.target_slots) >= len(references):
        raise ValueError("requested target slot is outside clean-static manifest")
    if str(mask_manifest.get("profile")) != "pa-multiview":
        raise ValueError("this attribution is defined for the pa-multiview v4 clean-static manifest")
    track = _track_from_manifest(options.actor_dataset_manifest, options.track_id)
    from ncore.impl.sensors.camera import FThetaCameraModel

    models = {camera_id: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera_id, sensor in sensors.items()}
    scene_v3 = load_gaussian_ply(options.static_v3_ply)
    scene_v4 = load_gaussian_ply(options.static_v4_ply)
    static_v3 = GsplatRenderer(scene_v3, RenderOptions(device=options.device))
    static_v4 = GsplatRenderer(scene_v4, RenderOptions(device=options.device))
    # gsplat's FTheta CUDA path is not robust to a renderer whose static
    # tensors are literally empty (it can abort in native code before Python
    # reports an error).  Keep the proven v4 tensors and make them transparent
    # through the reversible renderer option instead.
    actor_only_options = RenderOptions(device=options.device, static_opacity_scale=0.0)
    canonical_renderer = GsplatRenderer(scene_v4, actor_only_options, canonical_actor_asset=canonical)
    all_dynamic_renderer = GsplatRenderer(scene_v4, actor_only_options, dynamic_scene=all_track_dynamic)
    slot_renderers = {
        slot: GsplatRenderer(scene_v4, actor_only_options, dynamic_scene=scene)
        for slot, scene in per_slot_dynamic.items()
    }
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    count = len(options.camera_ids) * len(options.target_slots)
    ordinal = 0
    print(
        f"Auditing native source slots {track_slots} versus canonical track {options.track_id}: "
        f"{len(options.camera_ids)} cameras x {len(options.target_slots)} target slots", flush=True,
    )
    for camera_id in options.camera_ids:
        sensor = sensors[camera_id]
        for target_slot in options.target_slots:
            ordinal += 1
            frame_index = int(selected[camera_id][target_slot])
            source_native = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
            source = np.asarray(Image.fromarray(source_native, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR), dtype=np.uint8).astype(np.float32) / 255.0
            camera = _load_ncore_source_camera_path(
                {"frame_indices": [frame_index], "output": {"width": options.width, "height": options.height}}, loader, camera_id
            ).frames[0]
            start = int(camera.timestamp_start_us if camera.timestamp_start_us is not None else camera.timestamp_us)
            end = int(camera.timestamp_us if camera.timestamp_us is not None else start)
            midpoint = (start + end) // 2
            # The projection only needs a midpoint pose.  The renderer itself
            # keeps the source camera's native START/END rolling-shutter poses.
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
            camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
            cuboid = _ftheta_cuboid_mask(
                [track], midpoint, camera_to_world, models[camera_id], height=options.height, width=options.width,
                source_height=int(source_native.shape[0]), source_width=int(source_native.shape[1]),
                margin_pixels=options.cuboid_margin_pixels, device=options.device,
            ).astype(bool)
            probability, margin = segmenter.vehicle_scores(Image.fromarray((source * 255.0).astype(np.uint8), mode="RGB"))
            semantic = vehicle_roi(cuboid, np.ones_like(cuboid), probability, margin,
                                   probability_threshold=options.probability_threshold, margin_threshold=options.margin_threshold)
            roi = semantic if semantic.any() else cuboid
            roi_kind = "vehicle_semantic_intersection" if semantic.any() else "cuboid_fallback_no_vehicle_semantic"
            mask_path = mask_paths[(camera_id, target_slot)]
            if mask_path is None:
                source_exclusion = np.zeros_like(roi)
            else:
                native_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
                source_exclusion = np.asarray(Image.fromarray(native_mask.astype(np.uint8) * 255, mode="L").resize((options.width, options.height), Image.Resampling.NEAREST), dtype=np.uint8) > 0
            ring = surrounding_ring(cuboid, outer_pixels=options.surrounding_ring_outer_pixels, inner_pixels=options.surrounding_ring_inner_pixels)
            clean_ring = ring & ~source_exclusion

            print(f"  [{ordinal:02d}/{count}] rendering static v3", flush=True)
            rendered_v3 = static_v3.render(camera, include_depth=True)
            print(f"  [{ordinal:02d}/{count}] rendering static v4", flush=True)
            rendered_v4 = static_v4.render(camera, include_depth=True)
            print(f"  [{ordinal:02d}/{count}] rendering canonical actor", flush=True)
            rendered_canonical = canonical_renderer.render(camera, include_depth=True)
            print(f"  [{ordinal:02d}/{count}] rendering all native source slots", flush=True)
            rendered_dynamic = all_dynamic_renderer.render(camera, include_depth=True)
            rendered_slots = {}
            for source_slot, renderer in slot_renderers.items():
                print(f"  [{ordinal:02d}/{count}] rendering native source slot {source_slot}", flush=True)
                rendered_slots[source_slot] = renderer.render(camera, include_depth=True)
            dynamic_composite = np.clip(rendered_dynamic.rgb + (1.0 - rendered_dynamic.alpha[..., None]) * rendered_v4.rgb, 0.0, 1.0)
            canonical_composite = np.clip(rendered_canonical.rgb + (1.0 - rendered_canonical.alpha[..., None]) * rendered_v4.rgb, 0.0, 1.0)
            source_camera_candidates = []
            for source_camera_index in track_camera_indices:
                source_slot = nearest_source_slot_for_camera(dynamic, options.track_id, source_camera_index, midpoint)
                all_dynamic_renderer.options = replace(
                    actor_only_options,
                    dynamic_source_camera_index=source_camera_index,
                    dynamic_source_slot=local_slot_by_original[source_slot],
                )
                print(
                    f"  [{ordinal:02d}/{count}] rendering source camera-index {source_camera_index}, slot {source_slot}",
                    flush=True,
                )
                rendered_candidate = all_dynamic_renderer.render(camera, include_depth=True)
                candidate_composite = np.clip(
                    rendered_candidate.rgb + (1.0 - rendered_candidate.alpha[..., None]) * rendered_v4.rgb,
                    0.0, 1.0,
                )
                candidate = {
                    "source_camera_index": source_camera_index, "source_slot": source_slot,
                    "gaussian_count": int(((dynamic.source_track_ids == options.track_id)
                                             & (dynamic.source_camera_indices == source_camera_index)
                                             & (dynamic.source_frame_slots == source_slot)).sum()),
                    **_layer_record(rendered_candidate, source, roi),
                    "v4_composite_mae": masked_mae(candidate_composite, source, roi),
                }
                source_camera_candidates.append(candidate)
            all_dynamic_renderer.options = actor_only_options
            oracle_candidate = min(
                source_camera_candidates,
                key=lambda candidate: float("inf") if candidate["v4_composite_mae"] is None else candidate["v4_composite_mae"],
            )
            slot_rows = []
            for source_slot, rendered in rendered_slots.items():
                row = {"source_slot": int(source_slot), "gaussian_count": int(per_slot_dynamic[source_slot].count)}
                row.update(_layer_record(rendered, source, roi))
                row["alpha_iou_with_canonical"] = alpha_iou(rendered.alpha, rendered_canonical.alpha, roi)
                row["alpha_iou_with_all_dynamic"] = alpha_iou(rendered.alpha, rendered_dynamic.alpha, roi)
                slot_rows.append(row)
            records.append({
                "camera_id": camera_id, "target_slot": int(target_slot), "source_frame_index": frame_index,
                "reference_timestamp_us": int(references[target_slot]), "timestamp_midpoint_us": midpoint,
                "roi_kind": roi_kind, "roi_pixels": int(roi.sum()), "cuboid_pixels": int(cuboid.sum()),
                "semantic_vehicle_pixels": int(semantic.sum()), "clean_static_exclusion_fraction_in_roi": _fraction(source_exclusion, roi),
                "v3_static": {"roi_mae": masked_mae(rendered_v3.rgb, source, roi), **_rgb_depth_difference(rendered_v3, rendered_v4, roi)},
                "v4_static": {"roi_mae": masked_mae(rendered_v4.rgb, source, roi)},
                "surrounding_background": {
                    "ring_pixels": int(ring.sum()), "source_mask_excluded_ring_pixels": int((ring & source_exclusion).sum()),
                    "clean_ring_pixels": int(clean_ring.sum()), "v3_mae": masked_mae(rendered_v3.rgb, source, clean_ring),
                    "v4_mae": masked_mae(rendered_v4.rgb, source, clean_ring), **_rgb_depth_difference(rendered_v3, rendered_v4, clean_ring),
                },
                "all_native_dynamic": _layer_record(rendered_dynamic, source, roi),
                "canonical": _layer_record(rendered_canonical, source, roi),
                "native_canonical_alpha_iou": alpha_iou(rendered_dynamic.alpha, rendered_canonical.alpha, roi),
                "v4_composite_mae": {"all_native_dynamic": masked_mae(dynamic_composite, source, roi), "canonical": masked_mae(canonical_composite, source, roi)},
                "single_observation_candidates": source_camera_candidates,
                "single_observation_oracle": dict(oracle_candidate),
                "source_slots": slot_rows,
            })
            print(
                f"[{ordinal:02d}/{count}] {camera_id} target_slot={target_slot} frame={frame_index} roi={int(roi.sum())} "
                f"native_alpha={_fraction(rendered_dynamic.alpha >= .02, roi)} canonical_alpha={_fraction(rendered_canonical.alpha >= .02, roi)} "
                f"iou={records[-1]['native_canonical_alpha_iou']}", flush=True,
            )
    report = {
        "schema_version": 1,
        "purpose": "Exact-FTheta attribution of native pa-multiview source-slot dynamic coverage, source-camera single-observation candidates, canonical fusion and v3/v4 surrounding static background.",
        "interpretation": {
            "native_vs_canonical": "If native source dynamic coverage/MAE is already poor, canonical fusion is not the root cause. Low alpha IoU with good native performance implicates fusion coverage.",
            "surrounding_background": "The ring excludes the track cuboid and v4 source dynamic mask. A v3/v4 difference here is background-reconstruction error, not actor error.",
            "source_slot": "Each source slot is rendered alone at the target time with its original three-keyframe temporal support; it is not a re-timed static point cloud.",
            "single_observation": "Each candidate retains exactly one exported source-camera index and nearest available source slot. Index-to-camera-ID was not stored in this v2 asset, so candidate ranking is diagnostic only, not a production camera mapping.",
        },
        "inputs": {"dynamic_gaussians": str(options.dynamic_gaussians.resolve()), "canonical_actors": str(options.canonical_actors.resolve()),
                   "static_v3_ply": str(options.static_v3_ply.resolve()), "static_v4_ply": str(options.static_v4_ply.resolve()),
                   "clean_static_mask_dir": str(options.clean_static_mask_dir.resolve())},
        "track": {"track_id": int(options.track_id), "native_source_slots": list(track_slots), "native_source_camera_indices": list(track_camera_indices), "native_track_gaussian_count": int(all_track_dynamic.count), "canonical_gaussian_count": int(canonical.count)},
        "parameters": {"camera_ids": list(options.camera_ids), "target_slots": list(options.target_slots), "resolution": [options.width, options.height],
                       "surrounding_ring_outer_pixels": options.surrounding_ring_outer_pixels, "surrounding_ring_inner_pixels": options.surrounding_ring_inner_pixels},
        "mean": {
            "native_actor_only_mae": _mean([row["all_native_dynamic"]["actor_only_mae"] for row in records]),
            "canonical_actor_only_mae": _mean([row["canonical"]["actor_only_mae"] for row in records]),
            "native_composite_mae": _mean([row["v4_composite_mae"]["all_native_dynamic"] for row in records]),
            "canonical_composite_mae": _mean([row["v4_composite_mae"]["canonical"] for row in records]),
            "native_canonical_alpha_iou": _mean([row["native_canonical_alpha_iou"] for row in records]),
            "single_observation_oracle_composite_mae": _mean([row["single_observation_oracle"]["v4_composite_mae"] for row in records]),
            "single_observation_oracle_alpha_fraction": _mean([row["single_observation_oracle"]["alpha_fraction_in_roi"] for row in records]),
            "v3_static_roi_mae": _mean([row["v3_static"]["roi_mae"] for row in records]),
            "v4_static_roi_mae": _mean([row["v4_static"]["roi_mae"] for row in records]),
            "v3_surrounding_mae": _mean([row["surrounding_background"]["v3_mae"] for row in records]),
            "v4_surrounding_mae": _mean([row["surrounding_background"]["v4_mae"] for row in records]),
            "v3_v4_surrounding_rgb_mae": _mean([row["surrounding_background"]["rgb_mae"] for row in records]),
        },
        "observations": records,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dynamic/canonical attribution audit: {report_path}", flush=True)
    return report_path
