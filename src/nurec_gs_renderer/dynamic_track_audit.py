"""Exact-camera evidence audit for one associated InstantNuRec dynamic track.

This deliberately precedes any class override or actor optimisation.  It
isolates the source dynamic Gaussians attached to one hash-bound association
track, renders them with the production FTheta/rolling-shutter path, and
compares their footprint with the original image, NCore cuboid and conservative
vehicle semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask, vehicle_roi
from .camera import compose_ncore_camera_pose
from .dynamic import DynamicGaussianScene, load_dynamic_gaussians
from .dynamic_association import DynamicActorAssociation
from .ply_io import load_gaussian_ply
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path, _ncore_exposure_timestamps
from .render import GsplatRenderer, RenderOptions
from .street_dataset import SegformerVehicleSegmenter


@dataclass(frozen=True)
class DynamicTrackAuditOptions:
    dynamic_gaussians: Path
    dynamic_actor_association: Path
    ncore_path: Path
    static_ply: Path
    segformer_model_dir: Path
    output_dir: Path
    track_id: int
    frame_indices: tuple[int, ...] = (100, 107, 157, 160)
    camera_id: str = "camera_front_wide_120fov"
    width: int = 960
    height: int = 540
    device: str = "cuda"
    probability_threshold: float = 0.60
    margin_threshold: float = 0.05
    cuboid_margin_pixels: int = 8

    def __post_init__(self) -> None:
        if not self.frame_indices or any(index < 0 for index in self.frame_indices):
            raise ValueError("frame_indices must be non-empty and non-negative")
        if self.width <= 0 or self.height <= 0 or self.cuboid_margin_pixels < 0:
            raise ValueError("image dimensions and cuboid margin are invalid")
        if not 0.0 <= self.probability_threshold <= 1.0 or not -1.0 <= self.margin_threshold <= 1.0:
            raise ValueError("semantic thresholds are invalid")


def isolate_associated_track(
    scene: DynamicGaussianScene,
    association: DynamicActorAssociation,
    track_id: int,
) -> tuple[DynamicGaussianScene, dict[str, object]]:
    """Return the exact source dynamic Gaussians belonging to one numeric track."""
    matching = [index for index, record in enumerate(association.tracks) if str(record.get("track_id")) == str(track_id)]
    if len(matching) != 1:
        raise ValueError(f"association has {len(matching)} tracks matching track_id={track_id}; expected one")
    track_index = matching[0]
    selected = association.track_indices == track_index
    if not selected.any():
        raise ValueError(f"track {track_id} has no associated dynamic Gaussians")
    metadata = dict(scene.metadata)
    metadata.update({
        "isolated_association_track_id": int(track_id),
        "isolated_association_track_index": int(track_index),
        "isolated_dynamic_gaussian_count": int(selected.sum()),
        "isolation": "hash-bound dynamic actor association; source properties preserved",
    })
    isolated = DynamicGaussianScene(
        keyframe_positions=scene.keyframe_positions[selected].copy(),
        keyframe_timestamps_us=scene.keyframe_timestamps_us[selected].copy(),
        rotations_wxyz=scene.rotations_wxyz[selected].copy(), scales=scene.scales[selected].copy(),
        rgb=scene.rgb[selected].copy(), max_opacities=scene.max_opacities[selected].copy(), metadata=metadata,
    )
    return isolated, dict(association.tracks[track_index])


def _load_track_geometry(ncore_path: Path, scene: DynamicGaussianScene, track_id: int) -> dict[str, object]:
    """Recover the frozen NCore cuboid trajectory for the association's raw ID."""
    try:
        from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
        from instant_nurec.utils.types import HalfClosedInterval
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("track audit requires the local InstantNuRec checkout on PYTHONPATH") from exc
    loader = _create_ncore_loader(ncore_path)
    middle = scene.keyframe_timestamps_us[:, 1]
    interval = HalfClosedInterval(int(middle.min()) - 1_000_000, int(middle.max()) + 1_000_001)
    tracks = consolidate_cuboid_tracks(compute_cuboid_df(loader, interval), loader, ["AUTOLABEL"], 0.0, np.eye(4))
    record = next((track for raw_id, track in tracks.items() if str(raw_id) == str(track_id)), None)
    if record is None:
        raise ValueError(f"NCore query could not recover association track {track_id}")
    times = np.asarray(record["timestamps_us"], dtype=np.int64)
    poses = np.asarray(record["poses"], dtype=np.float32)
    dimensions = np.asarray(record.get("dimension", record.get("size")), dtype=np.float32).reshape(-1)
    if len(times) < 2 or poses.shape != (len(times), 4, 4) or dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError(f"NCore association track {track_id} has malformed geometry")
    return {
        "track_id": int(track_id), "label": str(record["label_class"]),
        "timestamps_us": times.tolist(), "actor_to_world": poses.tolist(),
        "length_width_height": dimensions.astype(float).tolist(),
    }


def _overlay(base: np.ndarray, cuboid: np.ndarray, semantic: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Overlay cuboid boundary (red), semantic vehicle (green), alpha (blue)."""
    output = np.asarray(base, dtype=np.uint8).copy()
    box = np.asarray(cuboid, dtype=bool)
    edge = box & ~(np.roll(box, 1, 0) & np.roll(box, -1, 0) & np.roll(box, 1, 1) & np.roll(box, -1, 1))
    output[semantic] = (0.55 * output[semantic] + 0.45 * np.asarray([0, 255, 0])).astype(np.uint8)
    output[alpha >= 0.02] = (0.55 * output[alpha >= 0.02] + 0.45 * np.asarray([0, 96, 255])).astype(np.uint8)
    output[edge] = (255, 32, 32)
    return output


def audit_dynamic_track(options: DynamicTrackAuditOptions) -> Path:
    """Write exact-camera images, masks and a report for a candidate dynamic track."""
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty track-audit output: {output}")
    scene = load_dynamic_gaussians(options.dynamic_gaussians)
    association = DynamicActorAssociation.load(options.dynamic_actor_association)
    association.validate_source(options.dynamic_gaussians, scene)
    isolated, associated_track = isolate_associated_track(scene, association, options.track_id)
    geometry_track = _load_track_geometry(options.ncore_path, scene, options.track_id)
    if str(associated_track.get("class_id")) != str(geometry_track["label"]):
        raise ValueError("association and fresh NCore track labels disagree; refusing ambiguous audit")

    (output / "rgb").mkdir(parents=True)
    (output / "alpha").mkdir()
    (output / "masks").mkdir()
    (output / "previews").mkdir()
    static_scene = load_gaussian_ply(options.static_ply)
    renderer = GsplatRenderer(
        static_scene, RenderOptions(device=options.device, static_opacity_scale=0.0), dynamic_scene=isolated
    )
    loader = _create_ncore_loader(options.ncore_path)
    sensor = loader.get_camera_sensor(options.camera_id)
    exposures = _ncore_exposure_timestamps(sensor)
    from ncore.impl.sensors.camera import FThetaCameraModel

    model = FThetaCameraModel(sensor.model_parameters, device=options.device)
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    frames: list[dict[str, object]] = []
    for ordinal, frame_index in enumerate(options.frame_indices, start=1):
        if not 0 <= frame_index < len(exposures):
            raise ValueError(f"frame index out of range: {frame_index}")
        camera = _load_ncore_source_camera_path(
            {"frame_indices": [frame_index], "output": {"width": options.width, "height": options.height}}, loader, options.camera_id
        ).frames[0]
        rendered = renderer.render(camera, include_depth=True)
        source = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
        target = np.asarray(
            Image.fromarray(source, mode="RGB").resize((options.width, options.height), Image.Resampling.BILINEAR), dtype=np.uint8
        )
        midpoint = int((int(exposures[frame_index, 0]) + int(exposures[frame_index, 1])) // 2)
        rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
        camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
        cuboid = _ftheta_cuboid_mask(
            [geometry_track], midpoint, camera_to_world, model, height=options.height, width=options.width,
            source_height=int(source.shape[0]), source_width=int(source.shape[1]),
            margin_pixels=options.cuboid_margin_pixels, device=options.device,
        ).astype(bool)
        probability, margin = segmenter.vehicle_scores(Image.fromarray(target, mode="RGB"))
        semantic = vehicle_roi(cuboid, np.ones_like(cuboid, dtype=bool), probability, margin,
                               probability_threshold=options.probability_threshold, margin_threshold=options.margin_threshold)
        alpha = np.asarray(rendered.alpha, dtype=np.float32)
        actor = alpha >= 0.02
        stem = f"{ordinal - 1:06d}_{frame_index:06d}"
        Image.fromarray(np.clip(np.rint(rendered.rgb * 255.0), 0, 255).astype(np.uint8), mode="RGB").save(output / "rgb" / f"{stem}.png")
        Image.fromarray(np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8), mode="L").save(output / "alpha" / f"{stem}.png")
        Image.fromarray((cuboid * 255).astype(np.uint8), mode="L").save(output / "masks" / f"{stem}_cuboid.png")
        Image.fromarray((semantic * 255).astype(np.uint8), mode="L").save(output / "masks" / f"{stem}_vehicle_semantic.png")
        alpha_rgb = np.repeat(np.clip(np.rint(alpha[..., None] * 255.0), 0, 255).astype(np.uint8), 3, axis=2)
        panels = [target, np.clip(np.rint(rendered.rgb * 255.0), 0, 255).astype(np.uint8), alpha_rgb, _overlay(target, cuboid, semantic, alpha)]
        preview = Image.new("RGB", (options.width * len(panels), options.height))
        for index, panel in enumerate(panels):
            preview.paste(Image.fromarray(panel, mode="RGB"), (index * options.width, 0))
        preview.save(output / "previews" / f"{stem}.png")
        frames.append({
            "source_frame_index": int(frame_index), "timestamp_midpoint_us": midpoint,
            "cuboid_pixels": int(cuboid.sum()), "semantic_vehicle_pixels": int(semantic.sum()),
            "actor_alpha_pixels": int(actor.sum()), "actor_in_cuboid_pixels": int((actor & cuboid).sum()),
            "actor_in_semantic_vehicle_pixels": int((actor & semantic).sum()),
            "actor_coverage_of_cuboid": float((actor & cuboid).sum() / max(int(cuboid.sum()), 1)),
            "actor_coverage_of_semantic_vehicle": float((actor & semantic).sum() / max(int(semantic.sum()), 1)),
            "semantic_fraction_of_cuboid": float(semantic.sum() / max(int(cuboid.sum()), 1)),
            "preview": f"previews/{stem}.png",
        })
        print(
            f"[{ordinal}/{len(options.frame_indices)}] frame={frame_index} cuboid={int(cuboid.sum())} "
            f"semantic={int(semantic.sum())} actor={int(actor.sum())} actor∩semantic={int((actor & semantic).sum())}", flush=True,
        )
    report = {
        "schema_version": 1,
        "purpose": "Candidate track identity/visibility audit before rigid-vehicle canonical fusion or training.",
        "inputs": {"dynamic_gaussians": str(options.dynamic_gaussians.resolve()), "association": str(options.dynamic_actor_association.resolve()), "ncore_path": str(options.ncore_path.resolve()), "static_ply": str(options.static_ply.resolve())},
        "candidate": {"association_track": associated_track, "ncore_track": geometry_track, "isolated_dynamic_gaussian_count": isolated.count},
        "render": {"camera_id": options.camera_id, "resolution": [options.width, options.height], "static_opacity_scale": 0.0, "dynamic_time_mode": "blend"},
        "semantic_roi": {"definition": "projected candidate cuboid ∩ high-confidence SegFormer car/bus/truck/van", "probability_threshold": options.probability_threshold, "margin_threshold": options.margin_threshold},
        "frames": frames,
        "interpretation": "Do not whitelist this track for rigid vehicle training until visual identity and semantic/alpha overlap support it.",
    }
    path = output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dynamic track audit: {path}", flush=True)
    return path

