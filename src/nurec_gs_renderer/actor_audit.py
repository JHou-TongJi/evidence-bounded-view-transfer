"""Audit a rigid actor asset on exact NCore FTheta vehicle regions.

The Street-Gaussians training proxy is intentionally not used as the metric
domain here.  This module evaluates rendered layers against the original
front FTheta image and uses an NCore cuboid projection only to constrain a
local SegFormer vehicle segmentation.  It therefore separates three common
failure modes: missing actor coverage, poor actor appearance/geometry, and
harmful static--actor overlap during compositing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import compose_ncore_camera_pose
from .pose_sources import _create_ncore_loader, _ncore_exposure_timestamps
from .street_dataset import (
    SegformerVehicleSegmenter,
    _interpolate_track_pose,
    _semantic_actor_supervision_mask,
)


@dataclass(frozen=True)
class ActorAuditOptions:
    ncore_path: Path
    static_render_dir: Path
    composite_render_dir: Path
    actor_render_dir: Path
    actor_dataset_manifest: Path
    segformer_model_dir: Path
    output_dir: Path
    camera_id: str = "camera_front_wide_120fov"
    device: str = "cuda"
    probability_threshold: float = 0.60
    margin_threshold: float = 0.05
    cuboid_margin_pixels: int = 8
    write_previews: bool = True

    def __post_init__(self) -> None:
        if self.cuboid_margin_pixels < 0:
            raise ValueError("cuboid_margin_pixels must be non-negative")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0, 1]")
        if not -1.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [-1, 1]")


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean absolute RGB error over a non-empty boolean image mask."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must be matching HxWx3 arrays")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask does not match RGB dimensions")
    return None if not mask.any() else float(np.abs(prediction[mask] - target[mask]).mean())


def temporal_delta_residual(
    previous_prediction: np.ndarray,
    prediction: np.ndarray,
    previous_target: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    """Measure how faithfully a rendered frame-to-frame RGB change matches NCore.

    The caller supplies the union of the two semantic actor masks.  Using the
    union, rather than only the current ROI, keeps a moving object's departing
    silhouette in the metric and makes a missing/dragged actor count as a
    temporal error.
    """
    return masked_mae(
        np.asarray(prediction, dtype=np.float32) - np.asarray(previous_prediction, dtype=np.float32),
        np.asarray(target, dtype=np.float32) - np.asarray(previous_target, dtype=np.float32),
        mask,
    )


def vehicle_roi(
    cuboid_mask: np.ndarray,
    valid_mask: np.ndarray,
    vehicle_probability: np.ndarray,
    vehicle_margin: np.ndarray,
    *,
    probability_threshold: float,
    margin_threshold: float,
) -> np.ndarray:
    """Return the conservative ``cuboid ∩ semantic vehicle ∩ valid`` ROI."""
    return _semantic_actor_supervision_mask(
        cuboid_mask,
        valid_mask,
        vehicle_probability,
        vehicle_margin,
        probability_threshold=probability_threshold,
        margin_threshold=margin_threshold,
        erosion_pixels=0,
    ).astype(bool)


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError(f"render manifest is not complete: {path}")
    if not isinstance(manifest.get("frames"), list) or not manifest["frames"]:
        raise ValueError(f"render manifest contains no frames: {path}")
    return manifest


def _read_actor_tracks(path: Path) -> list[dict[str, Any]]:
    """Read the frozen full-chunk trajectories used to train the actor asset.

    ``consolidate_cuboid_tracks`` assigns its integer keys after temporal
    filtering, so a fresh four-frame query can renumber the same physical
    track.  The Street dataset manifest is therefore the only safe source of
    IDs/poses for an audit of its exported canonical actors.
    """
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError(f"actor dataset manifest has no frozen tracks: {path}")
    required = {"track_id", "timestamps_us", "actor_to_world", "length_width_height"}
    if any(not isinstance(track, dict) or not required.issubset(track) for track in tracks):
        raise ValueError(f"actor dataset manifest has malformed tracks: {path}")
    return tracks


def _frame_lookup(manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for frame in manifest["frames"]:
        if not frame.get("complete"):
            continue
        frame_id = str(frame["frame_id"])
        outputs = frame.get("outputs", {})
        if "rgb" not in outputs or "alpha" not in outputs:
            raise ValueError(f"frame {frame_id} in {root} does not include rgb and alpha")
        output[frame_id] = frame
    return output


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _cube_corners(track: dict[str, Any], timestamp_us: int) -> np.ndarray | None:
    pose = _interpolate_track_pose(track, timestamp_us)
    if pose is None:
        return None
    dimensions = np.asarray(track["length_width_height"], dtype=np.float64)
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    local = signs * dimensions[None, :] / 2.0
    return local @ pose[:3, :3].T + pose[:3, 3]


def _ftheta_cuboid_mask(
    tracks: list[dict[str, Any]],
    timestamp_us: int,
    camera_to_world: np.ndarray,
    model: object,
    *,
    height: int,
    width: int,
    source_height: int,
    source_width: int,
    margin_pixels: int,
    device: str,
) -> np.ndarray:
    """Project 3-D cuboid corners through NCore's actual FTheta model.

    The original camera is rolling shutter, while cuboid labels are sampled at
    an instant.  We use the exposure midpoint pose and deliberately enlarge
    only the resulting box; the semantic intersection removes most of this
    unavoidable temporal approximation before scoring.
    """
    import torch

    cuboids = [corners for track in tracks if (corners := _cube_corners(track, timestamp_us)) is not None]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not cuboids:
        return mask
    world_to_camera = np.linalg.inv(camera_to_world)
    corners_world = np.concatenate(cuboids, axis=0)
    corners_camera = corners_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    rays = corners_camera / np.maximum(np.linalg.norm(corners_camera, axis=1, keepdims=True), 1e-12)
    returned = model.camera_rays_to_pixels(torch.from_numpy(rays.astype(np.float32)).to(device))
    pixels = returned.pixels.detach().cpu().numpy()
    # The NCore FTheta model returns coordinates in the native sensor image.
    # Match CameraIntrinsics.resized(): pixel centres undergo scale then the
    # half-pixel offset, rather than a naïve multiplication that shifts boxes
    # at the image boundary.
    pixels[:, 0] = (pixels[:, 0] + 0.5) * (width / source_width) - 0.5
    pixels[:, 1] = (pixels[:, 1] + 0.5) * (height / source_height) - 0.5
    valid = returned.valid_flag.detach().cpu().numpy().astype(bool) & (corners_camera[:, 2] > 0.10)
    for start in range(0, len(corners_camera), 8):
        points = pixels[start : start + 8][valid[start : start + 8]]
        if len(points) < 2:
            continue
        xmin, ymin = np.floor(points.min(axis=0)).astype(int) - margin_pixels
        xmax, ymax = np.ceil(points.max(axis=0)).astype(int) + margin_pixels
        xmin, xmax = max(0, xmin), min(width - 1, xmax)
        ymin, ymax = max(0, ymin), min(height - 1, ymax)
        if xmin <= xmax and ymin <= ymax:
            mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return mask


def _write_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray((np.asarray(value, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)


def _comparison_preview(
    ground_truth: np.ndarray,
    rendered: list[np.ndarray],
    roi: np.ndarray,
    path: Path,
) -> None:
    """Write GT/static/composite/actor diagnostic strip with ROI outlines."""
    panels = [ground_truth, *rendered]
    height, width = ground_truth.shape[:2]
    canvas = Image.new("RGB", (width * len(panels), height))
    outline = np.asarray(roi, dtype=bool)
    edge = outline & ~(
        np.roll(outline, 1, 0) & np.roll(outline, -1, 0) & np.roll(outline, 1, 1) & np.roll(outline, -1, 1)
    )
    for index, panel in enumerate(panels):
        image = np.clip(np.rint(panel * 255.0), 0, 255).astype(np.uint8).copy()
        image[edge] = (255, 32, 32)
        canvas.paste(Image.fromarray(image, mode="RGB"), (index * width, 0))
    canvas.save(path)


def _summary(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def _temporal_summary(frames: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Summarise static-versus-composite temporal residuals over valid pairs."""
    records = [frame["temporal_delta_residual"] for frame in frames if frame.get("temporal_delta_residual")]
    static = np.asarray([record["static"] for record in records if record["static"] is not None], dtype=np.float64)
    composite = np.asarray([record["composite"] for record in records if record["composite"] is not None], dtype=np.float64)
    if len(static) != len(composite):
        raise ValueError("temporal static/composite records must be aligned")
    if not len(static):
        return {"pair_count": 0, "static_mean": None, "composite_mean": None,
                "relative_improvement_percent": None, "improved_pair_fraction": None}
    return {
        "pair_count": int(len(static)),
        "static_mean": float(static.mean()),
        "composite_mean": float(composite.mean()),
        "relative_improvement_percent": float((static.mean() - composite.mean()) / max(static.mean(), 1e-12) * 100.0),
        "improved_pair_fraction": float((composite < static).mean()),
    }


def audit_actor_layers(options: ActorAuditOptions) -> Path:
    """Run the exact-FTheta ROI audit and write a reproducible JSON report."""
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output}")
    (output / "masks").mkdir(parents=True)
    if options.write_previews:
        (output / "previews").mkdir()
    static_root = options.static_render_dir.resolve()
    composite_root = options.composite_render_dir.resolve()
    actor_root = options.actor_render_dir.resolve()
    manifests = [_read_manifest(root) for root in (static_root, composite_root, actor_root)]
    lookups = [_frame_lookup(manifest, root) for manifest, root in zip(manifests, (static_root, composite_root, actor_root), strict=True)]
    common_ids = set(lookups[0]).intersection(lookups[1], lookups[2])
    if not common_ids:
        raise ValueError("the three render directories have no common completed frame IDs")
    ordered = [str(frame["frame_id"]) for frame in manifests[1]["frames"] if str(frame["frame_id"]) in common_ids]
    reference = lookups[1][ordered[0]]
    height = int(reference["intrinsics"]["height"])
    width = int(reference["intrinsics"]["width"])
    if any((int(lookup[frame_id]["intrinsics"]["height"]), int(lookup[frame_id]["intrinsics"]["width"])) != (height, width) for lookup in lookups for frame_id in ordered):
        raise ValueError("all render inputs must have the same output resolution")

    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(options.ncore_path)
    sensor = loader.get_camera_sensor(options.camera_id)
    exposures = _ncore_exposure_timestamps(sensor)
    selected_indices = np.asarray([int(frame_id.split("_", 1)[0]) for frame_id in ordered], dtype=np.int64)
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(exposures)):
        raise ValueError("render frame IDs do not encode valid NCore source frame indices")
    midpoints = ((exposures[selected_indices, 0].astype(np.int64) + exposures[selected_indices, 1].astype(np.int64)) // 2)
    tracks = _read_actor_tracks(options.actor_dataset_manifest)
    model = FThetaCameraModel(sensor.model_parameters, device=options.device)
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    # SegFormer is independent per image.  Batch this expensive inference so a
    # full temporal window is practical, while keeping all cuboid projection
    # and metric calculations below strictly frame-local.
    ground_truth_images: list[Image.Image] = []
    source_sizes: list[tuple[int, int]] = []
    for source_index in selected_indices:
        source = np.asarray(sensor.get_frame_image_array(int(source_index)), dtype=np.uint8)
        source_sizes.append((int(source.shape[0]), int(source.shape[1])))
        ground_truth_images.append(
            Image.fromarray(source, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        )
    vehicle_scores = segmenter.vehicle_scores_batch(ground_truth_images, batch_size=8)
    frames: list[dict[str, Any]] = []
    previous_ground_truth: np.ndarray | None = None
    previous_roi: np.ndarray | None = None
    previous_rgb: list[np.ndarray] | None = None
    for ordinal, (frame_id, source_index, midpoint, ground_truth_image, source_size, scores) in enumerate(
        zip(ordered, selected_indices, midpoints, ground_truth_images, source_sizes, vehicle_scores, strict=True), start=1
    ):
        print(f"[{ordinal}/{len(ordered)}] FTheta vehicle ROI: source frame {source_index}", flush=True)
        ground_truth = np.asarray(ground_truth_image, dtype=np.float32) / 255.0
        rig_to_world = np.asarray(
            loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64))
        )[0]
        camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
        cuboid = _ftheta_cuboid_mask(
            tracks, int(midpoint), camera_to_world, model, height=height, width=width,
            source_height=source_size[0], source_width=source_size[1],
            margin_pixels=options.cuboid_margin_pixels, device=options.device,
        ).astype(bool)
        probability, margin = scores
        roi = vehicle_roi(cuboid, np.ones((height, width), dtype=bool), probability, margin,
                          probability_threshold=options.probability_threshold, margin_threshold=options.margin_threshold)
        static_frame, composite_frame, actor_frame = (lookup[frame_id] for lookup in lookups)
        roots = (static_root, composite_root, actor_root)
        rgb = [_read_rgb(root / frame["outputs"]["rgb"]) for root, frame in zip(roots, (static_frame, composite_frame, actor_frame), strict=True)]
        alpha = [_read_alpha(root / frame["outputs"]["alpha"]) for root, frame in zip(roots, (static_frame, composite_frame, actor_frame), strict=True)]
        masks = {
            "all": np.ones((height, width), dtype=bool),
            "cuboid": cuboid,
            "vehicle_roi": roi,
            "outside_vehicle_roi": ~roi,
        }
        metrics = {
            name: {label: masked_mae(image, ground_truth, mask) for label, mask in masks.items()}
            for name, image in zip(("static", "composite", "actor_only"), rgb, strict=True)
        }
        actor_present = alpha[2] >= 0.02
        static_present = alpha[0] >= 0.02
        overlap = actor_present & static_present
        metrics["composite_delta_from_static"] = {label: masked_mae(rgb[1], rgb[0], mask) for label, mask in masks.items()}
        mask_stats = {
            "cuboid_fraction": float(cuboid.mean()), "vehicle_roi_fraction": float(roi.mean()),
            "actor_alpha_fraction_in_roi": None if not roi.any() else float(actor_present[roi].mean()),
            "static_actor_alpha_overlap_fraction_in_roi": None if not roi.any() else float(overlap[roi].mean()),
            "composite_changed_fraction_in_roi": None if not roi.any() else float((np.abs(rgb[1] - rgb[0]).max(axis=-1)[roi] > 1.0 / 255.0).mean()),
            "composite_changed_fraction_outside_vehicle_roi": float(
                (np.abs(rgb[1] - rgb[0]).max(axis=-1)[~roi] > 1.0 / 255.0).mean()
            ),
        }
        prefix = f"{ordinal - 1:06d}_{frame_id}"
        _write_mask(output / "masks" / f"{prefix}_cuboid.png", cuboid)
        _write_mask(output / "masks" / f"{prefix}_vehicle_roi.png", roi)
        if options.write_previews:
            _comparison_preview(ground_truth, rgb, roi, output / "previews" / f"{prefix}.png")
        temporal: dict[str, float | int | None] | None = None
        if previous_ground_truth is not None and previous_roi is not None and previous_rgb is not None:
            temporal_mask = roi | previous_roi
            temporal = {
                "mask_pixel_count": int(temporal_mask.sum()),
                "static": temporal_delta_residual(previous_rgb[0], rgb[0], previous_ground_truth, ground_truth, temporal_mask),
                "composite": temporal_delta_residual(previous_rgb[1], rgb[1], previous_ground_truth, ground_truth, temporal_mask),
                "actor_only": temporal_delta_residual(previous_rgb[2], rgb[2], previous_ground_truth, ground_truth, temporal_mask),
                "composite_delta_from_static": temporal_delta_residual(previous_rgb[1], rgb[1], previous_rgb[0], rgb[0], temporal_mask),
            }
        frames.append({"frame_id": frame_id, "source_frame_index": int(source_index), "timestamp_midpoint_us": int(midpoint), "metrics": metrics, "mask_stats": mask_stats, "temporal_delta_residual": temporal})
        previous_ground_truth, previous_roi, previous_rgb = ground_truth, roi, rgb

    report = {
        "schema_version": 1,
        "purpose": "Exact original-FTheta vehicle ROI decomposition: static vs static+actor vs actor-only.",
        "inputs": {"ncore_path": str(options.ncore_path.resolve()), "camera_id": options.camera_id,
                   "static_render_dir": str(static_root), "composite_render_dir": str(composite_root),
                   "actor_render_dir": str(actor_root),
                   "actor_dataset_manifest": str(options.actor_dataset_manifest.resolve()),
                   "segformer_model_dir": str(options.segformer_model_dir.resolve())},
        "roi": {"definition": "projected NCore rigid cuboid (exposure midpoint, FTheta) ∩ high-confidence local SegFormer car/bus/truck/van",
                "probability_threshold": options.probability_threshold, "margin_threshold": options.margin_threshold,
                "cuboid_margin_pixels": options.cuboid_margin_pixels, "labels": list(segmenter.vehicle_labels)},
        "write_previews": options.write_previews,
        "frame_count": len(frames), "frames": frames,
        "mean_mae": {method: {region: _summary([frame["metrics"][method][region] for frame in frames]) for region in ("all", "cuboid", "vehicle_roi", "outside_vehicle_roi")}
                     for method in ("static", "composite", "actor_only", "composite_delta_from_static")},
        "mean_mask_stats": {name: _summary([frame["mask_stats"][name] for frame in frames]) for name in frames[0]["mask_stats"]},
        "temporal_delta": {
            "definition": "MAE((render_t-render_t-1) - (NCore_t-NCore_t-1)) over union of adjacent semantic vehicle ROIs",
            **_temporal_summary(frames),
        },
    }
    report_path = output / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote exact-FTheta actor audit: {report_path}", flush=True)
    return report_path
