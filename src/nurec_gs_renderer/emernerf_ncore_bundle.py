"""Architecture-neutral NCore ray-bundle contract for an EmerNeRF pilot.

The public EmerNeRF loaders assume pinhole Waymo/nuScenes cameras.  This module
does *not* convert NCore fisheye images to pinhole tiles.  Instead it freezes a
small selection of native image records and supplies the information an adapter
needs to generate exact FTheta + rolling-shutter ray batches at run time.

The asset is deliberately a JSON manifest that references original NCore data;
it never copies camera frames, LiDAR clouds or model weights into this repo.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .color_correction import file_sha256
from .pose_sources import _create_ncore_loader


SCHEMA = "ncore-emernerf-native-ray-bundle"
VERSION = 1
TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"
DYNAMIC_RECONSTRUCTION_SCHEMA = "ncore-dynamic-reconstruction-dataset"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def output_pixels_to_native(pixels_xy: np.ndarray, *, output_width: int, output_height: int, source_width: int, source_height: int) -> np.ndarray:
    """Map output pixel centres to native NCore image-point coordinates.

    This keeps the native FTheta model unchanged.  It is intentionally not an
    intrinsic resize: the downstream ray source calls NCore's native FTheta
    image-point unprojection at these floating-point coordinates.
    """
    pixels = np.asarray(pixels_xy, np.float32)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_xy must have shape [N,2]")
    if min(output_width, output_height, source_width, source_height) <= 0:
        raise ValueError("image dimensions must be positive")
    result = np.empty_like(pixels)
    result[:, 0] = (pixels[:, 0] + .5) * source_width / output_width - .5
    result[:, 1] = (pixels[:, 1] + .5) * source_height / output_height - .5
    return result


def sampled_output_pixels(width: int, height: int) -> np.ndarray:
    """Five deterministic points used for a calibration-only CPU ray audit."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    return np.asarray(((0, 0), (width - 1, 0), (width // 2, height // 2), (0, height - 1), (width - 1, height - 1)), np.float32)


@dataclass(frozen=True)
class EmerNeRFNCoreBundleOptions:
    trusted_manifest: Path
    output: Path
    camera_ids: tuple[str, ...]
    reference_start: int
    reference_end: int
    output_width: int = 240
    output_height: int = 135
    holdout_camera_ids: tuple[str, ...] = ()
    actor_track_ids: tuple[int, ...] = ()
    allow_holdout_background_train: bool = False
    ray_smoke: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be non-empty and unique")
        if self.reference_start < 0 or self.reference_end <= self.reference_start:
            raise ValueError("reference range is invalid")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output dimensions must be positive")
        if set(self.holdout_camera_ids).difference(self.camera_ids):
            raise ValueError("holdout cameras must be selected cameras")
        if any(int(track_id) < 0 for track_id in self.actor_track_ids):
            raise ValueError("actor_track_ids must be non-negative")
        if len(set(self.actor_track_ids)) != len(self.actor_track_ids):
            raise ValueError("actor_track_ids must be unique")
        if self.output.suffix != ".json":
            raise ValueError("output must be a .json manifest")


def _record_payload(record: dict[str, Any], *, reference: int, camera_index: int, timestamp_min: int, timestamp_max: int, split: str,
                    evaluation_holdout: bool, default_mask_root: str) -> dict[str, Any]:
    source_width, source_height = (int(value) for value in record["source_resolution"])
    midpoint = int(record["timestamp_midpoint_us"])
    span = max(timestamp_max - timestamp_min, 1)
    instances = []
    for item in record.get("instances", []):
        instances.append({
            "track_id": int(item["track_id"]), "actor_family": str(item.get("actor_family", "unknown")),
            "label": str(item.get("label", "unknown")), "mask": item.get("mask"),
            "mask_root": item.get("mask_root", default_mask_root),
            "mask_origin": item.get("mask_origin"), "lidar_depth": item.get("lidar_depth"), "lidar_depth_root": item.get("lidar_depth_root"),
        })
    return {
        "reference_frame_index": int(reference), "camera_id": str(record["camera_id"]), "camera_index": int(camera_index), "split": split,
        "evaluation_holdout": bool(evaluation_holdout),
        "source_frame_index": int(record["source_frame_index"]), "source_resolution": [source_width, source_height],
        "timestamp_start_us": int(record["timestamp_start_us"]), "timestamp_end_us": int(record["timestamp_end_us"]), "timestamp_midpoint_us": midpoint,
        "normalized_timestamp_start": (int(record["timestamp_start_us"]) - timestamp_min) / span,
        "normalized_timestamp_end": (int(record["timestamp_end_us"]) - timestamp_min) / span,
        "normalized_timestamp_midpoint": (midpoint - timestamp_min) / span,
        "camera_to_world_start": record["camera_to_world_start"], "camera_to_world_end": record["camera_to_world_end"],
        "lidar_frame_index": None if record.get("lidar_frame_index") is None else int(record["lidar_frame_index"]),
        "lidar_timestamp_us": None if record.get("lidar_timestamp_us") is None else int(record["lidar_timestamp_us"]), "instances": instances,
    }


def _ray_smoke(manifest: dict[str, Any], records: list[dict[str, Any],], options: EmerNeRFNCoreBundleOptions) -> list[dict[str, Any]]:
    """Check exact NCore rolling rays at five output pixels per selected camera."""
    from ncore.impl.sensors.camera import FThetaCameraModel

    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        camera_id = str(record["camera_id"])
        if camera_id in seen:
            continue
        seen.add(camera_id)
        sensor = loader.get_camera_sensor(camera_id)
        model = FThetaCameraModel(sensor.model_parameters, device=options.device)
        output_pixels = sampled_output_pixels(options.output_width, options.output_height)
        source_width, source_height = (int(value) for value in record["source_resolution"])
        native_pixels = output_pixels_to_native(output_pixels, output_width=options.output_width, output_height=options.output_height, source_width=source_width, source_height=source_height)
        returned = model.image_points_to_world_rays_shutter_pose(
            native_pixels, np.asarray(record["camera_to_world_start"], np.float64), np.asarray(record["camera_to_world_end"], np.float64),
            int(record["timestamp_start_us"]), int(record["timestamp_end_us"]), return_timestamps=True,
        )
        world_rays = returned.world_rays.detach().float().cpu().numpy()
        timestamps = returned.timestamps_us.detach().cpu().numpy().astype(np.int64)
        direction_norm = np.linalg.norm(world_rays[:, 3:], axis=1)
        probes.append({"camera_id": camera_id, "reference_frame_index": int(record["reference_frame_index"]),
                       "native_pixels_xy": native_pixels.round(4).tolist(), "world_ray_direction_norm": direction_norm.round(7).tolist(),
                       "ray_timestamp_us": timestamps.tolist(), "all_finite": bool(np.isfinite(world_rays).all())})
    return probes


def build_emernerf_ncore_bundle(options: EmerNeRFNCoreBundleOptions) -> Path:
    """Freeze native image/pose/LiDAR references for an EmerNeRF adapter pilot."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite bundle manifest: {options.output}")
    trusted_path = options.trusted_manifest.resolve(); trusted = _read_json(trusted_path)
    source_schema = trusted.get("schema")
    if source_schema not in (TRUSTED_SCHEMA, DYNAMIC_RECONSTRUCTION_SCHEMA) or trusted.get("status") != "complete":
        raise ValueError("source manifest must be a complete trusted supervision or dynamic reconstruction dataset")
    roots = dict(trusted.get("roots", {}))
    roots.setdefault("source_dataset", str(trusted_path.parent))
    selected: list[tuple[int, dict[str, Any]]] = []
    required = set(options.camera_ids)
    for frame in trusted["frames"]:
        reference = int(frame["reference_frame_index"])
        if not options.reference_start <= reference < options.reference_end:
            continue
        by_camera = {str(record["camera_id"]): record for record in frame["cameras"]}
        if required.difference(by_camera):
            continue
        selected.extend((reference, by_camera[camera]) for camera in options.camera_ids)
    if not selected:
        raise RuntimeError("no complete selected multi-camera references in requested window")
    all_times = [int(record[key]) for _, record in selected for key in ("timestamp_start_us", "timestamp_end_us")]
    timestamp_min, timestamp_max = min(all_times), max(all_times)
    camera_index = {camera: index for index, camera in enumerate(options.camera_ids)}
    records = []
    for reference, record in selected:
        is_holdout = str(record["camera_id"]) in options.holdout_camera_ids
        records.append(_record_payload(
            record, reference=reference, camera_index=camera_index[str(record["camera_id"])], timestamp_min=timestamp_min,
            timestamp_max=timestamp_max, split="train" if (options.allow_holdout_background_train or not is_holdout) else "holdout",
            evaluation_holdout=is_holdout, default_mask_root="source_dataset",
        ))
    lidar = sorted({(int(item["lidar_frame_index"]), int(item["lidar_timestamp_us"])) for item in records if item["lidar_frame_index"] is not None})
    payload: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "representation": "native_ncore_ftheta_rolling_ray_bundle",
        "trusted_manifest": str(trusted_path), "trusted_manifest_sha256": file_sha256(trusted_path), "source_schema": source_schema,
        "ncore_path": trusted["ncore_path"], "roots": roots,
        "camera_ids": list(options.camera_ids), "camera_model": "NCore FThetaCameraModel; rays generated at native image points with exposure start/end poses",
        "output_resolution": [options.output_width, options.output_height], "reference_range": [options.reference_start, options.reference_end],
        "timestamp_range_us": [timestamp_min, timestamp_max], "records": records,
        "actor_track_ids": [int(value) for value in options.actor_track_ids],
        "evaluation_protocol": "actor_holdout_with_background_context" if options.allow_holdout_background_train else "strict_camera_holdout",
        "lidar_scans": [{"frame_index": index, "timestamp_us": timestamp} for index, timestamp in lidar],
        "summary": {"reference_count": len(records) // len(options.camera_ids), "image_records": len(records), "train_records": sum(item["split"] == "train" for item in records),
                    "holdout_records": sum(bool(item["evaluation_holdout"]) for item in records), "lidar_scan_count": len(lidar)},
        "limitations": ["This manifest references original NCore images and point clouds; it does not copy frames or convert FTheta to pinhole.",
                        "Dynamic masks remain conservative supervision/evaluation metadata, not 3D pseudo-labels.",
                        "A dedicated EmerNeRF NCore pixel source must generate rolling rays at run time; stock Waymo/NuScenes loaders are not valid for this bundle."],
    }
    if options.ray_smoke:
        payload["ray_smoke"] = _ray_smoke(trusted, records, options)
    _atomic_json(options.output, payload)
    print(f"Wrote EmerNeRF native NCore ray bundle: {options.output} ({len(records)} images, {len(lidar)} LiDAR scans)", flush=True)
    return options.output
