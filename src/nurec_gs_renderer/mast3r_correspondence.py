"""MASt3R correspondence assets for native NCore dynamic-object supervision.

This module deliberately treats MASt3R as a *frozen proposal generator*, not
as a source of camera geometry.  Every returned correspondence is reprojected
through NCore's original FTheta/rolling-shutter API and is rejected unless it
is mutually inside the source instance masks, triangulates inside the tracked
actor cuboid and is geometrically self-consistent.  The output is a compact,
hash-bound actor-local point asset suitable for a later occupancy field; it
never writes a PLY, a Gaussian actor, or pseudo RGB labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .camera import invert_pose
from .roma_multiview_audit import (
    _native_reprojection_error,
    _native_world_rays,
    _rectify_tangent_patch,
    _lidar_distances,
    pixels_in_mask,
    sample_raw_pixel_map,
    triangulate_rays,
)
from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-mast3r-ftheta-dynamic-correspondence"
VERSION = 1
DATASET_SCHEMA = "ncore-dynamic-reconstruction-dataset"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".npz") as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((value for value in record["instances"] if int(value["track_id"]) == track_id), None)


def _load_mask(dataset_root: Path, instance: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(dataset_root / str(instance["mask"])).convert("L"), np.uint8) > 0
    if mask.shape != shape:
        raise ValueError(f"source mask/image shape mismatch: {instance['mask']}")
    return mask


def _mast3r_paths(root: Path) -> None:
    for value in (root, root / "dust3r"):
        if not value.is_dir():
            raise FileNotFoundError(f"MASt3R source/submodule missing: {value}")
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))


def _load_mast3r(root: Path, weights: Path, device: str) -> Any:
    """Load the HF snapshot only from the supplied local directory."""
    _mast3r_paths(root)
    if not (weights / "config.json").is_file() or not (weights / "model.safetensors").is_file():
        raise FileNotFoundError("MASt3R local HF snapshot needs config.json and model.safetensors")
    try:
        import torch
        from mast3r.model import AsymmetricMASt3R
    except ImportError as exc:  # pragma: no cover - depends on optional external checkout
        raise RuntimeError("MASt3R import failed; add its repository and dust3r submodule to PYTHONPATH") from exc
    # The source repository calls the Hugging Face mixin for a directory.  The
    # explicit flag plus the caller's offline environment makes a missing local
    # file an error rather than an accidental download.
    model = AsymmetricMASt3R.from_pretrained(str(weights), local_files_only=True)
    return model.to(device).eval()


def _mast3r_view(patch: np.ndarray, name: str) -> dict[str, Any]:
    """Build a minimal DUSt3R view dictionary from an already rectified patch."""
    import torch
    from dust3r.utils.image import ImgNorm

    image = Image.fromarray(np.asarray(patch, np.uint8), mode="RGB")
    tensor = ImgNorm(image)
    return {
        "img": tensor[None],
        "true_shape": np.asarray([[tensor.shape[1], tensor.shape[2]]], np.int32),
        "idx": 0,
        "instance": name,
    }


def _mast3r_matches(model: Any, patch_a: np.ndarray, patch_b: np.ndarray, *, device: str, subsample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mutual MASt3R descriptor matches and their geometric confidence."""
    import torch
    # ``mast3r.cloud_opt.sparse_ga`` also imports the optional global-alignment
    # package named ``roma``.  That package is unrelated to descriptor
    # matching, and installing it solely to import two helpers would mutate the
    # established instant-nurec environment.  Keep the two small upstream
    # operations here and depend only on MASt3R's fast mutual-NN primitive.
    from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres

    with torch.inference_mode():
        first, second = _mast3r_view(patch_a, "a"), _mast3r_view(patch_b, "b")
        shape_a = torch.from_numpy(first["true_shape"]).to(device)
        shape_b = torch.from_numpy(second["true_shape"]).to(device)
        image_a, image_b = first["img"].to(device), second["img"].to(device)
        feature_a, feature_b, position_a, position_b = model._encode_image_pairs(image_a, image_b, shape_a, shape_b)

        def decode(feature_one: Any, position_one: Any, feature_two: Any, position_two: Any, shape_one: Any, shape_two: Any) -> tuple[Any, Any]:
            decoder_one, decoder_two = model._decoder(feature_one, position_one, feature_two, position_two)
            with torch.amp.autocast("cuda", enabled=False):
                result_one = model._downstream_head(1, [token.float() for token in decoder_one], shape_one)
                result_two = model._downstream_head(2, [token.float() for token in decoder_two], shape_two)
            return result_one, result_two

        # Keep both decode directions: the upstream extractor merges matches
        # proposed from either view, which reduces one-way descriptor holes.
        result = (*decode(feature_a, position_a, feature_b, position_b, shape_a, shape_b),
                  *decode(feature_b, position_b, feature_a, position_a, shape_b, shape_a))
        descriptions = [value["desc"][0] for value in result]
        confidences = [value["desc_conf"][0] for value in result]
        desc_aa, desc_ba, desc_bb, desc_ab = descriptions
        conf_aa, conf_ba, conf_bb, conf_ab = [value.detach().cpu() for value in confidences]
        index_a: list[np.ndarray] = []
        index_b: list[np.ndarray] = []
        confidence_a: list[np.ndarray] = []
        confidence_b: list[np.ndarray] = []
        for first, second, first_conf, second_conf in ((desc_aa, desc_ba, conf_aa, conf_ba), (desc_ab, desc_bb, conf_ab, conf_bb)):
            first_to_second = fast_reciprocal_NNs(first, second, subsample_or_initxy1=subsample, ret_xy=False, device=device, dist="dot", block_size=2 ** 13)
            second_to_first = fast_reciprocal_NNs(second, first, subsample_or_initxy1=subsample, ret_xy=False, device=device, dist="dot", block_size=2 ** 13)
            index_a.append(np.r_[first_to_second[0], second_to_first[1]])
            index_b.append(np.r_[first_to_second[1], second_to_first[0]])
            confidence_a.append(first_conf.reshape(-1).numpy()[index_a[-1]])
            confidence_b.append(second_conf.reshape(-1).numpy()[index_b[-1]])
        xy_a, xy_b, index = merge_corres(np.concatenate(index_a), np.concatenate(index_b), desc_aa.shape[:2], desc_bb.shape[:2], ret_xy=True, ret_index=True)
        confidence = torch.from_numpy(np.sqrt(np.concatenate(confidence_a)[index] * np.concatenate(confidence_b)[index])).to(device)
    return (
        np.asarray(xy_a, np.float32), np.asarray(xy_b, np.float32),
        confidence.detach().float().cpu().numpy().reshape(-1),
    )


def correspondence_acceptance(
    *, angle_deg: np.ndarray, separation_m: np.ndarray, distances_m: np.ndarray, in_cuboid: np.ndarray,
    reprojection_a_px: np.ndarray, reprojection_b_px: np.ndarray, min_ray_angle_deg: float,
    max_ray_separation_m: float, max_reprojection_error_px: float,
) -> np.ndarray:
    """One conservative, testable acceptance predicate for frozen proposals."""
    arrays = [np.asarray(value) for value in (angle_deg, separation_m, distances_m, in_cuboid, reprojection_a_px, reprojection_b_px)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("correspondence evidence arrays must have equal length")
    return (
        np.isfinite(angle_deg) & np.isfinite(separation_m) & np.isfinite(reprojection_a_px) & np.isfinite(reprojection_b_px)
        & (angle_deg >= min_ray_angle_deg) & (separation_m <= max_ray_separation_m)
        & np.all(np.asarray(distances_m) > .1, axis=1) & np.asarray(in_cuboid, bool)
        & (reprojection_a_px <= max_reprojection_error_px) & (reprojection_b_px <= max_reprojection_error_px)
    )


@dataclass(frozen=True)
class Mast3rCorrespondenceOptions:
    dataset_manifest: Path
    mast3r_root: Path
    mast3r_weights: Path
    output: Path
    track_id: int
    camera_a: str = "camera_cross_right_120fov"
    camera_b: str = "camera_front_wide_120fov"
    reference_start: int | None = None
    reference_end: int | None = None
    reference_stride: int = 1
    max_references: int = 3
    patch_size: int = 512
    patch_fov_deg: float = 35.0
    descriptor_subsample: int = 4
    minimum_confidence: float = .05
    min_ray_angle_deg: float = 1.0
    max_ray_separation_m: float = .35
    max_reprojection_error_px: float = 3.0
    cuboid_margin_m: float = .15
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.max_references <= 0 or self.patch_size < 128 or self.descriptor_subsample <= 0 or self.reference_stride <= 0:
            raise ValueError("track/reference/patch/subsample values are invalid")
        if self.output.suffix != ".json" or not 1 < self.patch_fov_deg < 170:
            raise ValueError("output and patch FOV are invalid")
        if not 0 <= self.minimum_confidence <= 1 or self.min_ray_angle_deg <= 0 or self.max_ray_separation_m <= 0 or self.max_reprojection_error_px <= 0 or self.cuboid_margin_m < 0:
            raise ValueError("correspondence thresholds are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")


def build_mast3r_correspondence(options: Mast3rCorrespondenceOptions) -> Path:
    """Build an auditable local correspondence asset; never downloads or trains."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {options.output}")
    manifest_path = options.dataset_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset_manifest must be a complete dynamic reconstruction dataset")
    tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("MASt3R correspondence currently supports a listed rigid actor")
    dataset_root = manifest_path.parent
    paired: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if options.reference_start is not None and reference < options.reference_start:
            continue
        if options.reference_end is not None and reference >= options.reference_end:
            continue
        phase = 0 if options.reference_start is None else options.reference_start
        if (reference - phase) % options.reference_stride:
            continue
        by_camera = {str(value["camera_id"]): value for value in frame["cameras"]}
        a, b = by_camera.get(options.camera_a), by_camera.get(options.camera_b)
        if a is not None and b is not None and _instance(a, options.track_id) is not None and _instance(b, options.track_id) is not None:
            paired.append((reference, a, b))
    paired = paired[:options.max_references]
    if not paired:
        raise RuntimeError("no paired source masks in the requested camera/reference window")

    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    from ncore.impl.sensors.camera import FThetaCameraModel
    models: dict[str, Any] = {}
    model = _load_mast3r(options.mast3r_root.resolve(), options.mast3r_weights.resolve(), options.device)
    accepted_points: list[np.ndarray] = []
    accepted_reference: list[np.ndarray] = []
    accepted_confidence: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for ordinal, (reference, record_a, record_b) in enumerate(paired, start=1):
        camera_a, camera_b = str(record_a["camera_id"]), str(record_b["camera_id"])
        sensor_a, sensor_b = loader.get_camera_sensor(camera_a), loader.get_camera_sensor(camera_b)
        native_a = models.setdefault(camera_a, FThetaCameraModel(sensor_a.model_parameters, device=options.device))
        native_b = models.setdefault(camera_b, FThetaCameraModel(sensor_b.model_parameters, device=options.device))
        image_a = np.asarray(sensor_a.get_frame_image_array(int(record_a["source_frame_index"])), np.uint8)
        image_b = np.asarray(sensor_b.get_frame_image_array(int(record_b["source_frame_index"])), np.uint8)
        instance_a, instance_b = _instance(record_a, options.track_id), _instance(record_b, options.track_id)
        assert instance_a is not None and instance_b is not None
        mask_a, mask_b = _load_mask(dataset_root, instance_a, image_a.shape[:2]), _load_mask(dataset_root, instance_b, image_b.shape[:2])
        centre_a = np.asarray(np.nonzero(mask_a)[::-1], np.float32).mean(axis=1)
        centre_b = np.asarray(np.nonzero(mask_b)[::-1], np.float32).mean(axis=1)
        patch_a, map_a, valid_a = _rectify_tangent_patch(native_a, image_a, centre_a, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
        patch_b, map_b, valid_b = _rectify_tangent_patch(native_b, image_b, centre_b, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
        xy_a, xy_b, confidence = _mast3r_matches(model, patch_a, patch_b, device=options.device, subsample=options.descriptor_subsample)
        raw_a, mapped_a = sample_raw_pixel_map(map_a, xy_a); raw_b, mapped_b = sample_raw_pixel_map(map_b, xy_b)
        # ``sample_raw_pixel_map`` deliberately marks out-of-patch maps NaN.
        # Avoid casting those sentinels to integer in the strict mask lookup.
        safe_a, safe_b = np.nan_to_num(raw_a, nan=-1.0), np.nan_to_num(raw_b, nan=-1.0)
        candidate = mapped_a & mapped_b & pixels_in_mask(mask_a, safe_a) & pixels_in_mask(mask_b, safe_b) & (confidence >= options.minimum_confidence)
        source_a, source_b, source_confidence = raw_a[candidate], raw_b[candidate], confidence[candidate]
        if len(source_a):
            origins_a, rays_a = _native_world_rays(native_a, source_a, record_a)
            origins_b, rays_b = _native_world_rays(native_b, source_b, record_b)
            points, separation, angle, distances = triangulate_rays(origins_a, rays_a, origins_b, rays_b)
            timestamp = (int(record_a["timestamp_midpoint_us"]) + int(record_b["timestamp_midpoint_us"])) // 2
            pose = _interpolate_track_pose(track, timestamp)
            if pose is None:
                raise RuntimeError(f"track {options.track_id} has no pose at paired observation {reference}")
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            half = np.asarray(track["length_width_height"], np.float32) * .5 + options.cuboid_margin_m
            in_cuboid = np.all(np.abs(local) <= half[None], axis=1)
            error_a = _native_reprojection_error(native_a, points, record_a, source_a)
            error_b = _native_reprojection_error(native_b, points, record_b, source_b)
            accepted = correspondence_acceptance(
                angle_deg=angle, separation_m=separation, distances_m=distances, in_cuboid=in_cuboid,
                reprojection_a_px=error_a, reprojection_b_px=error_b, min_ray_angle_deg=options.min_ray_angle_deg,
                max_ray_separation_m=options.max_ray_separation_m, max_reprojection_error_px=options.max_reprojection_error_px,
            )
            lidar_distance = _lidar_distances(loader, track, record_a, points)
            accepted_points.append(local[accepted].astype(np.float32))
            accepted_reference.append(np.full(int(accepted.sum()), reference, np.int32))
            accepted_confidence.append(source_confidence[accepted].astype(np.float32))
        else:
            points = np.empty((0, 3), np.float32); separation = angle = error_a = error_b = lidar_distance = np.empty(0, np.float32); in_cuboid = accepted = np.empty(0, bool)
        report = {
            "reference_frame_index": reference, "camera_ids": [camera_a, camera_b],
            "source_frame_indices": [int(record_a["source_frame_index"]), int(record_b["source_frame_index"])],
            "mask_pixels": [int(mask_a.sum()), int(mask_b.sum())],
            "patch_valid_fraction": [float(valid_a.mean()), float(valid_b.mean())],
            "mast3r_candidates": int(len(confidence)), "inside_source_masks": int(candidate.sum()),
            "accepted": int(accepted.sum()), "cuboid_kept": int(in_cuboid.sum()),
            "ray_angle_median_deg": None if not len(angle) else float(np.median(angle)),
            "ray_separation_median_m": None if not len(separation) else float(np.median(separation)),
            "lidar_distance_median_m": None if not np.isfinite(lidar_distance).any() else float(np.nanmedian(lidar_distance)),
        }
        reports.append(report)
        print(f"MASt3R FTheta correspondence [{ordinal}/{len(paired)}] ref={reference} {camera_a}->{camera_b}: candidates={len(confidence)} masks={int(candidate.sum())} accepted={int(accepted.sum())}", flush=True)

    points = np.concatenate(accepted_points, axis=0) if accepted_points else np.empty((0, 3), np.float32)
    references = np.concatenate(accepted_reference) if accepted_reference else np.empty(0, np.int32)
    confidence = np.concatenate(accepted_confidence) if accepted_confidence else np.empty(0, np.float32)
    asset_path = options.output.with_suffix(".npz")
    _atomic_npz(asset_path, points_actor_local=points, reference_frame_indices=references, confidence=confidence)
    result = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path), "asset": str(asset_path.resolve()), "track_id": options.track_id,
        "camera_ids": [options.camera_a, options.camera_b], "mast3r": {"root": str(options.mast3r_root.resolve()), "weights": str(options.mast3r_weights.resolve()), "local_files_only": True},
        "geometry": {"camera_model": "native NCore FTheta + rolling shutter", "patch_size": options.patch_size, "patch_fov_deg": options.patch_fov_deg, "thresholds": {"minimum_confidence": options.minimum_confidence, "min_ray_angle_deg": options.min_ray_angle_deg, "max_ray_separation_m": options.max_ray_separation_m, "max_reprojection_error_px": options.max_reprojection_error_px, "cuboid_margin_m": options.cuboid_margin_m}},
        "summary": {"references": len(reports), "accepted_total": int(len(points)), "references_with_accepted": int(sum(item["accepted"] > 0 for item in reports))}, "reports": reports,
        "limitations": ["MASt3R only proposes descriptors; all geometry is re-established with NCore FTheta/rolling-shutter rays.", "No static PLY, legacy dynamic Gaussian, SAM2-only mask, RGB pseudo label or generated surface is used.", "The point asset constrains only observed local surfaces; all other actor volume remains unknown until validated by ray supervision.", "A zero-point asset is a failed correspondence experiment and the 4D trainer must refuse it."],
    }
    _atomic_json(options.output, result)
    print(f"Wrote MASt3R native correspondence asset: {options.output} (accepted={len(points)})", flush=True)
    return options.output
