"""RoMa multi-view correspondence audit with native NCore FTheta geometry.

This is intentionally an evidence-gathering stage, not an actor trainer.  It
rectifies only a small tangent-plane patch around an already-trusted dynamic
instance, asks locally installed RoMa for dense candidate correspondences, and
then validates every candidate using NCore's rolling-shutter FTheta rays.  No
static PLY, Gaussian asset, or mask is changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .camera import invert_pose
from .dynamic_mask_crossview_audit import project_world_points_native_rolling
from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-roma-ftheta-multiview-audit"
VERSION = 1
TRUSTED_SCHEMA = "ncore-dynamic-layer-trusted-supervision"


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


def normalize_rows(value: np.ndarray) -> np.ndarray:
    """Normalize an ``[...,3]`` vector array while retaining finite zeros."""
    array = np.asarray(value, np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def tangent_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right/down/forward OpenCV-style tangent-frame axes.

    The construction deliberately fixes the local ``down`` direction near the
    native camera down axis.  It avoids roll ambiguity while remaining stable
    for an actor away from the extreme fisheye rim.
    """
    fwd = normalize_rows(np.asarray(forward, np.float32).reshape(1, 3))[0]
    reference_down = np.asarray((0.0, 1.0, 0.0), np.float32)
    if abs(float(np.dot(fwd, reference_down))) > .96:
        reference_down = np.asarray((0.0, 0.0, 1.0), np.float32)
    right = normalize_rows(np.cross(reference_down, fwd).reshape(1, 3))[0]
    down = normalize_rows(np.cross(fwd, right).reshape(1, 3))[0]
    return right, down, fwd


def tangent_patch_rays(size: int, horizontal_fov_deg: float, forward: np.ndarray) -> np.ndarray:
    """Generate a square virtual-pinhole patch in *native camera* directions."""
    if size <= 1 or not 1.0 < horizontal_fov_deg < 170.0:
        raise ValueError("patch size/FOV are invalid")
    right, down, fwd = tangent_basis(forward)
    axis = (np.arange(size, dtype=np.float32) + .5) / size * 2.0 - 1.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    scale = float(np.tan(np.deg2rad(horizontal_fov_deg) * .5))
    rays = fwd[None, None] + xx[..., None] * scale * right[None, None] + yy[..., None] * scale * down[None, None]
    return normalize_rows(rays)


def triangulate_rays(origins_a: np.ndarray, directions_a: np.ndarray, origins_b: np.ndarray, directions_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closest-point triangulation for paired world rays.

    Returns midpoint, closest-line distance, ray angle (degrees), and the two
    positive ray distances.  Degenerate near-parallel pairs stay finite but
    receive a zero-distance ray parameter and must be rejected by the caller.
    """
    oa, da = np.asarray(origins_a, np.float32), normalize_rows(directions_a)
    ob, db = np.asarray(origins_b, np.float32), normalize_rows(directions_b)
    if oa.shape != da.shape or ob.shape != db.shape or oa.shape != ob.shape or oa.ndim != 2 or oa.shape[1] != 3:
        raise ValueError("origins/directions must have matching [N,3] shapes")
    offset = ob - oa
    dot = np.einsum("ij,ij->i", da, db)
    det = 1.0 - dot * dot
    rhs_a = np.einsum("ij,ij->i", offset, da)
    rhs_b = np.einsum("ij,ij->i", offset, db)
    valid = det > 1e-7
    ta = np.zeros(len(oa), np.float32); tb = np.zeros(len(oa), np.float32)
    ta[valid] = (rhs_a[valid] - dot[valid] * rhs_b[valid]) / det[valid]
    tb[valid] = (dot[valid] * rhs_a[valid] - rhs_b[valid]) / det[valid]
    pa, pb = oa + ta[:, None] * da, ob + tb[:, None] * db
    midpoint = (pa + pb) * .5
    separation = np.linalg.norm(pa - pb, axis=1)
    angle = np.rad2deg(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
    return midpoint.astype(np.float32), separation.astype(np.float32), angle.astype(np.float32), np.stack((ta, tb), axis=1)


def pixels_in_mask(mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Strict nearest-pixel instance-mask membership for ``[N,2]`` pixels."""
    value = np.asarray(mask, bool); xy = np.asarray(pixels, np.float32)
    if value.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("mask/pixels shapes are invalid")
    x, y = np.rint(xy[:, 0]).astype(np.int64), np.rint(xy[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < value.shape[1]) & (y >= 0) & (y < value.shape[0])
    result = np.zeros(len(x), bool)
    result[inside] = value[y[inside], x[inside]]
    return result


def sample_raw_pixel_map(raw_pixels: np.ndarray, patch_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest sample a tangent-patch-to-native-pixel map at RoMa coordinates."""
    mapping = np.asarray(raw_pixels, np.float32); xy = np.asarray(patch_pixels, np.float32)
    if mapping.ndim != 3 or mapping.shape[2] != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("mapping must be [H,W,2] and coordinates [N,2]")
    x, y = np.rint(xy[:, 0]).astype(np.int64), np.rint(xy[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < mapping.shape[1]) & (y >= 0) & (y < mapping.shape[0])
    result = np.full((len(x), 2), np.nan, np.float32)
    result[valid] = mapping[y[valid], x[valid]]
    valid &= np.isfinite(result).all(axis=1)
    return result, valid


def _rectify_tangent_patch(model: Any, image: np.ndarray, centre_native_xy: np.ndarray, *, size: int, fov_deg: float, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FTheta-inverse-project a virtual tangent patch and retain native pixel map."""
    import torch
    import torch.nn.functional as functional

    # ``pixels_to_camera_rays`` is intentionally a pixel-index API in NCore;
    # use the nearest native source pixel for patch orientation.  RoMa matches
    # themselves remain sub-pixel and are later unprojected through the image-
    # point (not pixel-index) rolling-shutter API.
    centre = np.rint(np.asarray(centre_native_xy, np.float32)).astype(np.int64).reshape(1, 2)
    centre_ray = model.pixels_to_camera_rays(torch.from_numpy(centre).to(device)).detach().float().cpu().numpy()[0]
    rays = tangent_patch_rays(size, fov_deg, centre_ray).reshape(-1, 3)
    returned = model.camera_rays_to_pixels(torch.from_numpy(rays).to(device))
    raw = returned.pixels.reshape(size, size, 2).detach().float()
    valid = returned.valid_flag.reshape(size, size).detach().bool()
    height, width = image.shape[:2]
    grid = raw.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    valid &= torch.isfinite(grid).all(dim=-1) & (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    source = torch.from_numpy(np.array(image, copy=True)).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    sampled = functional.grid_sample(source, grid[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    sampled[:, ~valid] = 0.0
    raw_np = raw.cpu().numpy().astype(np.float32); raw_np[~valid.cpu().numpy()] = np.nan
    patch = sampled.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().cpu().numpy()
    return patch, raw_np, valid.byte().cpu().numpy().astype(bool)


def _load_roma(roma_root: Path, roma_weights: Path, dinov2_weights: Path, device: str) -> Any:
    """Load RoMa exclusively from already-present source and local raw weights."""
    if not (roma_root / "romatch").is_dir():
        raise FileNotFoundError(f"RoMa source package missing under: {roma_root}")
    if not roma_weights.is_file() or not dinov2_weights.is_file():
        raise FileNotFoundError("RoMa or DINOv2 local weight file is missing")
    if str(roma_root) not in sys.path:
        sys.path.insert(0, str(roma_root))
    try:
        import torch
        from romatch.models.model_zoo import roma_outdoor
    except ImportError as exc:  # pragma: no cover - external optional dependency
        raise RuntimeError("RoMa import failed; use PYTHONPATH=<repo>/external/roma and install its local dependencies") from exc
    torch.set_float32_matmul_precision("highest")
    try:
        weights = torch.load(roma_weights, map_location="cpu", weights_only=True)
        dino = torch.load(dinov2_weights, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("could not read local RoMa/DINOv2 raw state dictionaries") from exc
    model = roma_outdoor(device=device, weights=weights, dinov2_weights=dino, use_custom_corr=False)
    return model.eval()


def _roma_matches(model: Any, patch_a: np.ndarray, patch_b: np.ndarray, device: str, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source pixels, target pixels and RoMa certainty, without a download."""
    warp, certainty = model.match(Image.fromarray(patch_a), Image.fromarray(patch_b), device=device)
    matches, sampled_certainty = model.sample(warp, certainty, num=maximum)
    source, target = model.to_pixel_coordinates(matches, patch_a.shape[0], patch_a.shape[1], patch_b.shape[0], patch_b.shape[1])
    return (source.detach().float().cpu().numpy(), target.detach().float().cpu().numpy(),
            sampled_certainty.detach().float().cpu().numpy().reshape(-1))


def _instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((item for item in record["instances"] if int(item["track_id"]) == track_id and item.get("mask_origin") == "source"), None)


def _mask(roots: dict[str, Path], instance: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(roots[str(instance["mask_root"])] / str(instance["mask"])).convert("L"), np.uint8) > 0
    if value.shape != shape:
        raise ValueError("source mask/image shape mismatch")
    return value


def _native_world_rays(model: Any, pixels: np.ndarray, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """NCore's own exact FTheta + row/column rolling-shutter unprojection."""
    returned = model.image_points_to_world_rays_shutter_pose(
        np.asarray(pixels, np.float32), np.asarray(record["camera_to_world_start"], np.float64),
        np.asarray(record["camera_to_world_end"], np.float64), int(record["timestamp_start_us"]), int(record["timestamp_end_us"]),
    )
    rays = returned.world_rays.detach().float().cpu().numpy()
    return rays[:, :3].astype(np.float32), normalize_rows(rays[:, 3:]).astype(np.float32)


def _native_reprojection_error(model: Any, points: np.ndarray, record: dict[str, Any], expected: np.ndarray) -> np.ndarray:
    """Exact iterative rolling-shutter reprojection error, NaN when invalid."""
    projected = model.world_points_to_pixels_shutter_pose(
        np.asarray(points, np.float32), invert_pose(np.asarray(record["camera_to_world_start"], np.float64)),
        invert_pose(np.asarray(record["camera_to_world_end"], np.float64)), int(record["timestamp_start_us"]), int(record["timestamp_end_us"]),
        return_valid_indices=True,
    )
    errors = np.full(len(points), np.nan, np.float32)
    indices = projected.valid_indices.detach().cpu().numpy().astype(np.int64)
    pixels = projected.pixels.detach().float().cpu().numpy()
    errors[indices] = np.linalg.norm(pixels - np.asarray(expected, np.float32)[indices], axis=1)
    return errors


def _lidar_distances(loader: Any, track: dict[str, Any], record: dict[str, Any], points: np.ndarray) -> np.ndarray:
    """Distance to raw actor-cuboid LiDAR at the observation time (report-only)."""
    lidar_index = record.get("lidar_frame_index")
    lidar_time = record.get("lidar_timestamp_us")
    if lidar_index is None or lidar_time is None or not len(points):
        return np.full(len(points), np.nan, np.float32)
    lidar = loader.get_lidar_sensor("lidar_top_360fov")
    cloud = lidar.get_frame_point_cloud(int(lidar_index), False, False).xyz_m_end
    transform = np.asarray(lidar.get_frames_T_sensor_target("world", int(lidar_index)), np.float64)
    world = np.asarray(cloud, np.float32) @ transform[:3, :3].T + transform[:3, 3]
    pose = _interpolate_track_pose(track, int(lidar_time))
    if pose is None:
        return np.full(len(points), np.nan, np.float32)
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    half = np.asarray(track["length_width_height"], np.float32) * .5 + .15
    actor = world[np.all(np.abs(local) <= half[None], axis=1)]
    if not len(actor):
        return np.full(len(points), np.nan, np.float32)
    distances = np.linalg.norm(points[:, None] - actor[None], axis=-1)
    return distances.min(axis=1).astype(np.float32)


@dataclass(frozen=True)
class RomaMultiviewAuditOptions:
    trusted_manifest: Path
    roma_root: Path
    roma_weights: Path
    dinov2_weights: Path
    output: Path
    track_id: int
    camera_a: str = "camera_front_tele_30fov"
    camera_b: str = "camera_front_wide_120fov"
    reference_start: int | None = None
    reference_end: int | None = None
    max_references: int = 3
    patch_size: int = 512
    # The first verified pair contains the 30-degree front tele camera.  A
    # 20-degree tangent patch retains 82% of its native valid area (whereas a
    # 70-degree patch is almost entirely outside that sensor), while fitting
    # the same physical directions in the front-wide image.
    patch_fov_deg: float = 20.0
    max_matches: int = 512
    # RoMa's own outdoor sampler uses a 0.05 confidence floor.  Keep the same
    # documented floor for a feasibility audit and report the score
    # distribution; selecting a higher threshold is a later evidence-based
    # decision, not a way to manufacture a passing result.
    minimum_certainty: float = .05
    min_ray_angle_deg: float = 1.0
    max_ray_separation_m: float = .35
    max_reprojection_error_px: float = 3.0
    cuboid_margin_m: float = .15
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.max_references <= 0 or self.patch_size < 64 or self.max_matches <= 0:
            raise ValueError("track/reference/patch/match values are invalid")
        if not 1.0 < self.patch_fov_deg < 170.0 or not 0.0 <= self.minimum_certainty <= 1.0:
            raise ValueError("patch FOV or certainty are invalid")
        if self.min_ray_angle_deg <= 0.0 or self.max_ray_separation_m <= 0.0 or self.max_reprojection_error_px <= 0.0 or self.cuboid_margin_m < 0.0:
            raise ValueError("geometry thresholds are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")
        if self.output.suffix != ".json":
            raise ValueError("output must be a .json report")


def audit_roma_multiview(options: RomaMultiviewAuditOptions) -> Path:
    """Run a small, evidence-only RoMa cross-camera FTheta correspondence audit."""
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {options.output}")
    manifest_path = options.trusted_manifest.resolve(); manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRUSTED_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("trusted manifest must be complete raw dynamic supervision")
    tracks = {int(item["track_id"]): item for item in manifest["tracks"]}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("requested track must be a trusted rigid actor")
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if options.reference_start is not None and reference < options.reference_start:
            continue
        if options.reference_end is not None and reference >= options.reference_end:
            continue
        records = {str(item["camera_id"]): item for item in frame["cameras"]}
        a, b = records.get(options.camera_a), records.get(options.camera_b)
        if a is not None and b is not None and _instance(a, options.track_id) is not None and _instance(b, options.track_id) is not None:
            candidates.append((reference, a, b))
    candidates = candidates[:options.max_references]
    if not candidates:
        raise RuntimeError("no paired trusted source masks for the requested camera/reference window")
    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    from ncore.impl.sensors.camera import FThetaCameraModel
    models: dict[str, Any] = {}
    roma = _load_roma(options.roma_root.resolve(), options.roma_weights.resolve(), options.dinov2_weights.resolve(), options.device)
    reports: list[dict[str, Any]] = []
    for position, (reference, record_a, record_b) in enumerate(candidates, start=1):
        camera_a, camera_b = str(record_a["camera_id"]), str(record_b["camera_id"])
        sensor_a, sensor_b = loader.get_camera_sensor(camera_a), loader.get_camera_sensor(camera_b)
        model_a = models.setdefault(camera_a, FThetaCameraModel(sensor_a.model_parameters, device=options.device))
        model_b = models.setdefault(camera_b, FThetaCameraModel(sensor_b.model_parameters, device=options.device))
        image_a = np.asarray(sensor_a.get_frame_image_array(int(record_a["source_frame_index"])), np.uint8)
        image_b = np.asarray(sensor_b.get_frame_image_array(int(record_b["source_frame_index"])), np.uint8)
        instance_a, instance_b = _instance(record_a, options.track_id), _instance(record_b, options.track_id)
        assert instance_a is not None and instance_b is not None
        mask_a, mask_b = _mask(roots, instance_a, image_a.shape[:2]), _mask(roots, instance_b, image_b.shape[:2])
        centre_a = np.asarray(np.nonzero(mask_a)[::-1], np.float32).mean(axis=1)
        centre_b = np.asarray(np.nonzero(mask_b)[::-1], np.float32).mean(axis=1)
        patch_a, map_a, valid_a = _rectify_tangent_patch(model_a, image_a, centre_a, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
        patch_b, map_b, valid_b = _rectify_tangent_patch(model_b, image_b, centre_b, size=options.patch_size, fov_deg=options.patch_fov_deg, device=options.device)
        coords_a, coords_b, certainty = _roma_matches(roma, patch_a, patch_b, options.device, options.max_matches)
        raw_a, in_patch_a = sample_raw_pixel_map(map_a, coords_a); raw_b, in_patch_b = sample_raw_pixel_map(map_b, coords_b)
        native_mask = in_patch_a & in_patch_b & pixels_in_mask(mask_a, raw_a) & pixels_in_mask(mask_b, raw_b) & (certainty >= options.minimum_certainty)
        selected_a, selected_b, selected_certainty = raw_a[native_mask], raw_b[native_mask], certainty[native_mask]
        if len(selected_a):
            origins_a, rays_a = _native_world_rays(model_a, selected_a, record_a); origins_b, rays_b = _native_world_rays(model_b, selected_b, record_b)
            points, separation, angle, distances = triangulate_rays(origins_a, rays_a, origins_b, rays_b)
            pose = _interpolate_track_pose(track, int((int(record_a["timestamp_midpoint_us"]) + int(record_b["timestamp_midpoint_us"])) // 2))
            local = (points - pose[:3, 3]) @ pose[:3, :3] if pose is not None else np.full_like(points, np.nan)
            half = np.asarray(track["length_width_height"], np.float32) * .5 + options.cuboid_margin_m
            in_cuboid = np.all(np.abs(local) <= half[None], axis=1)
            error_a = _native_reprojection_error(model_a, points, record_a, selected_a)
            error_b = _native_reprojection_error(model_b, points, record_b, selected_b)
            lidar_distance = _lidar_distances(loader, track, record_a, points)
            accepted = (angle >= options.min_ray_angle_deg) & (separation <= options.max_ray_separation_m) & np.all(distances > .1, axis=1) & in_cuboid & np.isfinite(error_a) & np.isfinite(error_b) & (error_a <= options.max_reprojection_error_px) & (error_b <= options.max_reprojection_error_px)
        else:
            points = np.empty((0, 3), np.float32); separation = angle = error_a = error_b = lidar_distance = np.empty(0, np.float32); in_cuboid = accepted = np.empty(0, bool)
        report = {
            "reference_frame_index": reference, "camera_ids": [camera_a, camera_b],
            "source_frame_indices": [int(record_a["source_frame_index"]), int(record_b["source_frame_index"])],
            "midpoint_timestamp_delta_us": abs(int(record_a["timestamp_midpoint_us"]) - int(record_b["timestamp_midpoint_us"])),
            "mask_pixels": [int(mask_a.sum()), int(mask_b.sum())], "patch_valid_fraction": [float(valid_a.mean()), float(valid_b.mean())],
            "roma_candidates": int(len(certainty)), "inside_both_trusted_masks": int((in_patch_a & in_patch_b & pixels_in_mask(mask_a, raw_a) & pixels_in_mask(mask_b, raw_b)).sum()),
            "certainty_kept": int(len(selected_a)), "accepted": int(accepted.sum()),
        }
        if len(selected_a):
            report.update({
                "certainty": {"median": float(np.median(selected_certainty)), "p05": float(np.percentile(selected_certainty, 5))},
                "ray_angle_deg": {"median": float(np.median(angle)), "p05": float(np.percentile(angle, 5))},
                "ray_separation_m": {"median": float(np.median(separation)), "p95": float(np.percentile(separation, 95))},
                "reprojection_error_px": {"a_median": float(np.nanmedian(error_a)), "b_median": float(np.nanmedian(error_b))},
                "cuboid_kept": int(in_cuboid.sum()), "lidar_distance_m": {"finite_count": int(np.isfinite(lidar_distance).sum()), "median": float(np.nanmedian(lidar_distance)) if np.isfinite(lidar_distance).any() else None},
                "accepted_points_actor_local": local[accepted].round(5).tolist(),
            })
        reports.append(report)
        print(f"RoMa FTheta audit [{position}/{len(candidates)}] ref={reference} {camera_a}->{camera_b}: candidates={len(certainty)} masks={len(selected_a)} accepted={int(accepted.sum())}", flush=True)
    accepted_total = int(sum(item["accepted"] for item in reports))
    result = {"schema": SCHEMA, "version": VERSION, "status": "complete", "trusted_manifest": str(manifest_path),
              "track_id": options.track_id, "camera_ids": [options.camera_a, options.camera_b], "references": [item["reference_frame_index"] for item in reports],
              "roma": {"source": str(options.roma_root.resolve()), "weights": str(options.roma_weights.resolve()), "dinov2_weights": str(options.dinov2_weights.resolve()), "local_files_only": True},
              "geometry": {"camera_model": "NCore native FTheta + rolling-shutter", "patch": {"size": options.patch_size, "horizontal_fov_deg": options.patch_fov_deg}, "thresholds": {"minimum_certainty": options.minimum_certainty, "min_ray_angle_deg": options.min_ray_angle_deg, "max_ray_separation_m": options.max_ray_separation_m, "max_reprojection_error_px": options.max_reprojection_error_px, "cuboid_margin_m": options.cuboid_margin_m}},
              "summary": {"accepted_total": accepted_total, "references_with_accepted": int(sum(item["accepted"] > 0 for item in reports))}, "reports": reports,
              "limitations": ["This is a read-only feasibility audit, not actor training or Gaussian generation.", "Only pre-existing trusted source masks gate matches; SAM2-only additions and static PLY data are excluded.", "LiDAR distance is diagnostic only because sparse returns cannot reject an otherwise coherent unobserved surface.", "A passing smoke still requires a separate temporal/cross-camera holdout before any dynamic reconstruction is trained."]}
    _atomic_json(options.output, result)
    print(f"Wrote RoMa exact-FTheta multiview audit: {options.output} (accepted={accepted_total})", flush=True)
    return options.output
