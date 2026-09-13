"""Build an actor-centred, canonical pinhole proxy dataset for official 2DGS.

Unlike ``two_dgs_dataset``, this module never asks one static 2DGS field to
explain a vehicle at many world positions.  Every rigid instance observation
is transformed into its NCore cuboid-local frame, where the actor is static.
The official 2DGS implementation remains an approximate pinhole/global-shutter
trainer; its result is an object-layer experiment, not a native NCore asset.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader
from .street_dataset import _interpolate_track_pose, pinhole_intrinsics, tile_to_source_pose
from .two_dgs_dataset import _rectify_binary_mask, _rectify_frame, _rotation_to_colmap_quaternion


SCHEMA = "ncore-2dgs-canonical-rigid-actor-proxy"
VERSION = 1


@dataclass(frozen=True)
class TwoDgsActorDatasetOptions:
    ncore_path: Path
    dynamic_dataset_manifest: Path
    output: Path
    track_id: int
    camera_ids: tuple[str, ...] = ()
    holdout_camera_ids: tuple[str, ...] = ()
    reference_start: int | None = None
    reference_end: int | None = None
    width: int = 480
    height: int = 270
    horizontal_fov_deg: float = 45.0
    minimum_mask_pixels: int = 1_000
    holdout_every: int = 6
    initial_surface_spacing_m: float = .35
    max_observations: int = 120
    rectify_device: str = "cuda"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.width <= 0 or self.height <= 0 or self.minimum_mask_pixels <= 0:
            raise ValueError("track, dimensions, and minimum mask pixels are invalid")
        if not 20.0 <= self.horizontal_fov_deg < 120.0 or self.holdout_every < 2:
            raise ValueError("actor proxy FOV or holdout period is invalid")
        if self.initial_surface_spacing_m <= 0.0 or self.max_observations <= 1:
            raise ValueError("surface spacing and max observations are invalid")
        if self.reference_start is not None and self.reference_start < 0:
            raise ValueError("reference_start must be non-negative")
        if self.reference_end is not None and self.reference_start is not None and self.reference_end <= self.reference_start:
            raise ValueError("reference_end must exceed reference_start")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        if len(set(self.holdout_camera_ids)) != len(self.holdout_camera_ids):
            raise ValueError("holdout_camera_ids must be unique")


def actor_local_camera_pose(actor_to_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Return ``T_camera_actor`` for a rigid canonical actor coordinate frame."""
    actor = np.asarray(actor_to_world, dtype=np.float64)
    camera = np.asarray(camera_to_world, dtype=np.float64)
    if actor.shape != (4, 4) or camera.shape != (4, 4):
        raise ValueError("actor and camera poses must be 4x4")
    return np.linalg.inv(actor) @ camera


def actor_centre_tile_angles(actor_to_world: np.ndarray, source_camera_to_world: np.ndarray) -> tuple[float, float]:
    """Choose a virtual pinhole tile whose optical axis points at the actor centre."""
    actor = np.asarray(actor_to_world, dtype=np.float64)
    camera = np.asarray(source_camera_to_world, dtype=np.float64)
    direction = camera[:3, :3].T @ (actor[:3, 3] - camera[:3, 3])
    horizontal = float(np.hypot(direction[0], direction[2]))
    if not np.isfinite(direction).all() or direction[2] <= .10 or horizontal <= 1e-6:
        raise ValueError("actor is behind or coincident with source camera")
    yaw = float(np.degrees(np.arctan2(direction[0], direction[2])))
    # In OpenCV, positive camera Y points down.  ``tile_to_source_pose`` uses
    # negative pitch to turn the optical axis toward positive source Y.
    pitch = float(-np.degrees(np.arctan2(direction[1], horizontal)))
    return yaw, pitch


def cuboid_surface_points(dimensions: np.ndarray, spacing_m: float) -> np.ndarray:
    """Return a deterministic local cuboid shell used only as a weak 2DGS seed."""
    size = np.asarray(dimensions, dtype=np.float32).reshape(3)
    if np.any(size <= 0.0) or spacing_m <= 0.0:
        raise ValueError("cuboid dimensions and spacing must be positive")
    axes = [np.linspace(-side * .5, side * .5, max(2, int(np.ceil(side / spacing_m)) + 1), dtype=np.float32) for side in size]
    points: list[np.ndarray] = []
    for fixed_axis in range(3):
        free = [axis for axis in range(3) if axis != fixed_axis]
        aa, bb = np.meshgrid(axes[free[0]], axes[free[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((aa.size, 3), np.float32)
            face[:, fixed_axis] = sign * size[fixed_axis] * .5
            face[:, free[0]], face[:, free[1]] = aa.reshape(-1), bb.reshape(-1)
            points.append(face)
    quantized = np.unique(np.round(np.concatenate(points) / 1e-5).astype(np.int64), axis=0)
    return (quantized.astype(np.float32) * 1e-5).astype(np.float32)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _midpoint_pose(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    first, second = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("camera endpoint poses must be 4x4")
    output = np.eye(4, dtype=np.float64)
    left, _, right = np.linalg.svd(.5 * (first[:3, :3] + second[:3, :3]))
    output[:3, :3] = left @ right
    if np.linalg.det(output[:3, :3]) < 0.0:
        left[:, -1] *= -1.0
        output[:3, :3] = left @ right
    output[:3, 3] = .5 * (first[:3, 3] + second[:3, 3])
    return output


def _write_initial_ply(path: Path, dimensions: np.ndarray, spacing_m: float) -> int:
    from plyfile import PlyData, PlyElement
    points = cuboid_surface_points(dimensions, spacing_m)
    vertices = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["red"], vertices["green"], vertices["blue"] = 128, 128, 128
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    return int(len(points))


def build_two_dgs_actor_dataset(options: TwoDgsActorDatasetOptions) -> Path:
    """Build actor-masked canonical images and COLMAP poses from raw NCore observations."""
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty actor dataset: {output}")
    dataset = _read_json(options.dynamic_dataset_manifest.resolve())
    if dataset.get("schema") != "ncore-dynamic-reconstruction-dataset" or dataset.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be a complete raw dynamic reconstruction dataset")
    if Path(str(dataset.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic_dataset_manifest NCore path does not match --ncore-path")
    tracks = {int(track["track_id"]): track for track in dataset.get("tracks", []) if isinstance(track, dict) and "track_id" in track}
    track = tracks.get(options.track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("canonical 2DGS currently supports only one rigid track from the dynamic dataset")
    allowed_cameras = set(options.camera_ids) if options.camera_ids else set(dataset.get("camera_ids", []))
    if not allowed_cameras:
        raise ValueError("actor dataset has no selected source cameras")
    holdout_cameras = set(options.holdout_camera_ids)
    unknown_holdout_cameras = holdout_cameras - allowed_cameras
    if unknown_holdout_cameras:
        raise ValueError(
            "holdout_camera_ids must be selected by camera_ids: "
            + ", ".join(sorted(unknown_holdout_cameras))
        )
    output.mkdir(parents=True); images = output / "images"; sparse = output / "sparse" / "0"
    images.mkdir(); sparse.mkdir(parents=True)
    loader = _create_ncore_loader(options.ncore_path)
    intrinsic = pinhole_intrinsics(options.width, options.height, options.horizontal_fov_deg)
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for frame in dataset.get("frames", []):
        reference = int(frame.get("reference_frame_index", -1))
        if reference < 0 or (options.reference_start is not None and reference < options.reference_start) or (options.reference_end is not None and reference >= options.reference_end):
            continue
        for record in frame.get("cameras", []):
            if str(record.get("camera_id")) not in allowed_cameras:
                continue
            for instance in record.get("instances", []):
                if int(instance.get("track_id", -1)) == options.track_id and int(instance.get("mask_pixels", 0)) >= options.minimum_mask_pixels:
                    candidates.append((reference, record, instance))
    candidates.sort(key=lambda item: (item[0], str(item[1]["camera_id"])))
    if len(candidates) > options.max_observations:
        chosen = np.linspace(0, len(candidates) - 1, options.max_observations, dtype=np.int64)
        candidates = [candidates[int(index)] for index in chosen]
    if len(candidates) < 3:
        raise RuntimeError("fewer than three trusted actor observations remain after filtering")
    camera_line = f"1 PINHOLE {options.width} {options.height} {intrinsic[0,0]:.9f} {intrinsic[1,1]:.9f} {intrinsic[0,2]:.9f} {intrinsic[1,2]:.9f}"
    image_lines = ["# Image list with two lines of data per image:", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    records: list[dict[str, Any]] = []
    for image_id, (reference, record, instance) in enumerate(candidates, start=1):
        camera_id = str(record["camera_id"]); sensor = loader.get_camera_sensor(camera_id)
        timestamp = int(record["timestamp_midpoint_us"])
        actor_to_world = _interpolate_track_pose(track, timestamp)
        if actor_to_world is None:
            continue
        source_c2w = _midpoint_pose(record["camera_to_world_start"], record["camera_to_world_end"])
        # A trusted source mask may still be present at the edge of the
        # selected temporal interval after the cuboid centre has crossed
        # behind this particular camera.  Such an observation cannot define
        # an actor-centred forward tile; it is an expected visibility case,
        # not a malformed dataset.
        try:
            yaw, pitch = actor_centre_tile_angles(actor_to_world, source_c2w)
        except ValueError as error:
            print(
                f"Skipping actor 2DGS reference={reference} camera={camera_id}: {error}",
                flush=True,
            )
            continue
        if abs(yaw) >= 80.0 or abs(pitch) >= 55.0:
            continue
        raw = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), dtype=np.uint8)
        mask_path = options.dynamic_dataset_manifest.parent / str(instance["mask"])
        with Image.open(mask_path) as opened:
            source_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
        rectified, valid = _rectify_frame(sensor, raw, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch)
        actor_mask = _rectify_binary_mask(sensor, source_mask, intrinsic, width=options.width, height=options.height, device=options.rectify_device, yaw_deg=yaw, pitch_deg=pitch)
        actor_mask = ((actor_mask != 0) & (valid != 0)).astype(np.uint8)
        if int(actor_mask.sum()) < options.minimum_mask_pixels:
            continue
        filename = f"{reference:06d}_{camera_id}_track{options.track_id}.png"
        Image.fromarray(np.concatenate((rectified, actor_mask[..., None] * 255), axis=-1), mode="RGBA").save(images / filename)
        camera_to_actor = actor_local_camera_pose(actor_to_world, source_c2w @ tile_to_source_pose(yaw, pitch))
        world_to_camera = np.linalg.inv(camera_to_actor)
        qvec = _rotation_to_colmap_quaternion(world_to_camera[:3, :3])
        image_lines.append("%d %s %s %s %s %s %s %s 1 %s" % (image_id, *[f"{value:.12g}" for value in qvec], *[f"{value:.12g}" for value in world_to_camera[:3, 3]], filename)); image_lines.append("")
        # A held-out camera creates a genuine novel-view check.  In that
        # mode, keep all observations from the remaining cameras for the
        # canonical surface rather than also discarding periodic timestamps.
        split = (
            "holdout"
            if camera_id in holdout_cameras
            else ("train" if holdout_cameras else ("holdout" if reference % options.holdout_every == 0 else "train"))
        )
        records.append({"image_id": image_id, "image": f"images/{filename}", "split": split, "reference_frame": reference, "source_camera": camera_id, "source_frame": int(record["source_frame_index"]), "timestamp_us": timestamp, "tile_yaw_deg": yaw, "tile_pitch_deg": pitch, "actor_mask_pixels": int(actor_mask.sum()), "actor_fraction": float(actor_mask.mean()), "camera_to_actor": camera_to_actor.tolist()})
        print(f"Actor 2DGS [{len(records)}/{len(candidates)}] reference={reference} camera={camera_id} mask={int(actor_mask.sum())}", flush=True)
    if sum(record["split"] == "train" for record in records) < 2 or sum(record["split"] == "holdout" for record in records) < 1:
        raise RuntimeError("actor dataset needs at least two train and one holdout observation")
    (sparse / "cameras.txt").write_text("# Camera list with one line of data per camera:\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + camera_line + "\n", encoding="utf-8")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    seed_count = _write_initial_ply(sparse / "points3D.ply", np.asarray(track["length_width_height"], np.float32), options.initial_surface_spacing_m)
    manifest = {"schema": SCHEMA, "version": VERSION, "status": "complete", "ncore_path": str(options.ncore_path.resolve()), "dynamic_dataset_manifest": str(options.dynamic_dataset_manifest.resolve()), "track_id": options.track_id, "label": track.get("label"), "actor_family": "rigid", "coordinate_frame": "actor_local", "width": options.width, "height": options.height, "horizontal_fov_deg": options.horizontal_fov_deg, "holdout_policy": {"type": "camera" if holdout_cameras else "periodic_reference", "holdout_cameras": sorted(holdout_cameras), "holdout_every": None if holdout_cameras else options.holdout_every}, "initialisation": {"type": "cuboid_surface_shell", "spacing_m": options.initial_surface_spacing_m, "points": seed_count}, "records": records, "limitations": ["Images are exact-FTheta resampled but use a midpoint pinhole pose; rolling shutter is approximated.", "Alpha is an actor-only supervision mask, not a target alpha truth.", "Cuboid-shell initialization is only an optimization seed; unobserved surfaces must remain untrusted.", "This dataset must not be merged into static 2DGS training."]}
    path = output / "ncore_2dgs_actor_manifest.json"; path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote canonical actor 2DGS dataset: {path} (train={sum(r['split']=='train' for r in records)} holdout={sum(r['split']=='holdout' for r in records)})", flush=True)
    return path
