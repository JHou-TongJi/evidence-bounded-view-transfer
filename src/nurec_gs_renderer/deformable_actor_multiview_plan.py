"""Select trustworthy synchronous multi-camera windows for deformable actors.

The existing single-camera plan intentionally cannot establish geometry.  This
module is the fail-closed bridge: it uses only recorded NCore observation
metadata, actor cuboid centres and source masks to find views which observe the
same tracked person at nearly the same time with a usable physical baseline.
It writes a plan, not a reconstructed person or target-view image.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "ncore-deformable-actor-multiview-plan"
VERSION = 1


@dataclass(frozen=True)
class DeformableActorMultiviewPlanOptions:
    dynamic_dataset_manifest: Path
    output: Path
    min_mask_pixels: int = 800
    min_synchronized_frames: int = 8
    window_frames: int = 32
    max_windows_per_track: int = 2
    max_timestamp_spread_us: int = 50_000
    min_camera_baseline_m: float = .30
    min_triangulation_angle_deg: float = 5.0
    holdout_every: int = 5

    def __post_init__(self) -> None:
        if self.min_mask_pixels <= 0 or self.min_synchronized_frames < 3 or self.window_frames < 4 or self.max_windows_per_track <= 0:
            raise ValueError("invalid multiview window thresholds")
        if self.max_timestamp_spread_us <= 0 or self.min_camera_baseline_m <= 0 or not 0.0 < self.min_triangulation_angle_deg < 180.0 or self.holdout_every < 2:
            raise ValueError("invalid synchrony/geometry thresholds")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _camera_center(camera: dict[str, Any]) -> np.ndarray:
    start = np.asarray(camera["camera_to_world_start"], dtype=np.float64)
    end = np.asarray(camera["camera_to_world_end"], dtype=np.float64)
    if start.shape != (4, 4) or end.shape != (4, 4):
        raise ValueError("camera pose must be 4x4")
    return (start[:3, 3] + end[:3, 3]) * .5


def _actor_center(track: dict[str, Any], timestamp_us: int) -> np.ndarray:
    timestamps = np.asarray(track["timestamps_us"], dtype=np.int64)
    poses = np.asarray(track["actor_to_world"], dtype=np.float64)
    if timestamps.ndim != 1 or poses.shape != (len(timestamps), 4, 4) or not len(timestamps):
        raise ValueError("track pose samples are invalid")
    index = int(np.abs(timestamps - int(timestamp_us)).argmin())
    return poses[index, :3, 3]


def pair_geometry(first: dict[str, Any], second: dict[str, Any], actor_center: np.ndarray) -> tuple[float, float]:
    """Return baseline and centre-ray angle using recorded world poses."""
    c0, c1 = _camera_center(first), _camera_center(second)
    baseline = float(np.linalg.norm(c0 - c1))
    ray0, ray1 = actor_center - c0, actor_center - c1
    denominator = float(np.linalg.norm(ray0) * np.linalg.norm(ray1))
    if denominator <= 1.0e-9:
        return baseline, 0.0
    cosine = float(np.clip(np.dot(ray0, ray1) / denominator, -1.0, 1.0))
    return baseline, float(np.degrees(np.arccos(cosine)))


def _instance(camera: dict[str, Any], track_id: int, minimum_pixels: int) -> dict[str, Any] | None:
    for value in camera.get("instances", []):
        if int(value.get("track_id", -1)) == track_id and int(value.get("mask_pixels", 0)) >= minimum_pixels and value.get("mask"):
            return value
    return None


def _window_candidates(rows: list[dict[str, Any]], *, options: DeformableActorMultiviewPlanOptions) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for start_row in rows:
        start = int(start_row["reference_frame"])
        window = [row for row in rows if start <= int(row["reference_frame"]) < start + options.window_frames]
        if len(window) < options.min_synchronized_frames:
            continue
        held_out = [row for row in window if int(row["reference_frame"]) % options.holdout_every == 0]
        if not held_out or len(window) - len(held_out) < 2:
            continue
        score = float(sum(row["mask_pixels_sum"] * row["triangulation_angle_deg"] for row in window))
        selected.append({
            "reference_start": start, "reference_end": start + options.window_frames,
            "synchronized_frames": len(window), "train_references": [int(row["reference_frame"]) for row in window if row not in held_out],
            "holdout_references": [int(row["reference_frame"]) for row in held_out], "score": score,
            "mean_baseline_m": float(np.mean([row["baseline_m"] for row in window])),
            "mean_triangulation_angle_deg": float(np.mean([row["triangulation_angle_deg"] for row in window])),
            "observations": window,
        })
    return selected


def build_deformable_actor_multiview_plan(options: DeformableActorMultiviewPlanOptions) -> Path:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = options.dynamic_dataset_manifest.resolve(); dataset = _read(manifest_path)
    if dataset.get("schema") != "ncore-dynamic-reconstruction-dataset" or dataset.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be complete")
    tracks = {int(value["track_id"]): value for value in dataset.get("tracks", []) if isinstance(value, dict) and value.get("actor_family") == "deformable"}
    pair_rows: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    rejected = {"single_view": 0, "timestamp_spread": 0, "baseline": 0, "triangulation": 0}
    for frame in dataset.get("frames", []):
        reference = int(frame.get("reference_frame_index", -1)); cameras = [value for value in frame.get("cameras", []) if isinstance(value, dict)]
        if reference < 0:
            continue
        for track_id, track in tracks.items():
            visible = [(camera, _instance(camera, track_id, options.min_mask_pixels)) for camera in cameras]
            visible = [(camera, instance) for camera, instance in visible if instance is not None]
            if len(visible) < 2:
                rejected["single_view"] += 1
                continue
            for index, (first, first_instance) in enumerate(visible):
                for second, second_instance in visible[index + 1:]:
                    timestamps = [int(first["timestamp_midpoint_us"]), int(second["timestamp_midpoint_us"])]
                    if max(timestamps) - min(timestamps) > options.max_timestamp_spread_us:
                        rejected["timestamp_spread"] += 1; continue
                    actor_center = _actor_center(track, int(round(sum(timestamps) * .5)))
                    baseline, angle = pair_geometry(first, second, actor_center)
                    if baseline < options.min_camera_baseline_m:
                        rejected["baseline"] += 1; continue
                    if angle < options.min_triangulation_angle_deg:
                        rejected["triangulation"] += 1; continue
                    camera_ids = tuple(sorted((str(first["camera_id"]), str(second["camera_id"]))))
                    observation = {
                        "reference_frame": reference, "timestamp_us": int(round(sum(timestamps) * .5)), "camera_ids": list(camera_ids),
                        "baseline_m": baseline, "triangulation_angle_deg": angle,
                        "mask_pixels_sum": int(first_instance["mask_pixels"]) + int(second_instance["mask_pixels"]),
                        "views": [{"camera_id": str(first["camera_id"]), "source_frame": int(first["source_frame_index"]), "timestamp_us": timestamps[0], "mask": str(first_instance["mask"]), "mask_pixels": int(first_instance["mask_pixels"]), "lidar_pixels": int(first_instance.get("lidar_pixels", 0))}, {"camera_id": str(second["camera_id"]), "source_frame": int(second["source_frame_index"]), "timestamp_us": timestamps[1], "mask": str(second_instance["mask"]), "mask_pixels": int(second_instance["mask_pixels"]), "lidar_pixels": int(second_instance.get("lidar_pixels", 0))}],
                    }
                    pair_rows.setdefault((track_id, *camera_ids), []).append(observation)
    candidates: list[dict[str, Any]] = []
    for (track_id, first_camera, second_camera), rows in pair_rows.items():
        rows.sort(key=lambda value: int(value["reference_frame"]))
        for window in _window_candidates(rows, options=options):
            candidates.append({"track_id": track_id, "label": tracks[track_id].get("label"), "actor_family": "deformable", "camera_ids": [first_camera, second_camera], **window})
    chosen: list[dict[str, Any]] = []
    for track_id in sorted(tracks):
        per_track = sorted((value for value in candidates if int(value["track_id"]) == track_id), key=lambda value: (float(value["score"]), int(value["synchronized_frames"])), reverse=True)
        accepted: list[dict[str, Any]] = []
        for candidate in per_track:
            if len(accepted) >= options.max_windows_per_track:
                break
            candidate_refs = set(candidate["train_references"]) | set(candidate["holdout_references"])
            if any(candidate_refs & (set(prior["train_references"]) | set(prior["holdout_references"])) for prior in accepted):
                continue
            accepted.append(candidate)
        chosen.extend(accepted)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "version": VERSION, "status": "complete", "dynamic_dataset_manifest": str(manifest_path), "selection": {"min_mask_pixels": options.min_mask_pixels, "min_synchronized_frames": options.min_synchronized_frames, "window_frames": options.window_frames, "max_timestamp_spread_us": options.max_timestamp_spread_us, "min_camera_baseline_m": options.min_camera_baseline_m, "min_triangulation_angle_deg": options.min_triangulation_angle_deg, "holdout_every": options.holdout_every}, "candidates": chosen, "rejected": rejected, "limitations": ["Pair geometry is a source-observability gate based on coarse actor root and camera centres, not a reconstructed body surface.", "Only listed cameras/reference frames may enter a future multiview trainer; all other views must remain unknown/transparent.", "This plan contains no target-L4 render, generated completion or claim of 7V coverage."]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote deformable multiview plan: {output} ({len(chosen)} windows, rejected={rejected})", flush=True)
    return output
