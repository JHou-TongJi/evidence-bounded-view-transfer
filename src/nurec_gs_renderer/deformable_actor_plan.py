"""Prepare clip-agnostic short-window supervision plans for deformable actors.

This intentionally does not render people.  It prevents rigid 2DGS experts
from silently absorbing pedestrians while producing deterministic candidate
windows for a future deformable/video-conditioned trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeformableActorPlanOptions:
    dynamic_dataset_manifest: Path
    output: Path
    min_mask_pixels: int = 800
    min_observations: int = 12
    window_frames: int = 32
    max_windows_per_track: int = 2

    def __post_init__(self) -> None:
        if self.min_mask_pixels <= 0 or self.min_observations < 3 or self.window_frames < 4 or self.max_windows_per_track <= 0:
            raise ValueError("invalid deformable plan thresholds")


def build_deformable_actor_plan(options: DeformableActorPlanOptions) -> Path:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    dataset = json.loads(options.dynamic_dataset_manifest.resolve().read_text(encoding="utf-8"))
    if dataset.get("schema") != "ncore-dynamic-reconstruction-dataset" or dataset.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be complete")
    families = {int(track["track_id"]): track for track in dataset.get("tracks", []) if isinstance(track, dict) and track.get("actor_family") == "deformable"}
    groups: dict[tuple[int, str], list[tuple[int, int, str]]] = {}
    for frame in dataset.get("frames", []):
        reference = int(frame.get("reference_frame_index", -1))
        for camera in frame.get("cameras", []):
            camera_id = str(camera.get("camera_id", ""))
            for instance in camera.get("instances", []):
                track_id = int(instance.get("track_id", -1))
                pixels = int(instance.get("mask_pixels", 0))
                mask = instance.get("mask")
                if track_id in families and reference >= 0 and camera_id and mask and pixels >= options.min_mask_pixels:
                    groups.setdefault((track_id, camera_id), []).append((reference, pixels, str(mask)))
    candidates: list[dict[str, Any]] = []
    for (track_id, camera_id), observations in groups.items():
        observations.sort()
        for start, _, _ in observations:
            window = [item for item in observations if start <= item[0] < start + options.window_frames]
            if len(window) < options.min_observations:
                continue
            candidates.append({"track_id": track_id, "label": families[track_id].get("label"), "camera_id": camera_id, "reference_start": start, "reference_end": start + options.window_frames, "observations": len(window), "mask_pixels_sum": sum(item[1] for item in window), "masks": [item[2] for item in window]})
    selected: list[dict[str, Any]] = []
    for track_id in sorted(families):
        per_track = sorted((item for item in candidates if item["track_id"] == track_id), key=lambda item: (item["mask_pixels_sum"], item["observations"]), reverse=True)
        chosen: list[dict[str, Any]] = []
        for item in per_track:
            if len(chosen) >= options.max_windows_per_track:
                break
            if any(item["camera_id"] == prior["camera_id"] and max(item["reference_start"], prior["reference_start"]) < min(item["reference_end"], prior["reference_end"]) for prior in chosen):
                continue
            chosen.append(item)
        selected.extend(chosen)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "ncore-deformable-actor-window-plan", "version": 1, "status": "complete", "dynamic_dataset_manifest": str(options.dynamic_dataset_manifest.resolve()), "selection": {"min_mask_pixels": options.min_mask_pixels, "min_observations": options.min_observations, "window_frames": options.window_frames, "max_windows_per_track": options.max_windows_per_track}, "candidates": selected, "limitations": ["This is a supervision plan only; no rigid Gaussian asset, target rendering, or person completion is implied.", "A future trainer must use camera-conditioned/deformable appearance and retain source-view visibility gates."]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote deformable actor window plan: {output} ({len(selected)} windows)", flush=True)
    return output
