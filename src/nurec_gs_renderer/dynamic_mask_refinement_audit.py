"""Non-training acceptance audit for SAM2-refined dynamic supervision."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def longest_zero_run(values: list[bool]) -> int:
    current = best = 0
    for value in values:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def _bracket(values: list[bool]) -> tuple[int, int] | None:
    present = [index for index, value in enumerate(values) if value]
    return None if not present else (present[0], present[-1])


@dataclass(frozen=True)
class DynamicMaskRefinementAuditOptions:
    refinement_manifest: Path
    output: Path
    rigid_only: bool = False
    minimum_new_masks: int = 5


def audit_dynamic_mask_refinement(options: DynamicMaskRefinementAuditOptions) -> Path:
    manifest_path = options.refinement_manifest.resolve()
    refined = _read(manifest_path)
    if refined.get("schema") != "ncore-dynamic-reconstruction-mask-refinement" or refined.get("status") != "complete":
        raise ValueError("refinement manifest must be complete")
    source = _read(Path(refined["source_manifest"]))
    tracks = {int(value["track_id"]): value for value in source["tracks"]}
    observations = [
        dict(value, camera_id=camera["camera_id"])
        for camera in refined["cameras"] for value in camera.get("observations", [])
        if not options.rigid_only or tracks[int(value["track_id"])]["actor_family"] == "rigid"
    ]
    by_camera_track: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for value in observations:
        by_camera_track.setdefault((value["camera_id"], int(value["track_id"])), []).append(value)
    reports: list[dict[str, Any]] = []
    lost = []
    for (camera_id, track_id), values in sorted(by_camera_track.items()):
        values.sort(key=lambda value: int(value["reference_frame_index"]))
        source_present = [int(value["source_mask_pixels"]) > 0 for value in values]
        # ``rejected_overlap`` retains a temporary pixel count for diagnosis,
        # but deliberately does not write an instance mask.  Only an actual
        # file is a final supervision observation.
        refined_present = [isinstance(value.get("refined_mask"), str) for value in values]
        bracket = _bracket(source_present)
        if bracket is None:
            continue
        lo, hi = bracket
        source_window, refined_window = source_present[lo : hi + 1], refined_present[lo : hi + 1]
        new_indices = [
            index for index, value in enumerate(values)
            if value["source_mask_pixels"] == 0 and isinstance(value.get("refined_mask"), str)
        ]
        new = [values[index] for index in new_indices]
        source_areas = np.asarray([value["source_mask_pixels"] for value in values], dtype=np.float32)
        ratios = []
        for index in new_indices:
            value = values[index]
            distances = np.abs(np.arange(len(values)) - index)
            candidates = np.flatnonzero(source_areas > 0)
            if len(candidates):
                nearest = candidates[np.argmin(distances[candidates])]
                ratios.append(float(value["refined_mask_pixels"] / source_areas[nearest]))
        dropped = [
            value for value in values
            if value["source_mask_pixels"] > 0 and not isinstance(value.get("refined_mask"), str)
        ]
        lost.extend(dropped)
        reports.append({
            "camera_id": camera_id, "track_id": track_id, "actor_family": tracks[track_id]["actor_family"],
            "label": tracks[track_id]["label"], "source_observations": int(sum(source_present)),
            "refined_observations": int(sum(refined_present)), "new_observations": len(new),
            "source_lost": len(dropped), "source_max_gap": longest_zero_run(source_window),
            "refined_max_gap": longest_zero_run(refined_window),
            "new_with_lidar": int(sum(int(value.get("lidar_pixels", 0)) > 0 for value in new)),
            "new_area_ratio_median": None if not ratios else float(np.median(ratios)),
            "new_area_ratio_p90": None if not ratios else float(np.percentile(ratios, 90)),
        })
    per_track: dict[int, dict[str, Any]] = {}
    for report in reports:
        item = per_track.setdefault(report["track_id"], {
            "track_id": report["track_id"], "actor_family": report["actor_family"], "label": report["label"],
            "source_observations": 0, "refined_observations": 0, "new_observations": 0, "source_lost": 0,
            "new_with_lidar": 0, "camera_reports": 0, "gap_reduction_camera_reports": 0,
        })
        for key in ("source_observations", "refined_observations", "new_observations", "source_lost", "new_with_lidar"):
            item[key] += report[key]
        item["camera_reports"] += 1
        item["gap_reduction_camera_reports"] += int(report["refined_max_gap"] < report["source_max_gap"])
    candidates = [
        value for value in per_track.values()
        if value["actor_family"] == "rigid" and value["new_observations"] >= options.minimum_new_masks
        and value["gap_reduction_camera_reports"] > 0 and value["source_lost"] == 0
    ]
    output = options.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ncore-dynamic-mask-refinement-audit", "version": 1,
        "refinement_manifest": str(manifest_path), "source_manifest": refined["source_manifest"],
        "source_observations_lost": len(lost), "source_preservation": not lost,
        "source_preservation_pass": not lost,
        "camera_track_reports": reports, "track_reports": sorted(per_track.values(), key=lambda value: value["track_id"]),
        "rigid_training_candidates": candidates,
        "limitations": [
            "This audit checks supervision continuity/provenance, not rendered RGB quality.",
            "A candidate still requires exact-FTheta RGB, LiDAR and cross-camera holdout validation before training.",
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote dynamic mask refinement audit: {output}", flush=True)
    print(f"source_preservation={not lost} rigid_candidates={[value['track_id'] for value in candidates]}", flush=True)
    return output
