"""Audit target-view actor footprints before applying any heuristic rejection.

The target L4 view has no RGB ground truth, so an actor alpha morphology
metric is evidence for review, *not* a replacement for source-view validation.
This tool deliberately never alters pixels or selects a new expert.  It makes
fragmentation, sudden area changes and image-edge clipping explicit per track
so a future rejector can be calibrated against real failures rather than
silently suppressing novel-view actors.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ActorTargetVisibilityAuditOptions:
    actor_output: Path
    camera_id: str
    output: Path
    alpha_threshold: float = .50
    min_component_pixels: int = 64
    max_minor_component_fraction: float = .10
    max_adjacent_area_ratio: float = 2.5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in (0, 1]")
        if self.min_component_pixels <= 0:
            raise ValueError("min_component_pixels must be positive")
        if not 0.0 <= self.max_minor_component_fraction <= 1.0:
            raise ValueError("max_minor_component_fraction must be in [0, 1]")
        if self.max_adjacent_area_ratio <= 1.0:
            raise ValueError("max_adjacent_area_ratio must exceed one")


def footprint_metrics(mask: np.ndarray, min_component_pixels: int) -> dict[str, Any]:
    """Return 8-connected footprint statistics without an image-model dependency."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask must be HxW")
    height, width = value.shape
    # Dense actor masks are small at diagnostic resolution. A local DFS keeps
    # this utility portable: SciPy/OpenCV are not required by the renderer.
    visited = np.zeros_like(value, dtype=bool)
    component_sizes: list[int] = []
    component_boxes: list[tuple[int, int, int, int]] = []
    for y, x in zip(*np.nonzero(value)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        count = 0; y0 = y1 = int(y); x0 = x1 = int(x)
        while stack:
            cy, cx = stack.pop(); count += 1
            y0 = min(y0, cy); y1 = max(y1, cy); x0 = min(x0, cx); x1 = max(x1, cx)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if value[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; stack.append((ny, nx))
        if count >= min_component_pixels:
            component_sizes.append(count); component_boxes.append((y0, x0, y1, x1))
    pixels = int(value.sum())
    if not component_sizes:
        return {"pixels": pixels, "component_count": 0, "largest_component_fraction": 0.0,
                "minor_component_fraction": 0.0, "bbox": None, "bbox_fill_fraction": 0.0,
                "touches_edge": False}
    largest_index = int(np.argmax(component_sizes))
    largest = component_sizes[largest_index]
    y0 = min(box[0] for box in component_boxes); x0 = min(box[1] for box in component_boxes)
    y1 = max(box[2] for box in component_boxes); x1 = max(box[3] for box in component_boxes)
    bbox_pixels = (y1 - y0 + 1) * (x1 - x0 + 1)
    accepted_pixels = sum(component_sizes)
    return {
        "pixels": pixels, "component_count": len(component_sizes),
        "largest_component_fraction": float(largest / max(accepted_pixels, 1)),
        "minor_component_fraction": float((accepted_pixels - largest) / max(accepted_pixels, 1)),
        "bbox": [int(y0), int(x0), int(y1), int(x1)],
        "bbox_fill_fraction": float(accepted_pixels / max(bbox_pixels, 1)),
        "touches_edge": bool(y0 == 0 or x0 == 0 or y1 == height - 1 or x1 == width - 1),
    }


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened, np.uint16)


def audit_actor_target_visibility(options: ActorTargetVisibilityAuditOptions) -> Path:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    root = options.actor_output.resolve() / options.camera_id
    alpha_paths = sorted((root / "actor_alpha").glob("*.png"))
    owner_paths = sorted((root / "actor_owner").glob("*.png"))
    if not alpha_paths or [path.name for path in alpha_paths] != [path.name for path in owner_paths]:
        raise ValueError("actor output must contain aligned actor_alpha and actor_owner PNG frames")
    manifest = json.loads((options.actor_output.resolve() / "manifest.json").read_text(encoding="utf-8"))
    experts = manifest.get("experts", [])
    track_ids = sorted({int(value["track_id"]) for value in experts if isinstance(value, dict) and value.get("track_id") is not None})
    # Older actor renders predate track_id in their manifest. The uint16 owner
    # debug image stores ``track_id + 1`` precisely to make that provenance
    # recoverable without re-rendering an otherwise valid sequence.
    if not track_ids:
        track_ids = sorted({int(value) - 1 for path in owner_paths for value in np.unique(_read_gray(path)) if int(value) > 0})
    per_track: dict[int, list[dict[str, Any]]] = {track_id: [] for track_id in track_ids}
    previous_area: dict[int, int] = {}
    for index, (alpha_path, owner_path) in enumerate(zip(alpha_paths, owner_paths)):
        alpha = _read_gray(alpha_path).astype(np.float32) / 255.0
        owner = _read_gray(owner_path).astype(np.int32) - 1
        for track_id in track_ids:
            mask = (owner == track_id) & (alpha >= options.alpha_threshold)
            metrics = footprint_metrics(mask, options.min_component_pixels)
            # Single pixels may survive rasterisation/quantisation at a fade
            # boundary. They are not a visible actor footprint and must not
            # seed the next frame's area-ratio comparison.
            if not metrics["component_count"]:
                previous_area.pop(track_id, None)
                continue
            prior = previous_area.get(track_id)
            ratio = None if prior is None else float(metrics["pixels"] / max(prior, 1))
            flags: list[str] = []
            if metrics["minor_component_fraction"] > options.max_minor_component_fraction:
                flags.append("fragmented")
            if ratio is not None and (ratio > options.max_adjacent_area_ratio or ratio < 1.0 / options.max_adjacent_area_ratio):
                flags.append("abrupt_area_change")
            if metrics["touches_edge"]:
                flags.append("edge_clipped")
            per_track[track_id].append({"frame_index": index, "area_ratio_to_previous": ratio, "flags": flags, **metrics})
            previous_area[track_id] = int(metrics["pixels"])
    tracks: list[dict[str, Any]] = []
    for track_id in track_ids:
        frames = per_track[track_id]
        flags = [flag for record in frames for flag in record["flags"]]
        tracks.append({
            "track_id": track_id, "active_frames": len(frames),
            "mean_pixels": float(np.mean([value["pixels"] for value in frames])) if frames else 0.0,
            "mean_largest_component_fraction": float(np.mean([value["largest_component_fraction"] for value in frames])) if frames else 0.0,
            "fragmented_frames": int(flags.count("fragmented")), "abrupt_area_frames": int(flags.count("abrupt_area_change")),
            "edge_clipped_frames": int(flags.count("edge_clipped")), "frames": frames,
        })
    report = {
        "schema": "ncore-2dgs-target-actor-visibility-audit", "version": 1, "status": "complete",
        "actor_output": str(options.actor_output.resolve()), "camera_id": options.camera_id,
        "frame_count": len(alpha_paths), "thresholds": {"alpha": options.alpha_threshold, "min_component_pixels": options.min_component_pixels, "max_minor_component_fraction": options.max_minor_component_fraction, "max_adjacent_area_ratio": options.max_adjacent_area_ratio},
        "tracks": tracks,
        "limitations": [
            "Target-view alpha morphology is diagnostic evidence, not photometric or geometric target-view ground truth.",
            "Flags never alter the sequence. Calibrate any future rejector against source-view holdout and human review before enabling it.",
            "Image-edge clipping may be physically correct for an actor leaving the target view.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote target actor visibility audit: {output}", flush=True)
    return output
