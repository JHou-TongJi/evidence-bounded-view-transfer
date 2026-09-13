"""Clip-agnostic safety audit for RGB-over-static actor display composites."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ActorSequenceAuditOptions:
    actor_output: Path
    static_rgb_dir: Path
    output: Path
    camera_id: str
    alpha_threshold: float = 1.0 / 255.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1]")


def _static_path(directory: Path, index: int) -> Path:
    exact = directory / f"{index:06d}.png"
    if exact.is_file():
        return exact
    matches = sorted(directory.glob(f"{index:06d}_*.png"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"no unambiguous static RGB for output frame {index}: {directory}")


def _static_directory(root: Path, camera_id: str) -> Path:
    """Accept the same single-camera directory or multi-camera root as rendering."""
    per_camera = root / camera_id / "rgb"
    return per_camera if per_camera.is_dir() else root


def audit_actor_sequence(options: ActorSequenceAuditOptions) -> Path:
    root = options.actor_output.resolve() / options.camera_id
    rgb_files = sorted((root / "rgb").glob("*.png"))
    alpha_files = sorted((root / "actor_alpha").glob("*.png"))
    if not rgb_files or [item.name for item in rgb_files] != [item.name for item in alpha_files]:
        raise ValueError("actor output must contain aligned rgb/actor_alpha PNG frames")
    outside_mae: list[float] = []
    areas: list[float] = []
    ious: list[float] = []
    previous: np.ndarray | None = None
    for index, (rgb_path, alpha_path) in enumerate(zip(rgb_files, alpha_files)):
        with Image.open(rgb_path) as opened:
            final = np.asarray(opened.convert("RGB"), np.float32) / 255.0
        with Image.open(alpha_path) as opened:
            alpha = np.asarray(opened.convert("L"), np.float32) / 255.0
        with Image.open(_static_path(_static_directory(options.static_rgb_dir.resolve(), options.camera_id), index)) as opened:
            static = np.asarray(opened.convert("RGB").resize((final.shape[1], final.shape[0])), np.float32) / 255.0
        outside = alpha <= options.alpha_threshold
        outside_mae.append(float(np.abs(final - static)[outside].mean()) if np.any(outside) else 0.0)
        areas.append(float(alpha.mean()))
        binary = alpha > .5
        if previous is not None:
            union = int((previous | binary).sum())
            if union:
                ious.append(float((previous & binary).sum() / union))
        previous = binary
    report = {
        "schema": "ncore-2dgs-actor-sequence-audit", "version": 1, "status": "complete",
        "actor_output": str(options.actor_output.resolve()), "static_rgb_dir": str(options.static_rgb_dir.resolve()),
        "camera_id": options.camera_id, "frame_count": len(rgb_files),
        "metrics": {
            "outside_actor_rgb_mae": float(np.mean(outside_mae)),
            "outside_actor_rgb_mae_max": float(np.max(outside_mae)),
            "actor_active_frames": int(sum(value > options.alpha_threshold for value in areas)),
            "actor_alpha_area_mean": float(np.mean(areas)),
            "actor_alpha_area_active_mean": float(np.mean([value for value in areas if value > options.alpha_threshold])) if any(value > options.alpha_threshold for value in areas) else 0.0,
            "adjacent_alpha_iou_mean": float(np.mean(ious)) if ious else 1.0,
            "adjacent_alpha_iou_p05": float(np.percentile(ious, 5)) if ious else 1.0,
        },
        "limitations": [
            "Outside-actor RGB equality checks compositing safety only; it is not target-view photometric ground truth.",
            "Alpha temporal IoU does not judge whether an actor is geometrically correct in an unobserved target view.",
        ],
    }
    output = options.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote actor sequence audit: {output}", flush=True)
    return output
