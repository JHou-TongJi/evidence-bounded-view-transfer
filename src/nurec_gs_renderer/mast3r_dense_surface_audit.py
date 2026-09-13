"""Source-view acceptance audit for a visibility-bound MASt3R dense surface.

Unlike the older generic actor audit, this uses the exact track instance mask
that generated the dense depth proposal.  That matters here: one nearby truck
must not be scored together with every projected vehicle cuboid in the scene.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import masked_mae
from .mast3r_dense_depth import DATASET_SCHEMA
from .pose_sources import _create_ncore_loader


SCHEMA = "ncore-mast3r-metric-dense-surface-source-audit"
VERSION = 1


def alpha_iou(predicted_alpha: np.ndarray, reference_mask: np.ndarray, threshold: float = .02) -> tuple[float, float, float]:
    """Return alpha-mask IoU, recall and area ratio for one source instance."""
    prediction = np.asarray(predicted_alpha, np.float32) >= threshold
    reference = np.asarray(reference_mask, bool)
    if prediction.shape != reference.shape:
        raise ValueError("alpha and source mask dimensions must match")
    intersection = int((prediction & reference).sum())
    union = int((prediction | reference).sum())
    reference_count = int(reference.sum())
    return (
        float(intersection / union) if union else 1.0,
        float(intersection / reference_count) if reference_count else 1.0,
        float(prediction.sum() / max(reference_count, 1)),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    value = _read_json(root / "manifest.json")
    if value.get("status") != "complete" or not isinstance(value.get("frames"), list):
        raise ValueError(f"render directory is not complete: {root}")
    return root, value


def _frame(root: Path, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    complete = [(str(item["frame_id"]), item) for item in manifest["frames"] if item.get("complete")]
    if len(complete) != 1:
        raise ValueError(f"dense surface audit expects exactly one completed render frame: {root}")
    frame_id, frame = complete[0]
    if not {"rgb", "alpha"}.issubset(frame.get("outputs", {})):
        raise ValueError(f"render frame lacks rgb/alpha: {root}")
    return frame_id, frame


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def _resize_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.asarray(Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST), np.uint8) > 0


@dataclass(frozen=True)
class Mast3rDenseSurfaceAuditOptions:
    dataset_manifest: Path
    static_render_dir: Path
    actor_render_dir: Path
    composite_render_dir: Path
    output: Path
    track_id: int
    camera_id: str = "camera_cross_right_120fov"
    alpha_threshold: float = .02
    min_iou: float = .60
    min_recall: float = .75
    min_mae_improvement_percent: float = 3.0
    max_outside_change_fraction: float = .02

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.output.suffix != ".json":
            raise ValueError("track/output parameters are invalid")
        if not 0 <= self.alpha_threshold <= 1 or not 0 <= self.min_iou <= 1 or not 0 <= self.min_recall <= 1:
            raise ValueError("alpha/IoU/recall thresholds are invalid")
        if self.min_mae_improvement_percent < 0 or not 0 <= self.max_outside_change_fraction <= 1:
            raise ValueError("MAE/outside-change thresholds are invalid")


def audit_mast3r_dense_surface(options: Mast3rDenseSurfaceAuditOptions) -> dict[str, Any]:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    manifest_path = options.dataset_manifest.resolve()
    dataset = _read_json(manifest_path)
    if dataset.get("schema") != DATASET_SCHEMA or dataset.get("status") != "complete":
        raise ValueError("dataset manifest is not a completed dynamic reconstruction dataset")
    roots_and_manifests = [_manifest(path) for path in (options.static_render_dir, options.actor_render_dir, options.composite_render_dir)]
    frames = [_frame(root, manifest) for root, manifest in roots_and_manifests]
    frame_ids = {item[0] for item in frames}
    if len(frame_ids) != 1:
        raise ValueError("static, actor and composite inputs must use the same frame ID")
    frame_id = next(iter(frame_ids))
    try:
        reference_index = int(frame_id.split("_", 1)[0])
    except ValueError as exc:
        raise ValueError("render frame ID does not encode an NCore source index") from exc
    observation = next((item for item in dataset["frames"] if int(item["reference_frame_index"]) == reference_index), None)
    if observation is None:
        raise ValueError(f"reference {reference_index} is absent from dataset manifest")
    camera = next((item for item in observation["cameras"] if item["camera_id"] == options.camera_id), None)
    if camera is None:
        raise ValueError(f"dataset reference {reference_index} lacks {options.camera_id}")
    instance = next((item for item in camera["instances"] if int(item["track_id"]) == options.track_id), None)
    if instance is None:
        raise ValueError(f"dataset source observation has no track {options.track_id} instance mask")

    (static_root, _), (actor_root, _), (composite_root, _) = roots_and_manifests
    (_, static_frame), (_, actor_frame), (_, composite_frame) = frames
    static = _rgb(static_root / static_frame["outputs"]["rgb"])
    actor = _rgb(actor_root / actor_frame["outputs"]["rgb"])
    composite = _rgb(composite_root / composite_frame["outputs"]["rgb"])
    actor_alpha = _alpha(actor_root / actor_frame["outputs"]["alpha"])
    if static.shape != actor.shape or static.shape != composite.shape or static.shape[:2] != actor_alpha.shape:
        raise ValueError("static, actor and composite render dimensions must match")
    source_mask = _resize_mask(manifest_path.parent / str(instance["mask"]), static.shape[:2])
    loader = _create_ncore_loader(Path(dataset["ncore_path"]))
    sensor = loader.get_camera_sensor(options.camera_id)
    source = np.asarray(sensor.get_frame_image_array(int(camera["source_frame_index"])), np.uint8)
    source = np.asarray(Image.fromarray(source, mode="RGB").resize((static.shape[1], static.shape[0]), Image.Resampling.BILINEAR), np.float32) / 255.0

    iou, recall, area_ratio = alpha_iou(actor_alpha, source_mask, options.alpha_threshold)
    static_mae = masked_mae(static, source, source_mask)
    composite_mae = masked_mae(composite, source, source_mask)
    improvement = None if static_mae is None or composite_mae is None else float((static_mae - composite_mae) / max(static_mae, 1e-12) * 100.0)
    changed = np.max(np.abs(composite - static), axis=-1) > (1.0 / 255.0)
    outside = ~source_mask
    outside_fraction = float(changed[outside].mean()) if outside.any() else 0.0
    actor_front = None
    actor_front_path = composite_frame["outputs"].get("actor_front")
    if actor_front_path is not None:
        actor_front = _alpha(composite_root / str(actor_front_path)) >= .5
    front_recall = None if actor_front is None else float(actor_front[source_mask].mean())
    front_outside = None if actor_front is None else float(actor_front[outside].mean())
    passed = bool(
        iou >= options.min_iou and recall >= options.min_recall
        and improvement is not None and improvement >= options.min_mae_improvement_percent
        and outside_fraction <= options.max_outside_change_fraction
        and (front_recall is None or front_recall >= options.min_recall)
    )
    report = {
        "schema": SCHEMA, "version": VERSION, "status": "complete", "passed": passed,
        "inputs": {"dataset_manifest": str(manifest_path), "track_id": options.track_id, "camera_id": options.camera_id,
                   "static_render_dir": str(static_root), "actor_render_dir": str(actor_root), "composite_render_dir": str(composite_root)},
        "source": {"reference_index": reference_index, "source_frame_index": int(camera["source_frame_index"]), "mask": str(instance["mask"]), "mask_pixels_at_render_resolution": int(source_mask.sum())},
        "metrics": {"actor_alpha_iou": iou, "actor_alpha_recall": recall, "actor_alpha_area_ratio": area_ratio,
                    "static_mask_mae": static_mae, "composite_mask_mae": composite_mae, "composite_vs_static_mae_improvement_percent": improvement,
                    "composite_changed_fraction_outside_source_mask": outside_fraction,
                    "actor_front_recall_in_source_mask": front_recall, "actor_front_fraction_outside_source_mask": front_outside},
        "thresholds": {"min_iou": options.min_iou, "min_recall": options.min_recall, "min_mae_improvement_percent": options.min_mae_improvement_percent, "max_outside_change_fraction": options.max_outside_change_fraction},
        "limitations": ["This is one source-view/time acceptance test, not a full vehicle, temporal, cross-camera or target-L4 validation.", "Failure means do not expand this asset by iteration, copying it to adjacent frames, or compositing it into a sequence."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MASt3R dense surface audit frame={reference_index} iou={iou:.4f} recall={recall:.4f} improvement={improvement} outside={outside_fraction:.4f} passed={passed}", flush=True)
    return report
