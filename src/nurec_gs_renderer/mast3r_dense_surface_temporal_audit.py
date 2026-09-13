"""Temporal source-view audit for independently reconstructed dense surfaces."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import masked_mae, temporal_delta_residual
from .mast3r_dense_depth import DATASET_SCHEMA
from .mast3r_dense_surface_audit import _alpha, _frame, _manifest, _read_json, _resize_mask, _rgb
from .pose_sources import _create_ncore_loader


SCHEMA = "ncore-mast3r-metric-dense-surface-temporal-audit"
VERSION = 1


@dataclass(frozen=True)
class Mast3rDenseSurfaceTemporalAuditOptions:
    dataset_manifest: Path
    static_render_dirs: tuple[Path, ...]
    composite_render_dirs: tuple[Path, ...]
    output: Path
    track_id: int
    camera_id: str = "camera_cross_right_120fov"

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.output.suffix != ".json":
            raise ValueError("track/output parameters are invalid")
        if not self.static_render_dirs or len(self.static_render_dirs) != len(self.composite_render_dirs):
            raise ValueError("at least one paired static/composite render directory is required")


def temporal_summary(static: list[float], composite: list[float]) -> dict[str, float | int | None]:
    if len(static) != len(composite):
        raise ValueError("static/composite temporal values must be aligned")
    if not static:
        return {"pair_count": 0, "static_mean": None, "composite_mean": None, "improvement_percent": None, "all_pairs_improved": None}
    first, second = np.asarray(static, np.float64), np.asarray(composite, np.float64)
    return {"pair_count": int(len(first)), "static_mean": float(first.mean()), "composite_mean": float(second.mean()), "improvement_percent": float((first.mean() - second.mean()) / max(first.mean(), 1e-12) * 100.0), "all_pairs_improved": bool(np.all(second <= first))}


def audit_mast3r_dense_surface_temporal(options: Mast3rDenseSurfaceTemporalAuditOptions) -> dict[str, Any]:
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    manifest_path = options.dataset_manifest.resolve()
    dataset = _read_json(manifest_path)
    if dataset.get("schema") != DATASET_SCHEMA or dataset.get("status") != "complete":
        raise ValueError("dataset manifest is not a completed dynamic reconstruction dataset")
    pairs: list[tuple[int, Path, dict[str, Any], Path, dict[str, Any]]] = []
    for static_dir, composite_dir in zip(options.static_render_dirs, options.composite_render_dirs, strict=True):
        static_root, static_manifest = _manifest(static_dir)
        composite_root, composite_manifest = _manifest(composite_dir)
        static_frames = {str(item["frame_id"]): item for item in static_manifest["frames"] if item.get("complete")}
        composite_frames = {str(item["frame_id"]): item for item in composite_manifest["frames"] if item.get("complete")}
        if not static_frames or static_frames.keys() != composite_frames.keys():
            raise ValueError("paired static/composite renders must contain the same non-empty complete frame IDs")
        for frame_id, static_frame in static_frames.items():
            try:
                reference = int(frame_id.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError("render frame ID does not encode NCore reference index") from exc
            pairs.append((reference, static_root, static_frame, composite_root, composite_frames[frame_id]))
    pairs.sort(key=lambda item: item[0])
    references = [item[0] for item in pairs]
    if len(references) != len(set(references)):
        raise ValueError("temporal audit requires one render pair per unique reference")
    loader = _create_ncore_loader(Path(dataset["ncore_path"]))
    sensor = loader.get_camera_sensor(options.camera_id)
    previous: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
    records: list[dict[str, Any]] = []
    static_temporal: list[float] = []
    composite_temporal: list[float] = []
    for reference, static_root, static_frame, composite_root, composite_frame in pairs:
        frame = next((item for item in dataset["frames"] if int(item["reference_frame_index"]) == reference), None)
        if frame is None:
            raise ValueError(f"reference {reference} absent from dataset")
        camera = next((item for item in frame["cameras"] if item["camera_id"] == options.camera_id), None)
        if camera is None:
            raise ValueError(f"reference {reference} has no camera {options.camera_id}")
        instance = next((item for item in camera["instances"] if int(item["track_id"]) == options.track_id), None)
        if instance is None:
            raise ValueError(f"reference {reference} has no source instance for track {options.track_id}")
        static = _rgb(static_root / static_frame["outputs"]["rgb"])
        composite = _rgb(composite_root / composite_frame["outputs"]["rgb"])
        if static.shape != composite.shape:
            raise ValueError("static/composite RGB shapes differ")
        source = np.asarray(sensor.get_frame_image_array(int(camera["source_frame_index"])), np.uint8)
        source = np.asarray(Image.fromarray(source, mode="RGB").resize((static.shape[1], static.shape[0]), Image.Resampling.BILINEAR), np.float32) / 255.0
        mask = _resize_mask(manifest_path.parent / str(instance["mask"]), static.shape[:2])
        record: dict[str, Any] = {"reference_index": reference, "source_frame_index": int(camera["source_frame_index"]), "mask_pixels": int(mask.sum()), "static_mask_mae": masked_mae(static, source, mask), "composite_mask_mae": masked_mae(composite, source, mask), "temporal_delta": None}
        if previous is not None:
            previous_static, previous_composite, previous_source, previous_mask, previous_reference = previous
            union = previous_mask | mask
            static_delta = temporal_delta_residual(previous_static, static, previous_source, source, union)
            composite_delta = temporal_delta_residual(previous_composite, composite, previous_source, source, union)
            if static_delta is None or composite_delta is None:
                raise RuntimeError("non-empty source masks unexpectedly produced no temporal residual")
            static_temporal.append(static_delta)
            composite_temporal.append(composite_delta)
            record["temporal_delta"] = {"previous_reference_index": previous_reference, "union_mask_pixels": int(union.sum()), "static": static_delta, "composite": composite_delta, "improvement_percent": float((static_delta - composite_delta) / max(static_delta, 1e-12) * 100.0)}
        records.append(record)
        previous = (static, composite, source, mask, reference)
    summary = temporal_summary(static_temporal, composite_temporal)
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "passed": bool(summary["all_pairs_improved"]), "inputs": {"dataset_manifest": str(manifest_path), "track_id": options.track_id, "camera_id": options.camera_id, "references": references}, "frames": records, "temporal_delta": {"definition": "MAE((render_t-render_t-1)-(NCore_t-NCore_t-1)) over union of exact source instance masks", **summary}, "limitations": ["Only the listed source camera/time window is validated.", "Pass does not establish target-L4 occlusion correctness or full-sequence dynamic coverage."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MASt3R dense temporal audit references={references} improvement={summary['improvement_percent']} all_pairs_improved={summary['all_pairs_improved']}", flush=True)
    return report
