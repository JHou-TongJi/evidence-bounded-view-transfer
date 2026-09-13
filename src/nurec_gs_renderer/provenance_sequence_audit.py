"""Temporal acceptance audit for source-provenance dynamic rendering policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask, _read_actor_tracks, masked_mae, temporal_delta_residual, vehicle_roi
from .camera import compose_ncore_camera_pose
from .pose_sources import _create_ncore_loader, _load_ncore_source_camera_path
from .street_dataset import SegformerVehicleSegmenter


@dataclass(frozen=True)
class ProvenanceSequenceAuditOptions:
    ncore_path: Path
    static_dir: Path
    native_blend_dir: Path
    source_policy_dir: Path
    actor_dataset_manifest: Path
    segformer_model_dir: Path
    output: Path
    track_id: int
    camera_id: str = "camera_front_wide_120fov"
    device: str = "cuda"
    probability_threshold: float = 0.60
    margin_threshold: float = 0.05
    cuboid_margin_pixels: int = 8


def _manifest(directory: Path) -> dict[str, Any]:
    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("status") != "complete" or not isinstance(data.get("frames"), list):
        raise ValueError(f"incomplete render manifest: {directory}")
    if not data["frames"]:
        raise ValueError(f"render manifest has no frames: {directory}")
    return data


def _frame_lookup(manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in manifest["frames"]:
        if not record.get("complete") or "rgb" not in record.get("outputs", {}):
            raise ValueError(f"incomplete RGB record in {root}")
        output[str(record["frame_id"])] = record
    return output


def _rgb(root: Path, record: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(root / record["outputs"]["rgb"]).convert("RGB"), dtype=np.float32) / 255.0


def _track(path: Path, track_id: int) -> dict[str, Any]:
    matches = [item for item in _read_actor_tracks(path) if int(item["track_id"]) == int(track_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen track {track_id}; got {len(matches)}")
    return matches[0]


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not valid else float(np.mean(valid))


def audit_provenance_sequence(options: ProvenanceSequenceAuditOptions) -> Path:
    """Compare static, native blend and one provenance policy on one camera window."""
    manifests = {
        "static": _manifest(options.static_dir),
        "native_blend": _manifest(options.native_blend_dir),
        "source_policy": _manifest(options.source_policy_dir),
    }
    lookups = {name: _frame_lookup(manifest, root) for name, (manifest, root) in {
        "static": (manifests["static"], options.static_dir),
        "native_blend": (manifests["native_blend"], options.native_blend_dir),
        "source_policy": (manifests["source_policy"], options.source_policy_dir),
    }.items()}
    frame_ids = tuple(lookups["static"])
    if not frame_ids or any(tuple(lookup) != frame_ids for lookup in lookups.values()):
        raise ValueError("static/native/policy manifests do not contain the same ordered frames")
    loader = _create_ncore_loader(options.ncore_path)
    sensor = loader.get_camera_sensor(options.camera_id)
    from ncore.impl.sensors.camera import FThetaCameraModel
    model = FThetaCameraModel(sensor.model_parameters, device=options.device)
    segmenter = SegformerVehicleSegmenter(options.segformer_model_dir, options.device)
    track = _track(options.actor_dataset_manifest, options.track_id)
    previous: dict[str, np.ndarray] | None = None
    previous_source: np.ndarray | None = None
    previous_roi: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    for ordinal, frame_id in enumerate(frame_ids, start=1):
        record = lookups["static"][frame_id]
        frame_index = int(str(frame_id).split("_", 1)[0])
        rendered = {name: _rgb(root, lookups[name][frame_id]) for name, root in (
            ("static", options.static_dir), ("native_blend", options.native_blend_dir), ("source_policy", options.source_policy_dir)
        )}
        shape = rendered["static"].shape
        if any(value.shape != shape for value in rendered.values()):
            raise ValueError(f"render resolution mismatch at {frame_id}")
        source_native = np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8)
        source = np.asarray(Image.fromarray(source_native, mode="RGB").resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8).astype(np.float32) / 255.0
        camera = _load_ncore_source_camera_path(
            {"frame_indices": [frame_index], "output": {"width": shape[1], "height": shape[0]}}, loader, options.camera_id
        ).frames[0]
        start = int(camera.timestamp_start_us if camera.timestamp_start_us is not None else camera.timestamp_us)
        end = int(camera.timestamp_us if camera.timestamp_us is not None else start)
        midpoint = (start + end) // 2
        rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([midpoint], dtype=np.uint64)))[0]
        camera_to_world = compose_ncore_camera_pose(rig_to_world, np.asarray(sensor.T_sensor_rig))
        cuboid = _ftheta_cuboid_mask(
            [track], midpoint, camera_to_world, model, height=shape[0], width=shape[1],
            source_height=source_native.shape[0], source_width=source_native.shape[1],
            margin_pixels=options.cuboid_margin_pixels, device=options.device,
        ).astype(bool)
        probability, margin = segmenter.vehicle_scores(Image.fromarray((source * 255).astype(np.uint8), mode="RGB"))
        semantic = vehicle_roi(cuboid, np.ones_like(cuboid), probability, margin,
                               probability_threshold=options.probability_threshold, margin_threshold=options.margin_threshold)
        roi = semantic if semantic.any() else cuboid
        mae = {name: masked_mae(image, source, roi) for name, image in rendered.items()}
        temporal = {name: None for name in rendered}
        if previous is not None and previous_source is not None and previous_roi is not None:
            pair_roi = roi | previous_roi
            temporal = {name: temporal_delta_residual(previous[name], image, previous_source, source, pair_roi)
                        for name, image in rendered.items()}
        rows.append({
            "frame_id": frame_id, "frame_index": frame_index, "timestamp_midpoint_us": midpoint,
            "roi_kind": "vehicle_semantic_intersection" if semantic.any() else "cuboid_fallback_no_vehicle_semantic",
            "roi_pixels": int(roi.sum()), "mae": mae, "temporal_delta_residual": temporal,
        })
        print(f"[{ordinal:02d}/{len(frame_ids)}] frame={frame_index} roi={int(roi.sum())} native={mae['native_blend']} policy={mae['source_policy']}", flush=True)
        previous, previous_source, previous_roi = rendered, source, roi
    def metric(layer: str, key: str) -> float | None:
        return _mean([row[key][layer] for row in rows])
    native_mae, policy_mae = metric("native_blend", "mae"), metric("source_policy", "mae")
    native_temporal, policy_temporal = metric("native_blend", "temporal_delta_residual"), metric("source_policy", "temporal_delta_residual")
    report = {
        "schema_version": 1,
        "purpose": "Short-window temporal acceptance of a source-camera/nearest-slot provenance policy; diagnostic only.",
        "inputs": {"static_dir": str(options.static_dir.resolve()), "native_blend_dir": str(options.native_blend_dir.resolve()), "source_policy_dir": str(options.source_policy_dir.resolve())},
        "track_id": options.track_id, "camera_id": options.camera_id, "frame_count": len(rows),
        "mean": {
            "static_roi_mae": metric("static", "mae"), "native_blend_roi_mae": native_mae, "source_policy_roi_mae": policy_mae,
            "policy_vs_native_roi_improvement_percent": None if native_mae is None or policy_mae is None else float((native_mae - policy_mae) / max(native_mae, 1e-12) * 100.0),
            "native_blend_temporal_delta_residual": native_temporal, "source_policy_temporal_delta_residual": policy_temporal,
            "policy_vs_native_temporal_improvement_percent": None if native_temporal is None or policy_temporal is None else float((native_temporal - policy_temporal) / max(native_temporal, 1e-12) * 100.0),
            "policy_roi_improved_frame_fraction": float(np.mean([
                row["mae"]["source_policy"] < row["mae"]["native_blend"] for row in rows
            ])),
        },
        "frames": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote provenance temporal audit: {options.output}", flush=True)
    return options.output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a source-provenance dynamic policy on a short exact-FTheta sequence")
    parser.add_argument("--ncore-path", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--native-blend-dir", type=Path, required=True)
    parser.add_argument("--source-policy-dir", type=Path, required=True)
    parser.add_argument("--actor-dataset-manifest", type=Path, required=True)
    parser.add_argument("--segformer-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vehicle-probability-threshold", type=float, default=0.60)
    parser.add_argument("--vehicle-margin-threshold", type=float, default=0.05)
    parser.add_argument("--cuboid-margin-pixels", type=int, default=8)
    args = parser.parse_args(argv)
    audit_provenance_sequence(ProvenanceSequenceAuditOptions(
        ncore_path=args.ncore_path, static_dir=args.static_dir, native_blend_dir=args.native_blend_dir,
        source_policy_dir=args.source_policy_dir, actor_dataset_manifest=args.actor_dataset_manifest,
        segformer_model_dir=args.segformer_model_dir, output=args.output, track_id=args.track_id,
        camera_id=args.camera_id, device=args.device, probability_threshold=args.vehicle_probability_threshold,
        margin_threshold=args.vehicle_margin_threshold, cuboid_margin_pixels=args.cuboid_margin_pixels,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
