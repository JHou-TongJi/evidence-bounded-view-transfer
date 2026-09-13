"""Create a conservative, provenance-preserving dynamic-layer supervision manifest.

This is deliberately *not* a dynamic Gaussian trainer.  It selects raw NCore
observations for a future trainer without using a static PLY as either a
background target or occlusion ground truth.  Original semantic/cuboid masks
remain training supervision; a SAM2-only mask is admitted only when the exact
FTheta cross-view audit accepted it, and then it is held out from optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "ncore-dynamic-layer-trusted-supervision"
VERSION = 1


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _key(camera_id: str, reference_frame_index: int, track_id: int) -> tuple[str, int, int]:
    return str(camera_id), int(reference_frame_index), int(track_id)


@dataclass(frozen=True)
class DynamicTrainingSelectionOptions:
    source_manifest: Path
    refinement_manifest: Path
    crossview_reports: tuple[Path, ...]
    output: Path
    track_ids: tuple[int, ...]
    verify_referenced_files: bool = True

    def __post_init__(self) -> None:
        if not self.track_ids or len(set(self.track_ids)) != len(self.track_ids) or any(value < 0 for value in self.track_ids):
            raise ValueError("track_ids must be unique non-negative values")
        if not self.crossview_reports:
            raise ValueError("at least one exact cross-view audit report is required")


def select_dynamic_training_supervision(options: DynamicTrainingSelectionOptions) -> Path:
    """Write one manifest referencing trusted original and held-out SAM2 masks.

    A cross-view accepted SAM2 observation must stay outside training.  Every
    selected actor observation at its reference frame is therefore moved to the
    ``crossview_holdout`` split; this avoids leaking the independently checked
    target image into a future appearance/geometry optimiser.
    """
    source_path = options.source_manifest.resolve()
    refinement_path = options.refinement_manifest.resolve()
    source = _read(source_path)
    refined = _read(refinement_path)
    if source.get("schema") != "ncore-dynamic-reconstruction-dataset" or source.get("status") != "complete":
        raise ValueError("source manifest must be a complete dynamic reconstruction dataset")
    if refined.get("schema") != "ncore-dynamic-reconstruction-mask-refinement" or refined.get("status") != "complete":
        raise ValueError("refinement manifest must be complete")
    if Path(str(refined.get("source_manifest", ""))).resolve() != source_path:
        raise ValueError("refinement manifest does not reference --source-manifest")
    if refined.get("source_manifest_sha256") not in {None, _sha256(source_path)}:
        raise ValueError("refinement source manifest hash does not match --source-manifest")

    tracks = {int(value["track_id"]): value for value in source.get("tracks", [])}
    requested = set(options.track_ids)
    missing = requested.difference(tracks)
    if missing:
        raise ValueError(f"requested tracks are absent from source dataset: {sorted(missing)}")
    non_rigid = [track_id for track_id in options.track_ids if tracks[track_id].get("actor_family") != "rigid"]
    if non_rigid:
        raise ValueError(f"this first trusted selector supports rigid tracks only, got: {non_rigid}")

    accepted_sam2: set[tuple[str, int, int]] = set()
    report_metadata: list[dict[str, str]] = []
    for report_arg in options.crossview_reports:
        report_path = report_arg.resolve()
        report = _read(report_path)
        if report.get("schema") != "ncore-dynamic-mask-crossview-audit":
            raise ValueError(f"not an exact cross-view audit report: {report_path}")
        if Path(str(report.get("source_manifest", ""))).resolve() != source_path:
            raise ValueError(f"cross-view report uses a different source manifest: {report_path}")
        if Path(str(report.get("refinement_manifest", ""))).resolve() != refinement_path:
            raise ValueError(f"cross-view report uses a different refinement manifest: {report_path}")
        report_metadata.append({"path": str(report_path), "sha256": _sha256(report_path)})
        for observation in report.get("observations", []):
            if observation.get("status") != "accepted":
                continue
            track_id = int(observation["track_id"])
            if track_id not in requested:
                continue
            accepted_sam2.add(_key(
                str(observation["source_camera"]), int(observation["reference_frame_index"]), track_id,
            ))
    if not accepted_sam2:
        raise RuntimeError("no requested SAM2 observation was accepted by the supplied cross-view audit")

    refinement_entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for camera in refined.get("cameras", []):
        camera_id = str(camera["camera_id"])
        for entry in camera.get("observations", []):
            if not isinstance(entry.get("refined_mask"), str):
                continue
            key = _key(camera_id, int(entry["reference_frame_index"]), int(entry["track_id"]))
            if key in refinement_entries:
                raise ValueError(f"duplicate refinement observation: {key}")
            refinement_entries[key] = entry
    missing_refined = accepted_sam2.difference(refinement_entries)
    if missing_refined:
        raise ValueError(f"cross-view accepted observations have no final refinement mask: {sorted(missing_refined)}")
    invalid_origin = [key for key in accepted_sam2 if refinement_entries[key].get("mask_origin") != "sam2"]
    if invalid_origin:
        raise ValueError(f"cross-view accepted observations must be SAM2-only masks: {invalid_origin}")

    source_root, refinement_root = source_path.parent, refinement_path.parent
    holdout_references = {reference for _, reference, _ in accepted_sam2}
    selected_frames: list[dict[str, Any]] = []
    counts = {"train_source": 0, "validation_source": 0, "crossview_holdout_source": 0, "crossview_holdout_sam2": 0}
    missing_paths: list[str] = []
    for frame in source.get("frames", []):
        reference = int(frame["reference_frame_index"])
        if reference in holdout_references:
            split = "crossview_holdout"
        else:
            split = str(frame["split"])
        cameras: list[dict[str, Any]] = []
        for camera in frame.get("cameras", []):
            camera_id = str(camera["camera_id"])
            instances: list[dict[str, Any]] = []
            for instance in camera.get("instances", []):
                track_id = int(instance["track_id"])
                if track_id not in requested:
                    continue
                copied = dict(instance)
                copied.update({"mask_root": "source_dataset", "lidar_depth_root": "source_dataset", "mask_origin": "source"})
                copied["supervision_role"] = "crossview_holdout_source" if split == "crossview_holdout" else f"{split}_source"
                counts[copied["supervision_role"]] = counts.get(copied["supervision_role"], 0) + 1
                if options.verify_referenced_files:
                    for rel in (copied["mask"], copied.get("lidar_depth")):
                        if isinstance(rel, str) and not (source_root / rel).is_file():
                            missing_paths.append(str(source_root / rel))
                instances.append(copied)
            for key in sorted(accepted_sam2):
                accepted_camera, accepted_reference, track_id = key
                if accepted_camera != camera_id or accepted_reference != reference:
                    continue
                entry = refinement_entries[key]
                if split != "crossview_holdout":  # Defensive: key's reference must force it above.
                    raise AssertionError("cross-view accepted SAM2 mask leaked into a train/validation split")
                copied = {
                    "track_id": track_id,
                    "label": tracks[track_id]["label"],
                    "actor_family": tracks[track_id]["actor_family"],
                    "mask": entry["refined_mask"],
                    "lidar_depth": entry.get("lidar_depth"),
                    "mask_pixels": int(entry.get("refined_mask_pixels", 0)),
                    "lidar_pixels": int(entry.get("lidar_pixels", 0)),
                    "mask_root": "refinement", "lidar_depth_root": "refinement", "mask_origin": "sam2",
                    "supervision_role": "crossview_holdout_sam2",
                    "crossview_verified": True,
                }
                counts["crossview_holdout_sam2"] += 1
                if options.verify_referenced_files:
                    for rel in (copied["mask"], copied.get("lidar_depth")):
                        if isinstance(rel, str) and not (refinement_root / rel).is_file():
                            missing_paths.append(str(refinement_root / rel))
                instances.append(copied)
            if not instances:
                continue
            copied_camera = {
                key: value for key, value in camera.items()
                if key not in {"instances", "union_mask", "union_mask_fraction"}
            }
            copied_camera["camera_id"] = camera_id
            copied_camera["instances"] = instances
            cameras.append(copied_camera)
        if cameras:
            selected_frames.append({
                "reference_frame_index": reference,
                "reference_timestamp_end_us": int(frame["reference_timestamp_end_us"]),
                "split": split,
                "cameras": cameras,
            })
    if missing_paths:
        preview = "\n".join(sorted(set(missing_paths))[:10])
        raise FileNotFoundError(f"selected supervision references missing files (first 10):\n{preview}")
    if not counts.get("train_source"):
        raise RuntimeError("the selection contains no original-mask training observation")
    if not counts.get("crossview_holdout_sam2"):
        raise RuntimeError("the selection lost every cross-view accepted SAM2 holdout observation")

    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "complete",
        "source_dataset_manifest": str(source_path),
        "source_dataset_manifest_sha256": _sha256(source_path),
        "refinement_manifest": str(refinement_path),
        "refinement_manifest_sha256": _sha256(refinement_path),
        "crossview_reports": report_metadata,
        "roots": {"source_dataset": str(source_root), "refinement": str(refinement_root)},
        "ncore_path": source["ncore_path"],
        "chunk_index": int(source["chunk_index"]),
        "camera_model": source["camera_model"],
        "camera_ids": source["camera_ids"],
        "tracks": [tracks[track_id] for track_id in options.track_ids],
        "accepted_sam2_observations": [
            {"camera_id": camera, "reference_frame_index": reference, "track_id": track}
            for camera, reference, track in sorted(accepted_sam2)
        ],
        "counts": {**counts, "selected_frames": len(selected_frames), "crossview_holdout_references": sorted(holdout_references)},
        "frames": selected_frames,
        "policy": {
            "original_source_masks": "retained as raw training/validation supervision",
            "sam2_masks": "only exact-crossview accepted entries are retained, exclusively as crossview_holdout",
            "holdout_grouping": "all selected actor observations at an accepted SAM2 reference frame are excluded from optimisation",
            "static_ply": "not referenced; it must not be used as a background target or occlusion truth by a future trainer",
        },
        "limitations": [
            "This manifest is training supervision only; it is not a dynamic Gaussian asset or a rendered sequence.",
            "SAM2 acceptance is sparse (currently only track67 frames 28 and 42); it is a cross-camera holdout, not pseudo-label training data.",
            "Only rigid actors are selected in this first pass. Deformable people and riders require a separate temporal/deformation representation.",
        ],
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    print(
        "Wrote trusted dynamic-layer supervision: "
        f"{manifest_path} (train_source={counts['train_source']}, "
        f"validation_source={counts['validation_source']}, "
        f"crossview_holdout_sam2={counts['crossview_holdout_sam2']})",
        flush=True,
    )
    return manifest_path
