"""Build a clip-agnostic registry of independently validated actor experts.

The registry is deliberately a selection artifact, not a new model format. It
allows target rendering to consume only per-source-camera experts that passed
their own temporal holdout, while preserving source windows and metric
provenance for later audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SCHEMA = "ncore-2dgs-actor-expert-registry"


@dataclass(frozen=True)
class ActorExpertRegistryOptions:
    batch_manifests: tuple[Path, ...]
    output: Path
    expert_specs: tuple[Path, ...] = ()
    max_masked_rgb_mae: float = .07
    min_alpha_iou: float = .82
    min_alpha_area_ratio: float = .85
    max_alpha_area_ratio: float = 1.20
    overlap_fraction: float = .50

    def __post_init__(self) -> None:
        if not self.batch_manifests and not self.expert_specs:
            raise ValueError("at least one batch manifest or expert specification is required")
        if not 0.0 < self.max_masked_rgb_mae <= 1.0 or not 0.0 <= self.min_alpha_iou <= 1.0:
            raise ValueError("invalid RGB/IoU thresholds")
        if not 0.0 < self.min_alpha_area_ratio <= self.max_alpha_area_ratio:
            raise ValueError("invalid alpha area ratio bounds")
        if not 0.0 <= self.overlap_fraction <= 1.0:
            raise ValueError("overlap_fraction must be in [0, 1]")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _window_overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return 0.0 if union <= 0 else intersection / union


def _is_valid(metrics: dict[str, Any], options: ActorExpertRegistryOptions) -> bool:
    try:
        return (
            float(metrics["masked_rgb_mae"]) <= options.max_masked_rgb_mae
            and float(metrics["alpha_iou_0_5"]) >= options.min_alpha_iou
            and options.min_alpha_area_ratio <= float(metrics["alpha_area_ratio"]) <= options.max_alpha_area_ratio
        )
    except (KeyError, TypeError, ValueError):
        return False


def _candidate_quality(candidate: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["masked_rgb_mae"]),
        -float(metrics["alpha_iou_0_5"]),
        abs(float(metrics["alpha_area_ratio"]) - 1.0),
    )


def _candidate_from_actor_dataset(
    *,
    name: str,
    dataset: Path,
    model: Path,
    iteration: int,
    metrics: dict[str, Any],
    provenance: dict[str, str],
    rejected: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Normalize one independently evaluated actor asset into a registry row."""
    dataset_path = dataset.resolve()
    actor_manifest_path = dataset_path / "ncore_2dgs_actor_manifest.json"
    try:
        actor_manifest = _read(actor_manifest_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        rejected.append({**provenance, "name": name, "reason": f"invalid_actor_dataset: {error}"})
        return None
    if actor_manifest.get("schema") != "ncore-2dgs-canonical-rigid-actor-proxy" or actor_manifest.get("status") != "complete":
        rejected.append({**provenance, "name": name, "reason": "incomplete_actor_dataset"})
        return None
    records = [value for value in actor_manifest.get("records", []) if isinstance(value, dict) and value.get("timestamp_us") is not None]
    cameras = sorted({str(value.get("source_camera")) for value in records if value.get("source_camera")})
    timestamps = [int(value["timestamp_us"]) for value in records]
    if len(cameras) != 1 or not timestamps:
        rejected.append({**provenance, "name": name, "reason": "actor_dataset_is_not_single_source_camera"})
        return None
    return {
        "name": name, "track_id": int(actor_manifest["track_id"]),
        "label": actor_manifest.get("label"), "source_camera": cameras[0],
        "timestamp_start_us": min(timestamps), "timestamp_end_us": max(timestamps),
        "actor_dataset": str(dataset_path), "model_path": str(model.resolve()),
        "iteration": int(iteration), "metrics": metrics, **provenance,
    }


def _candidates_from_spec(
    spec_path: Path,
    options: ActorExpertRegistryOptions,
    rejected: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Read externally supplied historical evaluations without trusting their text.

    Historical experiments predate the batch manifest.  A small declarative
    specification keeps their dataset, model and *evaluation report* tied
    together, then subjects them to the exact same registry thresholds as
    newer batch experts.  It deliberately cannot promote an old model merely
    because its RGB score is low while its silhouette is over-expanded.
    """
    path = spec_path.resolve()
    spec = _read(path)
    if spec.get("schema") != "ncore-2dgs-actor-expert-spec" or spec.get("status") != "complete":
        raise ValueError(f"not a completed actor expert specification: {path}")
    ncore_path = spec.get("ncore_path")
    if not ncore_path:
        raise ValueError(f"expert specification has no ncore_path: {path}")
    candidates: list[dict[str, Any]] = []
    for entry in spec.get("experts", []):
        if not isinstance(entry, dict):
            rejected.append({"expert_spec": str(path), "reason": "non_object_entry"})
            continue
        name, dataset, model, report_path = (entry.get("name"), entry.get("actor_dataset"), entry.get("model_path"), entry.get("evaluation_report"))
        provenance = {"expert_spec": str(path), "evaluation_report": str(Path(str(report_path)).resolve()) if report_path else ""}
        if not name or not dataset or not model or not report_path:
            rejected.append({**provenance, "name": name, "reason": "incomplete_expert_spec_entry"})
            continue
        try:
            report = _read(Path(str(report_path)).resolve())
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            rejected.append({**provenance, "name": str(name), "reason": f"invalid_evaluation_report: {error}"})
            continue
        metrics = report.get("metrics")
        if report.get("schema") != "ncore-2dgs-canonical-actor-evaluation" or report.get("status") != "complete" or not isinstance(metrics, dict):
            rejected.append({**provenance, "name": str(name), "reason": "incomplete_evaluation_report"})
            continue
        if not _is_valid(metrics, options):
            rejected.append({**provenance, "name": str(name), "reason": "does_not_meet_registry_thresholds", "metrics": metrics})
            continue
        candidate = _candidate_from_actor_dataset(
            name=str(name), dataset=Path(str(dataset)), model=Path(str(model)),
            iteration=int(entry.get("iteration", report.get("iteration", -1))), metrics=metrics,
            provenance=provenance, rejected=rejected,
        )
        if candidate is None:
            continue
        if int(report.get("track_id", -1)) != candidate["track_id"]:
            rejected.append({**provenance, "name": str(name), "reason": "evaluation_track_does_not_match_actor_dataset"})
            continue
        candidates.append(candidate)
    return str(Path(str(ncore_path)).resolve()), candidates


def build_actor_expert_registry(options: ActorExpertRegistryOptions) -> Path:
    """Merge accepted batch outputs without blending overlapping source experts."""
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ncore_paths: set[str] = set()
    for manifest_path in options.batch_manifests:
        batch_path = manifest_path.resolve()
        batch = _read(batch_path)
        if batch.get("schema") != "ncore-2dgs-rigid-actor-batch":
            raise ValueError(f"not a rigid actor batch manifest: {batch_path}")
        ncore_path = batch.get("ncore_path")
        if not ncore_path:
            raise ValueError(f"batch manifest has no ncore_path: {batch_path}")
        ncore_paths.add(str(Path(str(ncore_path)).resolve()))
        for record in batch.get("train", []):
            if not isinstance(record, dict) or record.get("status") != "accepted":
                continue
            metrics = record.get("metrics")
            dataset = record.get("dataset")
            model = record.get("model")
            if not isinstance(metrics, dict) or not dataset or not model:
                rejected.append({"batch_manifest": str(batch_path), "name": record.get("name"), "reason": "incomplete accepted record"})
                continue
            if not _is_valid(metrics, options):
                rejected.append({"batch_manifest": str(batch_path), "name": record.get("name"), "reason": "does_not_meet_registry_thresholds", "metrics": metrics})
                continue
            candidate = _candidate_from_actor_dataset(
                name=str(record["name"]), dataset=Path(str(dataset)), model=Path(str(model)),
                iteration=int(record.get("iteration", -1)), metrics=metrics,
                provenance={"batch_manifest": str(batch_path)}, rejected=rejected,
            )
            if candidate is not None:
                candidates.append(candidate)
    for spec_path in options.expert_specs:
        ncore_path, spec_candidates = _candidates_from_spec(spec_path, options, rejected)
        ncore_paths.add(ncore_path)
        candidates.extend(spec_candidates)
    if len(ncore_paths) != 1:
        raise ValueError("all batch manifests must refer to the same NCore clip")
    # Competing models for the same actor/source time interval are selected by
    # validation quality. Disjoint windows remain separate experts, so a clip
    # can switch evidence over time without blending incompatible surfaces.
    candidates.sort(key=lambda value: (value["track_id"], value["source_camera"], *_candidate_quality(value)))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        window = (candidate["timestamp_start_us"], candidate["timestamp_end_us"])
        conflicting = [item for item in selected if item["track_id"] == candidate["track_id"] and item["source_camera"] == candidate["source_camera"] and _window_overlap(window, (item["timestamp_start_us"], item["timestamp_end_us"])) >= options.overlap_fraction]
        if conflicting:
            rejected.append({"name": candidate["name"], "batch_manifest": candidate["batch_manifest"], "reason": "overlapping_lower_quality_expert", "kept": conflicting[0]["name"], "metrics": candidate["metrics"]})
            continue
        candidate["registry_name"] = f"t{candidate['track_id']:03d}_{candidate['source_camera']}_{candidate['timestamp_start_us']}"
        selected.append(candidate)
    if not selected:
        raise RuntimeError("no actor experts meet the registry thresholds")
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema": SCHEMA, "version": 1, "status": "complete", "ncore_path": next(iter(ncore_paths)),
        "selection": {"max_masked_rgb_mae": options.max_masked_rgb_mae, "min_alpha_iou": options.min_alpha_iou, "alpha_area_ratio": [options.min_alpha_area_ratio, options.max_alpha_area_ratio], "overlap_fraction": options.overlap_fraction},
        "experts": selected, "rejected": rejected,
        "limitations": [
            "Selection is based on source-camera temporal holdout only; it is not target-view ground truth.",
            "Disjoint experts are retained per track/source camera; target rendering must still use view/time visibility gates.",
            "Deformable people are excluded because this registry only accepts rigid actor datasets.",
        ],
    }
    output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Wrote actor expert registry: {output} ({len(selected)} experts, {len(rejected)} rejected)", flush=True)
    return output
