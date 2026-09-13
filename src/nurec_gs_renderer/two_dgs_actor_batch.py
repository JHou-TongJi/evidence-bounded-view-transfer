"""Batch planning, construction, and validation for rigid 2DGS actor experts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from .two_dgs_actor_dataset import TwoDgsActorDatasetOptions, build_two_dgs_actor_dataset
from .two_dgs_actor_evaluate import TwoDgsActorEvaluateOptions, evaluate_two_dgs_actor


SCHEMA = "ncore-2dgs-rigid-actor-batch"


@dataclass(frozen=True)
class ActorBatchOptions:
    ncore_path: Path
    dynamic_dataset_manifest: Path
    output: Path
    phase: str
    two_dgs_root: Path | None = None
    device: str = "cuda"
    width: int = 480
    height: int = 270
    horizontal_fov_deg: float = 45.0
    min_observations: int = 18
    min_source_mask_pixels: int = 4_000
    max_experts_per_track: int = 2
    max_tracks: int | None = None
    max_observations_per_expert: int = 100
    quality_window_frames: int | None = None
    iterations: int = 1_750
    gpu_id: str | None = None
    min_alpha_iou: float = .70
    max_masked_rgb_mae: float = .10
    min_alpha_area_ratio: float = .75
    max_alpha_area_ratio: float = 1.30

    def __post_init__(self) -> None:
        if self.phase not in {"plan", "build", "train"}:
            raise ValueError("phase must be plan, build or train")
        if self.width <= 0 or self.height <= 0 or self.min_observations < 3 or self.min_source_mask_pixels <= 0:
            raise ValueError("invalid batch construction thresholds")
        if self.max_experts_per_track <= 0 or self.max_observations_per_expert <= 1 or self.iterations <= 0:
            raise ValueError("invalid batch limits")
        if self.quality_window_frames is not None and self.quality_window_frames < 8:
            raise ValueError("quality_window_frames must be at least 8 when supplied")
        if self.max_tracks is not None and self.max_tracks <= 0:
            raise ValueError("max_tracks must be positive when supplied")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def select_rigid_actor_candidates(
    dynamic: dict[str, Any],
    *,
    min_observations: int,
    min_source_mask_pixels: int,
    max_experts_per_track: int,
    max_tracks: int | None,
    quality_window_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Select only rigid source-camera tracks with enough original mask evidence."""
    tracks = {
        int(value["track_id"]): value
        for value in dynamic.get("tracks", [])
        if isinstance(value, dict) and value.get("actor_family") == "rigid" and "track_id" in value
    }
    grouped: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for frame in dynamic.get("frames", []):
        reference = int(frame.get("reference_frame_index", -1))
        for camera in frame.get("cameras", []):
            camera_id = str(camera.get("camera_id", ""))
            for instance in camera.get("instances", []):
                track_id = int(instance.get("track_id", -1))
                pixels = int(instance.get("mask_pixels", 0))
                if track_id in tracks and reference >= 0 and camera_id and pixels >= min_source_mask_pixels:
                    grouped.setdefault((track_id, camera_id), []).append((reference, pixels))
    by_track: dict[int, list[dict[str, Any]]] = {}
    for (track_id, camera_id), observations in grouped.items():
        observations.sort()
        selected_observations = observations
        reference_start = observations[0][0]
        reference_end = observations[-1][0] + 1
        if quality_window_frames is not None:
            # Prefer a compact high-coverage window over uniformly sampling
            # a long trajectory.  The latter mixes materially different
            # actor appearance/view directions and was the direct cause of
            # the weak batch-v1 experts.
            best: tuple[int, int, int, list[tuple[int, int]]] | None = None
            for start, _ in observations:
                end = start + quality_window_frames
                window = [(reference, pixels) for reference, pixels in observations if start <= reference < end]
                if len(window) < min_observations:
                    continue
                score = (sum(pixels for _, pixels in window), len(window), -start, window)
                if best is None or score[:3] > best[:3]:
                    best = score
            if best is None:
                continue
            _, _, negative_start, selected_observations = best
            reference_start = -negative_start
            reference_end = reference_start + quality_window_frames
        if len(selected_observations) < min_observations:
            continue
        refs = [value[0] for value in selected_observations]
        pixels = np.asarray([value[1] for value in selected_observations], np.int64)
        by_track.setdefault(track_id, []).append({
            "track_id": track_id,
            "label": str(tracks[track_id].get("label", "unknown")),
            "camera_id": camera_id,
            "reference_start": reference_start,
            "reference_end": reference_end,
            "source_observations": len(selected_observations),
            "full_source_observations": len(observations),
            "source_mask_pixels_median": float(np.median(pixels)),
            "source_mask_pixels_p90": float(np.percentile(pixels, 90)),
            "quality_window_frames": quality_window_frames,
        })
    selected: list[dict[str, Any]] = []
    track_rank = sorted(by_track, key=lambda track: max(value["source_observations"] for value in by_track[track]), reverse=True)
    if max_tracks is not None:
        track_rank = track_rank[:max_tracks]
    for track_id in track_rank:
        ranking = sorted(by_track[track_id], key=lambda value: (value["source_observations"], value["source_mask_pixels_median"]), reverse=True)
        for item in ranking[:max_experts_per_track]:
            item["name"] = f"track{track_id:03d}_{item['camera_id']}"
            selected.append(item)
    return selected


def _plan_path(output: Path) -> Path:
    return output / "batch_manifest.json"


def _write_plan(options: ActorBatchOptions) -> dict[str, Any]:
    dynamic = _read(options.dynamic_dataset_manifest.resolve())
    if dynamic.get("schema") != "ncore-dynamic-reconstruction-dataset" or dynamic.get("status") != "complete":
        raise ValueError("dynamic_dataset_manifest must be a completed dynamic reconstruction dataset")
    if Path(str(dynamic.get("ncore_path"))).resolve() != options.ncore_path.resolve():
        raise ValueError("dynamic dataset NCore path does not match")
    candidates = select_rigid_actor_candidates(
        dynamic, min_observations=options.min_observations, min_source_mask_pixels=options.min_source_mask_pixels,
        max_experts_per_track=options.max_experts_per_track, max_tracks=options.max_tracks,
        quality_window_frames=options.quality_window_frames,
    )
    plan = {
        "schema": SCHEMA, "version": 1, "status": "planned",
        "ncore_path": str(options.ncore_path.resolve()),
        "dynamic_dataset_manifest": str(options.dynamic_dataset_manifest.resolve()),
        "selection": {"min_observations": options.min_observations, "min_source_mask_pixels": options.min_source_mask_pixels, "max_experts_per_track": options.max_experts_per_track, "max_tracks": options.max_tracks, "quality_window_frames": options.quality_window_frames},
        "candidates": candidates, "builds": [], "train": [],
        "limitations": [
            "Only rigid tracks are eligible; people and other deformable actors are deliberately excluded.",
            "Candidate selection uses source-mask evidence only and is not an acceptance decision.",
            "Every trained expert must pass its own time-held-out RGB/alpha gate before target rendering.",
        ],
    }
    options.output.mkdir(parents=True, exist_ok=True)
    _plan_path(options.output).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote rigid actor batch plan: {_plan_path(options.output)} ({len(candidates)} candidates)", flush=True)
    return plan


def _load_or_plan(options: ActorBatchOptions) -> dict[str, Any]:
    path = _plan_path(options.output)
    return _read(path) if path.is_file() else _write_plan(options)


def _write_batch(options: ActorBatchOptions, value: dict[str, Any]) -> None:
    _plan_path(options.output).write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_rigid_actor_batch(options: ActorBatchOptions) -> dict[str, Any]:
    plan = _load_or_plan(options)
    existing = {str(value["name"]): value for value in plan.get("builds", [])}
    builds: list[dict[str, Any]] = []
    for index, candidate in enumerate(plan["candidates"], start=1):
        name = str(candidate["name"])
        dataset = options.output / "datasets" / name
        record = dict(candidate)
        record["dataset"] = str(dataset)
        completed_manifest = dataset / "ncore_2dgs_actor_manifest.json"
        if completed_manifest.is_file():
            manifest = _read(completed_manifest)
            record.update({"status": "complete", "train_observations": sum(value["split"] == "train" for value in manifest["records"]), "holdout_observations": sum(value["split"] == "holdout" for value in manifest["records"])})
        elif dataset.exists() and any(dataset.iterdir()):
            # Never overwrite a partial output.  It may be inspected or
            # recovered later, but cannot safely become a training asset.
            record.update({"status": "rejected_partial_output", "reason": f"non-empty partial dataset: {dataset}"})
        else:
            try:
                build_two_dgs_actor_dataset(TwoDgsActorDatasetOptions(
                    ncore_path=options.ncore_path, dynamic_dataset_manifest=options.dynamic_dataset_manifest,
                    output=dataset, track_id=int(candidate["track_id"]), camera_ids=(str(candidate["camera_id"]),),
                    reference_start=int(candidate["reference_start"]), reference_end=int(candidate["reference_end"]),
                    width=options.width, height=options.height, horizontal_fov_deg=options.horizontal_fov_deg,
                    minimum_mask_pixels=max(1_000, options.min_source_mask_pixels // 4), holdout_every=6,
                    initial_surface_spacing_m=.25, max_observations=options.max_observations_per_expert,
                    rectify_device=options.device,
                ))
                manifest = _read(dataset / "ncore_2dgs_actor_manifest.json")
                record.update({"status": "complete", "train_observations": sum(value["split"] == "train" for value in manifest["records"]), "holdout_observations": sum(value["split"] == "holdout" for value in manifest["records"])})
            except Exception as error:  # keep batch auditable; a single track must not hide others
                record.update({"status": "rejected_build", "reason": str(error)})
        builds.append(record)
        # Persist each completed/rejected candidate so an interrupted CUDA
        # build can be resumed without silently reusing or overwriting data.
        plan["builds"] = builds; _write_batch(options, plan)
        print(f"Actor batch build [{index}/{len(plan['candidates'])}] {name}: {record['status']}", flush=True)
    plan["builds"] = builds; plan["status"] = "built"; _write_batch(options, plan)
    return plan


def train_rigid_actor_batch(options: ActorBatchOptions) -> dict[str, Any]:
    if options.two_dgs_root is None:
        raise ValueError("--two-dgs-root is required for phase=train")
    plan = build_rigid_actor_batch(options)
    root = options.two_dgs_root.resolve()
    if not (root / "train.py").is_file():
        raise FileNotFoundError(root / "train.py")
    results: list[dict[str, Any]] = []
    for index, build in enumerate(plan["builds"], start=1):
        record = {"name": build["name"], "dataset": build.get("dataset"), "track_id": build["track_id"], "camera_id": build["camera_id"]}
        if build.get("status") != "complete":
            record.update({"status": "skipped_build"}); results.append(record)
            plan["train"] = results; _write_batch(options, plan); continue
        if int(build.get("train_observations", 0)) < 12 or int(build.get("holdout_observations", 0)) < 3:
            record.update({"status": "skipped_insufficient_holdout"}); results.append(record)
            plan["train"] = results; _write_batch(options, plan); continue
        model = options.output / "models" / str(build["name"])
        evaluation = options.output / "evaluation" / str(build["name"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{root}:{root / 'submodules' / 'simple-knn'}" + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else "")
        environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        if options.gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = options.gpu_id
        command = [sys.executable, "train.py", "-s", str(build["dataset"]), "-m", str(model), "--iterations", str(options.iterations), "--actor_only", "--lambda_actor_silhouette", ".10", "--lambda_dssim", "0", "--lambda_dist", "0", "--lambda_normal", "0", "--densify_from_iter", "100", "--densify_until_iter", "900", "--opacity_reset_interval", "300", "--test_iterations", str(options.iterations), "--save_iterations", str(options.iterations), "--checkpoint_iterations", str(options.iterations)]
        completed = subprocess.run(command, cwd=root, env=environment, check=False)
        if completed.returncode != 0:
            record.update({"status": "rejected_train", "returncode": completed.returncode})
        else:
            try:
                report_path = evaluate_two_dgs_actor(TwoDgsActorEvaluateOptions(two_dgs_root=root, source_path=Path(str(build["dataset"])), model_path=model, output=evaluation, iteration=options.iterations, device=options.device, save_previews=2))
                metrics = _read(report_path)["metrics"]
                accepted = metrics["masked_rgb_mae"] <= options.max_masked_rgb_mae and metrics["alpha_iou_0_5"] >= options.min_alpha_iou and options.min_alpha_area_ratio <= metrics["alpha_area_ratio"] <= options.max_alpha_area_ratio
                record.update({"status": "accepted" if accepted else "rejected_quality", "model": str(model), "evaluation": str(report_path), "metrics": metrics})
            except Exception as error:
                record.update({"status": "rejected_evaluate", "reason": str(error)})
        results.append(record)
        plan["train"] = results; _write_batch(options, plan)
        print(f"Actor batch train [{index}/{len(plan['builds'])}] {record['name']}: {record['status']}", flush=True)
    plan["train"] = results; plan["status"] = "trained"; _write_batch(options, plan)
    return plan


def run_actor_batch(options: ActorBatchOptions) -> Path:
    if options.phase == "plan": _write_plan(options)
    elif options.phase == "build": build_rigid_actor_batch(options)
    else: train_rigid_actor_batch(options)
    return _plan_path(options.output)
