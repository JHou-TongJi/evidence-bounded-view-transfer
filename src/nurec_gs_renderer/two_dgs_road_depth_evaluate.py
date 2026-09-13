"""Evaluate sparse held-out road LiDAR depth against an official 2DGS model.

This evaluator intentionally scores only proxy-camera depth.  It is a gate
for deciding whether a depth-supervised static model is worth target-view
rendering; it does not claim native FTheta/rolling-shutter validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace


@dataclass(frozen=True)
class TwoDgsRoadDepthEvaluateOptions:
    two_dgs_root: Path
    source_path: Path
    model_path: Path
    output: Path
    iteration: int = -1
    split: str = "holdout"
    device: str = "cuda"
    max_views: int | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "holdout"}:
            raise ValueError("split must be train or holdout")
        if self.max_views is not None and self.max_views <= 0:
            raise ValueError("max_views must be positive when supplied")


def evaluate_two_dgs_road_depth(options: TwoDgsRoadDepthEvaluateOptions) -> Path:
    """Write metric camera-z errors for a stored 2DGS iteration."""
    import torch

    root = options.two_dgs_root.resolve()
    if not (root / "scene").is_dir() or not (root / "gaussian_renderer").is_dir():
        raise FileNotFoundError(f"not an official 2DGS checkout: {root}")
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite road-depth evaluation: {options.output}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scene import GaussianModel, Scene
    from gaussian_renderer import render

    dataset = SimpleNamespace(
        sh_degree=3, source_path=str(options.source_path.resolve()), model_path=str(options.model_path.resolve()),
        images="images", resolution=-1, white_background=False, data_device=options.device, eval=True,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=options.iteration, shuffle=False)
    iteration = int(scene.loaded_iter)
    background = torch.zeros(3, dtype=torch.float32, device=options.device)
    cameras = [camera for camera in (scene.getTrainCameras() + scene.getTestCameras())
               if camera.road_depth is not None and camera.road_depth_split == options.split]
    cameras.sort(key=lambda camera: camera.image_name)
    if options.max_views is not None:
        cameras = cameras[:options.max_views]
    if not cameras:
        raise ValueError(f"no {options.split} road depth targets are attached to this proxy dataset")
    absolute_errors: list[torch.Tensor] = []
    relative_errors: list[torch.Tensor] = []
    per_view: list[dict[str, object]] = []
    with torch.no_grad():
        for ordinal, camera in enumerate(cameras, start=1):
            rendered = render(camera, gaussians, pipe, background)
            prediction = rendered["surf_depth"].clamp_min(0.25)
            target = camera.road_depth.to(options.device, non_blocking=True)
            weight = camera.road_depth_confidence.to(options.device, non_blocking=True)
            valid = (weight > 0.0) & (target > 0.25)
            if not bool(valid.any()):
                continue
            absolute = torch.abs(prediction[valid] - target[valid])
            relative = absolute / target[valid]
            absolute_errors.append(absolute.cpu())
            relative_errors.append(relative.cpu())
            per_view.append({"image": camera.image_name, "pixels": int(valid.sum().item()),
                             "mae_m": float(absolute.mean().item()), "median_m": float(absolute.median().item()),
                             "relative_mae": float(relative.mean().item())})
            if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(cameras):
                print(f"Road depth evaluate [{ordinal}/{len(cameras)}] {camera.image_name} mae={absolute.mean().item():.4f}m", flush=True)
    if not absolute_errors:
        raise ValueError("selected road depth targets contain no valid sparse pixels")
    abs_all = torch.cat(absolute_errors)
    rel_all = torch.cat(relative_errors)
    report = {
        "schema": "ncore-2dgs-road-depth-evaluation", "version": 1,
        "two_dgs_root": str(root), "source_path": str(options.source_path.resolve()),
        "model_path": str(options.model_path.resolve()), "iteration": iteration, "split": options.split,
        "views": len(per_view), "pixels": int(abs_all.numel()),
        "mae_m": float(abs_all.mean().item()), "median_m": float(abs_all.median().item()),
        "p90_m": float(torch.quantile(abs_all, 0.90).item()),
        "relative_mae": float(rel_all.mean().item()), "per_view": per_view,
        "limitations": [
            "Sparse target depth is raw LiDAR camera-z in the mid-exposure pinhole proxy, not dense geometry truth.",
            "This is depth-only holdout: the same RGB images may remain part of static RGB supervision.",
            "Passing this metric does not validate native FTheta/rolling-shutter or target-L4 novel-view quality.",
        ],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote 2DGS road depth evaluation: {options.output}", flush=True)
    return options.output
