"""Evaluate a canonical actor 2DGS only within explicit held-out actor masks."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TwoDgsActorEvaluateOptions:
    two_dgs_root: Path
    source_path: Path
    model_path: Path
    output: Path
    iteration: int = -1
    device: str = "cuda"
    save_previews: int = 3

    def __post_init__(self) -> None:
        if self.save_previews < 0:
            raise ValueError("save_previews must be non-negative")


def _import_external(root: Path) -> tuple[Any, Any, Any, Any]:
    text = str(root.resolve())
    if not (root / "scene" / "gaussian_model.py").is_file():
        raise FileNotFoundError(f"not an official 2DGS checkout: {root}")
    if text not in sys.path:
        sys.path.insert(0, text)
    from scene import GaussianModel
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import loadCam
    from gaussian_renderer import render
    return GaussianModel, readColmapSceneInfo, loadCam, render


def _checkpoint(model_path: Path, iteration: int) -> tuple[Path, int]:
    candidates: list[tuple[int, Path]] = []
    for directory in (model_path / "point_cloud").glob("iteration_*"):
        try: value = int(directory.name.removeprefix("iteration_"))
        except ValueError: continue
        point = directory / "point_cloud.ply"
        if point.is_file(): candidates.append((value, point))
    if not candidates: raise FileNotFoundError(f"no 2DGS point cloud in {model_path}")
    if iteration == -1: return max(candidates, key=lambda item: item[0])
    for value, point in candidates:
        if value == iteration: return point, value
    raise FileNotFoundError(model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")


def evaluate_two_dgs_actor(options: TwoDgsActorEvaluateOptions) -> Path:
    import torch
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"output exists: {output}")
    manifest = json.loads((options.source_path / "ncore_2dgs_actor_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ncore-2dgs-canonical-rigid-actor-proxy" or manifest.get("status") != "complete":
        raise ValueError("source_path must be a completed canonical actor 2DGS dataset")
    GaussianModel, read_scene, load_cam, render = _import_external(options.two_dgs_root)
    info = read_scene(str(options.source_path), "images", True)
    if not info.test_cameras: raise RuntimeError("actor dataset has no held-out cameras")
    point, loaded = _checkpoint(options.model_path.resolve(), options.iteration)
    model = GaussianModel(3); model.load_ply(str(point))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device=options.device)
    args = SimpleNamespace(resolution=1, data_device=options.device)
    output.mkdir(parents=True); preview = output / "previews"; preview.mkdir()
    maes: list[float] = []; ious: list[float] = []; areas: list[float] = []; records: list[dict[str, Any]] = []
    for index, camera_info in enumerate(info.test_cameras):
        camera = load_cam(args, index, camera_info, 1.0)
        if camera.gt_alpha_mask is None: raise ValueError("actor evaluation image lacks foreground alpha")
        with torch.inference_mode(): result = render(camera, model, pipe, background)
        rgb = result["render"].clamp(0.0, 1.0); alpha = result["rend_alpha"].clamp(0.0, 1.0)
        target = camera.original_image.to(options.device).clamp(0.0, 1.0); mask = camera.gt_alpha_mask.to(options.device).clamp(0.0, 1.0)
        mae = (torch.abs(rgb - target) * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
        binary = alpha >= .5; truth = mask >= .5; union = (binary | truth).sum(); iou = ((binary & truth).sum().float() / union.float()).item() if int(union) else 1.0
        maes.append(float(mae.item())); ious.append(float(iou)); areas.append(float(alpha.mean().item() / mask.mean().clamp_min(1e-6).item()))
        records.append({"image": camera_info.image_name, "masked_rgb_mae": maes[-1], "alpha_iou_0_5": ious[-1], "alpha_area_ratio": areas[-1]})
        if index < options.save_previews:
            Image.fromarray((target.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8), "RGB").save(preview / f"{index:02d}_gt.png")
            Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8), "RGB").save(preview / f"{index:02d}_render.png")
            Image.fromarray((alpha[0].cpu().numpy()*255+.5).astype(np.uint8), "L").save(preview / f"{index:02d}_alpha.png")
    report = {"schema": "ncore-2dgs-canonical-actor-evaluation", "version": 1, "status": "complete", "track_id": int(manifest["track_id"]), "iteration": loaded, "checkpoint": str(point), "holdout_count": len(records), "metrics": {"masked_rgb_mae": float(np.mean(maes)), "alpha_iou_0_5": float(np.mean(ious)), "alpha_area_ratio": float(np.mean(areas)), "positive_frames": int(sum(value > 0.0 for value in ious))}, "records": records, "limitations": ["Pinhole midpoint proxy evaluation only; not native FTheta/rolling evaluation.", "Metrics only score the actor foreground mask; no static background/depth/occlusion claim."]}
    result = output / "report.json"; result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote canonical actor 2DGS evaluation: {result} mae={report['metrics']['masked_rgb_mae']:.6f} iou={report['metrics']['alpha_iou_0_5']:.4f}", flush=True)
    return result
