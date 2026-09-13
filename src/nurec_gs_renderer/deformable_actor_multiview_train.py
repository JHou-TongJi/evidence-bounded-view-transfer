"""A small visibility-aware canonical neural surface pilot for paired people.

This is deliberately a proxy trainer: its rays are the exact virtual pinhole
tiles produced from NCore FTheta data and its actor-local cameras are derived
from recorded midpoint poses.  Unlike the old 2-D crop model it has a shared
3-D density/transmittance field, so an entire physical camera can be held out.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .deformable_actor_train import alpha_metrics, ring_mask
from .street_dataset import pinhole_intrinsics, tile_to_source_pose

SCHEMA = "ncore-deformable-actor-multiview-neural-surface"


@dataclass(frozen=True)
class DeformableActorMultiviewTrainOptions:
    dataset: Path
    output: Path
    iterations: int = 1_000
    device: str = "cuda"
    hidden_dim: int = 64
    samples_per_ray: int = 24
    rays_per_step: int = 1_024
    learning_rate: float = 2e-3
    validation_every: int = 100
    cuboid_margin_m: float = .20
    wide_ring_radius: int = 12
    max_holdout_rgb_mae: float = .12
    min_holdout_iou: float = .45
    max_holdout_ring_alpha: float = .20

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.hidden_dim < 16 or self.samples_per_ray < 8 or self.rays_per_step < 64 or self.learning_rate <= 0 or self.validation_every <= 0 or self.cuboid_margin_m < 0 or self.wide_ring_radius < 4:
            raise ValueError("invalid multiview neural-surface options")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError("JSON object expected")
    return value


def camera_to_actor(record: dict[str, Any]) -> np.ndarray:
    root = np.asarray(record["root_to_world"], np.float64); camera = np.asarray(record["source_camera_to_world_midpoint"], np.float64)
    if root.shape != (4, 4) or camera.shape != (4, 4): raise ValueError("root/camera pose must be 4x4")
    return np.linalg.inv(root) @ camera @ tile_to_source_pose(float(record["tile_yaw_deg"]), float(record["tile_pitch_deg"]))


def _load(options: DeformableActorMultiviewTrainOptions) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    root = options.dataset.resolve(); manifest = _read(root / "ncore_deformable_actor_multiview_manifest.json")
    if manifest.get("schema") != "ncore-deformable-actor-multiview-dataset" or manifest.get("status") != "complete": raise ValueError("dataset must be a completed deformable multiview dataset")
    cameras = [str(v) for v in manifest["camera_ids"]]
    if len(cameras) != 2: raise ValueError("neural-surface pilot requires exactly two physical cameras")
    rows: list[dict[str, Any]] = []
    for record in manifest["records"]:
        with Image.open(root / str(record["image"])) as image:
            rgba = np.asarray(image.convert("RGBA"), np.float32) / 255.0
        rows.append({**record, "rgb": rgba[..., :3], "mask": rgba[..., 3] >= .5, "camera_to_actor": camera_to_actor(record)})
    # Larger source-mask support becomes the only training camera.  The other
    # physical camera is fully held out, including its nominal train frames.
    support = {camera: sum(int(row["mask"].sum()) for row in rows if row["physical_camera_id"] == camera) for camera in cameras}
    train_camera = max(cameras, key=lambda camera: (support[camera], camera)); holdout_camera = next(camera for camera in cameras if camera != train_camera)
    if sum(row["physical_camera_id"] == train_camera and row["split"] == "train" for row in rows) < 2 or not any(row["physical_camera_id"] == holdout_camera for row in rows): raise RuntimeError("insufficient automatic camera train/holdout observations")
    return manifest, rows, train_camera, holdout_camera


def _rays(height: int, width: int, intrinsic: np.ndarray, c2a: np.ndarray, indices: "Any") -> tuple["Any", "Any"]:
    import torch
    xx = (indices % width).float(); yy = torch.div(indices, width, rounding_mode="floor").float()
    directions = torch.stack(((xx - float(intrinsic[0, 2])) / float(intrinsic[0, 0]), (yy - float(intrinsic[1, 2])) / float(intrinsic[1, 1]), torch.ones_like(xx)), dim=-1)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    rotation = torch.as_tensor(c2a[:3, :3], dtype=torch.float32, device=indices.device); origin = torch.as_tensor(c2a[:3, 3], dtype=torch.float32, device=indices.device)
    return origin.expand(len(indices), 3), directions @ rotation.T


class _NeuralSurface:
    @staticmethod
    def create(hidden: int, camera_count: int, bounds: np.ndarray) -> "Any":
        import torch
        from torch import nn
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__(); self.embed = nn.Embedding(camera_count, 8); self.register_buffer("bounds", torch.as_tensor(bounds, dtype=torch.float32))
                self.density = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                self.color = nn.Sequential(nn.Linear(4 + 3 + 8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
                nn.init.constant_(self.density[-1].bias, -3.0)
            def forward(self, points: "Any", time: "Any", direction: "Any", camera: "Any") -> tuple["Any", "Any"]:
                import torch
                import torch.nn.functional as functional
                feature = torch.cat((points, time[:, None]), -1); raw_sigma = functional.softplus(self.density(feature)[..., 0])
                # A person cannot occupy arbitrary distance along a ray.  The
                # soft cuboid support leaves room for articulated limbs while
                # suppressing the volume halo observed in the first pilot.
                support = torch.sigmoid((self.bounds - points.abs()) * 16.0).prod(dim=-1)
                sigma = raw_sigma * support
                colour = torch.sigmoid(self.color(torch.cat((feature, direction, self.embed(camera)), -1)))
                return sigma, colour
        return Module()


def _render(model: "Any", origins: "Any", directions: "Any", time: float, camera: int, samples: int) -> tuple["Any", "Any"]:
    import torch
    depths = torch.linspace(.25, 14.0, samples, device=origins.device); delta = float(depths[1] - depths[0])
    points = origins[:, None, :] + directions[:, None, :] * depths[None, :, None]; count = origins.shape[0]
    expanded_time = torch.full((count * samples,), time, device=origins.device); expanded_camera = torch.full((count * samples,), camera, dtype=torch.long, device=origins.device)
    sigma, colour = model(points.reshape(-1, 3), expanded_time, directions[:, None, :].expand(-1, samples, -1).reshape(-1, 3), expanded_camera)
    sigma, colour = sigma.reshape(count, samples), colour.reshape(count, samples, 3)
    alpha = 1.0 - torch.exp(-sigma * delta); transmittance = torch.cumprod(torch.cat((torch.ones((count, 1), device=origins.device), 1.0 - alpha + 1e-8), 1), 1)[:, :-1]; weights = transmittance * alpha
    return (weights[..., None] * colour).sum(1), weights.sum(1)


def train_deformable_actor_multiview(options: DeformableActorMultiviewTrainOptions) -> Path:
    import torch
    import torch.nn.functional as functional
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty multiview training output: {output}")
    manifest, rows, train_camera, holdout_camera = _load(options); height, width = int(manifest["height"]), int(manifest["width"]); intrinsic = pinhole_intrinsics(width, height, float(manifest["horizontal_fov_deg"])); camera_ids = [str(v) for v in manifest["camera_ids"]]; camera_index = {v:i for i,v in enumerate(camera_ids)}
    timestamps = np.asarray([int(row["timestamp_us"]) for row in rows]); start, span = int(timestamps.min()), max(int(timestamps.max()-timestamps.min()), 1)
    for row in rows: row["time"] = float((int(row["timestamp_us"])-start)/span)
    train = [row for row in rows if row["physical_camera_id"] == train_camera and row["split"] == "train"]
    dimensions = np.asarray(manifest.get("actor_dimensions_lwh_m"), dtype=np.float32)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0): raise ValueError("multiview dataset lacks valid actor_dimensions_lwh_m; rebuild it")
    bounds = dimensions * .5 + options.cuboid_margin_m
    output.mkdir(parents=True, exist_ok=True); device = torch.device(options.device); model = _NeuralSurface.create(options.hidden_dim, len(camera_ids), bounds).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    best: tuple[float,int,dict[str,float]]|None = None
    for step in range(1, options.iterations+1):
        row = train[(step-1) % len(train)]; mask = torch.from_numpy(row["mask"].reshape(-1)); mask_2d = mask.reshape(height,width); foreground = torch.nonzero(mask, as_tuple=False).flatten(); narrow_mask = ring_mask(mask_2d, radius=3); wide_mask = ring_mask(mask_2d, radius=options.wide_ring_radius) & ~narrow_mask; narrow = torch.nonzero(narrow_mask.reshape(-1), as_tuple=False).flatten(); wide = torch.nonzero(wide_mask.reshape(-1), as_tuple=False).flatten(); exterior = torch.nonzero(~mask, as_tuple=False).flatten()
        fg = foreground[torch.randint(len(foreground),(options.rays_per_step//2,))]; narrow_bg = narrow[torch.randint(len(narrow),(options.rays_per_step//4,))]; wide_bg = wide[torch.randint(len(wide),(options.rays_per_step//8,))]; global_bg = exterior[torch.randint(len(exterior),(options.rays_per_step-len(fg)-len(narrow_bg)-len(wide_bg),))]; bg = torch.cat((narrow_bg, wide_bg, global_bg)); indices = torch.cat((fg,bg)).to(device)
        origins, directions = _rays(height,width,intrinsic,row["camera_to_actor"],indices); predicted_rgb,predicted_alpha = _render(model,origins,directions,row["time"],camera_index[row["physical_camera_id"]],options.samples_per_ray)
        target_rgb = torch.from_numpy(row["rgb"].reshape(-1,3))[fg.cpu()].to(device); target_alpha = torch.cat((torch.ones(len(fg),device=device),torch.zeros(len(bg),device=device)))
        rgb_loss=functional.l1_loss(predicted_rgb[:len(fg)],target_rgb); alpha_loss=functional.binary_cross_entropy(predicted_alpha,target_alpha); loss=rgb_loss+.75*alpha_loss+.001*predicted_alpha.mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % options.validation_every == 0 or step == options.iterations:
            metrics = _evaluate(model, rows, holdout_camera, camera_index, intrinsic, options, device)
            score=metrics["rgb_mae"]+.5*(1-metrics["alpha_iou_0_5"])+.5*metrics["ring_alpha"]
            print(f"Multiview neural surface [{step:05d}/{options.iterations}] loss={float(loss):.6f} rgb={float(rgb_loss):.6f} alpha={float(alpha_loss):.6f} holdout_rgb={metrics['rgb_mae']:.6f} iou={metrics['alpha_iou_0_5']:.4f} ring={metrics['ring_alpha']:.4f}",flush=True)
            if best is None or score < best[0]: best=(score,step,metrics); torch.save({"schema":SCHEMA,"track_id":manifest["track_id"],"train_camera":train_camera,"holdout_camera":holdout_camera,"model":model.state_dict(),"iteration":step,"intrinsic":intrinsic},output/"best.pt")
    assert best is not None; passed=best[2]["rgb_mae"]<=options.max_holdout_rgb_mae and best[2]["alpha_iou_0_5"]>=options.min_holdout_iou and best[2]["ring_alpha"]<=options.max_holdout_ring_alpha
    report={"schema":SCHEMA,"status":"complete","dataset":str(options.dataset.resolve()),"track_id":manifest["track_id"],"train_camera":train_camera,"holdout_camera":holdout_camera,"density_bounds_half_extent_m":bounds.tolist(),"wide_ring_radius":options.wide_ring_radius,"best_iteration":best[1],"holdout":best[2],"passed_cross_camera_holdout":passed,"limitations":["Virtual pinhole midpoint rays proxy native FTheta/rolling shutter; this is not yet a target-L4 renderer.","Passing a fully held-out source camera is required before any target-view experiment."]}; (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return output/"best.pt"


def _evaluate(model: "Any", rows: list[dict[str,Any]], holdout_camera: str, camera_index: dict[str,int], intrinsic: np.ndarray, options: DeformableActorMultiviewTrainOptions, device: "Any") -> dict[str,float]:
    import torch
    values=[]
    for row in rows:
        if row["physical_camera_id"] != holdout_camera: continue
        height, width = row["mask"].shape
        total = height * width; colours=[]; alphas=[]
        with torch.inference_mode():
            for begin in range(0,total,2048):
                indices=torch.arange(begin,min(begin+2048,total),device=device); origins,directions=_rays(height,width,intrinsic,row["camera_to_actor"],indices); rgb,alpha=_render(model,origins,directions,row["time"],camera_index[holdout_camera],options.samples_per_ray); colours.append(rgb.cpu()); alphas.append(alpha.cpu())
        rgb=torch.cat(colours).reshape(height,width,3).numpy(); alpha=torch.cat(alphas).reshape(height,width).numpy(); mask=row["mask"]; ring=ring_mask(torch.from_numpy(mask)).numpy(); values.append({"rgb_mae":float(np.abs(rgb[mask]-row["rgb"][mask]).mean()),**alpha_metrics(alpha,mask,ring)})
    return {key:float(np.mean([value[key] for value in values])) for key in ("rgb_mae","alpha_iou_0_5","alpha_area_ratio","ring_alpha")}
