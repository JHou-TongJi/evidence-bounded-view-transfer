"""Train a small time- and camera-conditioned 2D deformable actor layer.

The representation is intentionally a source-view proxy: a canonical RGBA
atlas is sampled through a learned continuous 2-D deformation field.  It can
test temporal interpolation of an articulated person without declaring a
rigid Gaussian, a complete 3-D body, or a novel target-L4 view.  A later
target renderer must be separately justified by multi-view geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "ncore-deformable-actor-dynamic-plane"
VERSION = 2


@dataclass(frozen=True)
class DeformableActorTrainOptions:
    datasets: tuple[Path, ...]
    output: Path
    iterations: int = 1_000
    device: str = "cuda"
    atlas_size: int = 256
    hidden_dim: int = 96
    actor_samples: int = 2_048
    ring_samples: int = 1_024
    ring_loss_weight: float = .75
    learning_rate: float = 2.0e-3
    validation_every: int = 100
    max_holdout_rgb_mae: float = .12
    min_holdout_alpha_iou: float = .60
    min_holdout_alpha_area_ratio: float = .70
    max_holdout_alpha_area_ratio: float = 1.30
    max_holdout_ring_alpha: float = .15

    def __post_init__(self) -> None:
        if not self.datasets or self.iterations <= 0 or self.atlas_size < 32 or self.hidden_dim < 16:
            raise ValueError("datasets/iterations/atlas/hidden dimensions are invalid")
        if self.actor_samples <= 0 or self.ring_samples <= 0 or self.ring_loss_weight < 0 or self.learning_rate <= 0 or self.validation_every <= 0:
            raise ValueError("sampling, learning rate and validation interval must be positive")
        if not 0.0 < self.max_holdout_rgb_mae <= 1.0 or not 0.0 < self.min_holdout_alpha_iou <= 1.0:
            raise ValueError("invalid holdout RGB/IoU threshold")
        if not 0.0 < self.min_holdout_alpha_area_ratio <= self.max_holdout_alpha_area_ratio or not 0.0 < self.max_holdout_ring_alpha < 1.0:
            raise ValueError("invalid holdout alpha thresholds")


def ring_mask(mask: "Any", radius: int = 3) -> "Any":
    """Return a narrow safe exterior ring for explicit background-leak loss."""
    import torch.nn.functional as functional

    if radius <= 0:
        raise ValueError("radius must be positive")
    value = mask.float()[None, None]
    dilated = functional.max_pool2d(value, kernel_size=radius * 2 + 1, stride=1, padding=radius)[0, 0] > 0.0
    return dilated & ~mask.bool()


def alpha_metrics(predicted: np.ndarray, target: np.ndarray, ring: np.ndarray) -> dict[str, float]:
    """Compute source holdout silhouette metrics without using RGB background."""
    prediction = np.asarray(predicted, np.float32); truth = np.asarray(target, dtype=bool); exterior = np.asarray(ring, dtype=bool)
    binary = prediction >= .5
    union = int((binary | truth).sum())
    intersection = int((binary & truth).sum())
    return {
        "alpha_iou_0_5": float(intersection / union) if union else 1.0,
        "alpha_area_ratio": float(binary.sum() / max(int(truth.sum()), 1)),
        "ring_alpha": float(prediction[exterior].mean()) if np.any(exterior) else 0.0,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def actor_bounds(mask: np.ndarray, *, padding: int = 16) -> tuple[int, int, int, int]:
    """Return a padded pixel box around one actor, in x0/y0/x1/y1 form."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        raise ValueError("actor mask is empty")
    height, width = mask.shape
    return (
        max(int(xx.min()) - padding, 0), max(int(yy.min()) - padding, 0),
        min(int(xx.max()) + padding + 1, width), min(int(yy.max()) + padding + 1, height),
    )


def _local_coordinates(indices: "Any", *, width: int, bounds: tuple[int, int, int, int]) -> "Any":
    """Map flattened image pixel indices to the frame's actor-local UV domain."""
    import torch

    x0, y0, x1, y1 = bounds
    xx = (indices % width).float(); yy = torch.div(indices, width, rounding_mode="floor").float()
    denominator_x = max(x1 - x0 - 1, 1); denominator_y = max(y1 - y0 - 1, 1)
    return torch.stack(((xx - x0) * (2.0 / denominator_x) - 1.0, (yy - y0) * (2.0 / denominator_y) - 1.0), dim=-1)


def _load_frames(paths: tuple[Path, ...]) -> tuple[list[dict[str, Any]], list[str], int, int, int]:
    frames: list[dict[str, Any]] = []; camera_ids: set[str] = set(); width: int | None = None; height: int | None = None; track_id: int | None = None
    for raw_path in paths:
        root = raw_path.resolve(); manifest = _read(root / "ncore_deformable_actor_manifest.json")
        if manifest.get("schema") != "ncore-deformable-actor-time-conditioned-proxy" or manifest.get("status") != "complete":
            raise ValueError(f"not a completed deformable actor dataset: {root}")
        candidate_track = int(manifest["track_id"])
        if track_id is None: track_id = candidate_track
        if track_id != candidate_track: raise ValueError("all datasets must belong to one person track")
        candidate_width, candidate_height = int(manifest["width"]), int(manifest["height"])
        if width is None: width, height = candidate_width, candidate_height
        if (width, height) != (candidate_width, candidate_height): raise ValueError("all deformable datasets need matching resolution")
        camera_ids.add(str(manifest["source_camera"]))
        for record in manifest.get("records", []):
            if not isinstance(record, dict): continue
            image_path = root / str(record["image"])
            with Image.open(image_path) as opened:
                rgba = np.asarray(opened.convert("RGBA"), np.float32) / 255.0
            mask = rgba[..., 3] >= .5
            frames.append({"rgb": rgba[..., :3], "mask": mask, "bounds": actor_bounds(mask), "split": str(record["split"]), "timestamp_us": int(record["timestamp_us"]), "camera_id": str(record["source_camera"]), "reference": int(record["reference_frame"])})
    if width is None or height is None or track_id is None or len(frames) < 3:
        raise RuntimeError("too few deformable observations")
    cameras = sorted(camera_ids)
    # Individual camera/window datasets normalize their own time interval.
    # Recompute once here so a future multi-camera invocation shares a single
    # physical time axis rather than mapping every input window to [0, 1].
    first_timestamp = min(value["timestamp_us"] for value in frames)
    timestamp_span = max(max(value["timestamp_us"] for value in frames) - first_timestamp, 1)
    for frame in frames:
        frame["time"] = float((frame["timestamp_us"] - first_timestamp) / timestamp_span)
    if sum(value["split"] == "train" for value in frames) < 2 or sum(value["split"] == "holdout" for value in frames) < 1:
        raise RuntimeError("datasets need two train and one holdout observation")
    return frames, cameras, width, height, track_id


class DynamicPlaneLayer:  # constructed as nn.Module lazily to keep CPU imports light
    @staticmethod
    def create(*, atlas_size: int, hidden_dim: int, camera_count: int) -> "Any":
        import torch
        from torch import nn

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.atlas = nn.Parameter(torch.empty(1, 4, atlas_size, atlas_size))
                nn.init.normal_(self.atlas[:, :3], mean=0.0, std=.02)
                # Start from an identity sampler with a moderately
                # transparent atlas.  A near-zero alpha plus random initial
                # warp spreads the sparse silhouette gradient across unrelated
                # texels and made the first smoke remain fully transparent.
                nn.init.constant_(self.atlas[:, 3:], -1.0)
                self.camera_embedding = nn.Embedding(camera_count, 8)
                self.deformation = nn.Sequential(nn.Linear(2 + 1 + 8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
                nn.init.zeros_(self.deformation[-1].weight)
                nn.init.zeros_(self.deformation[-1].bias)
                # A static atlas plus a warp cannot express articulated
                # silhouette changes efficiently.  This bounded residual is
                # conditioned on time and camera; it starts at zero so the
                # canonical atlas remains the baseline, then learns only the
                # deformation/appearance not explainable by that baseline.
                self.residual = nn.Sequential(nn.Linear(2 + 1 + 8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)

            def forward(self, uv: "Any", time: "Any", camera: "Any") -> tuple["Any", "Any", "Any"]:
                import torch.nn.functional as functional
                condition = torch.cat((uv, time[:, None], self.camera_embedding(camera)), dim=-1)
                delta = .30 * torch.tanh(self.deformation(condition))
                grid = (uv + delta).clamp(-1.0, 1.0).view(1, -1, 1, 2)
                values = functional.grid_sample(self.atlas, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, :, 0].T
                residual = self.residual(condition)
                rgb_logits = values[:, :3] + .50 * torch.tanh(residual[:, :3])
                return torch.sigmoid(rgb_logits), torch.sigmoid(values[:, 3] + residual[:, 3]), delta
        return _Layer()


def _predict_full(model: "Any", *, height: int, width: int, bounds: tuple[int, int, int, int], time: float, camera: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x0, y0, x1, y1 = bounds
    yy, xx = torch.meshgrid(torch.arange(y0, y1, device=device), torch.arange(x0, x1, device=device), indexing="ij")
    indices = (yy * width + xx).reshape(-1)
    uv = _local_coordinates(indices, width=width, bounds=bounds).to(device)
    rgb_parts: list[torch.Tensor] = []; alpha_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(uv), 16_384):
            points = uv[start:start + 16_384]; count = len(points)
            rgb, alpha, _ = model(points, torch.full((count,), time, device=device), torch.full((count,), camera, dtype=torch.long, device=device))
            rgb_parts.append(rgb); alpha_parts.append(alpha)
    rgb = np.zeros((height, width, 3), np.float32); alpha = np.zeros((height, width), np.float32)
    rgb[y0:y1, x0:x1] = torch.cat(rgb_parts).reshape(y1 - y0, x1 - x0, 3).cpu().numpy()
    alpha[y0:y1, x0:x1] = torch.cat(alpha_parts).reshape(y1 - y0, x1 - x0).cpu().numpy()
    return rgb, alpha


def _evaluate(model: "Any", frames: list[dict[str, Any]], cameras: list[str], *, height: int, width: int, device: str) -> dict[str, Any]:
    values: list[dict[str, Any]] = []; mapping = {value: index for index, value in enumerate(cameras)}
    for frame in frames:
        if frame["split"] != "holdout": continue
        rgb, alpha = _predict_full(model, height=height, width=width, bounds=frame["bounds"], time=frame["time"], camera=mapping[frame["camera_id"]], device=device)
        mask = frame["mask"]; ring = ring_mask(__import__("torch").from_numpy(mask)).cpu().numpy()
        metrics = alpha_metrics(alpha, mask, ring)
        values.append({"reference_frame": frame["reference"], "rgb_mae": float(np.abs(rgb[mask] - frame["rgb"][mask]).mean()), **metrics})
    if not values: raise RuntimeError("no holdout frames available")
    return {"frames": values, "metrics": {key: float(np.mean([value[key] for value in values])) for key in ("rgb_mae", "alpha_iou_0_5", "alpha_area_ratio", "ring_alpha")}}


def train_deformable_actor(options: DeformableActorTrainOptions) -> Path:
    import torch
    import torch.nn.functional as functional

    output = options.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty deformable training output: {output}")
    frames, cameras, width, height, track_id = _load_frames(options.datasets)
    output.mkdir(parents=True); device = torch.device(options.device); camera_index = {value: index for index, value in enumerate(cameras)}
    model = DynamicPlaneLayer.create(atlas_size=options.atlas_size, hidden_dim=options.hidden_dim, camera_count=len(cameras)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    train = [value for value in frames if value["split"] == "train"]
    prepared: list[dict[str, Any]] = []
    for frame in train:
        mask = torch.from_numpy(frame["mask"])
        actor = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
        ring = torch.nonzero(ring_mask(mask).reshape(-1), as_tuple=False).reshape(-1)
        if len(actor) == 0 or len(ring) == 0: continue
        prepared.append({**frame, "actor": actor, "ring": ring, "rgb_tensor": torch.from_numpy(frame["rgb"]).reshape(-1, 3), "mask_tensor": mask.reshape(-1).float()})
    if len(prepared) < 2: raise RuntimeError("fewer than two train frames have actor and safe-ring samples")
    best: tuple[float, int, dict[str, Any]] | None = None
    for iteration in range(1, options.iterations + 1):
        frame = prepared[(iteration - 1) % len(prepared)]
        actor_indices = frame["actor"][torch.randint(len(frame["actor"]), (options.actor_samples,))]
        ring_indices = frame["ring"][torch.randint(len(frame["ring"]), (options.ring_samples,))]
        indices = torch.cat((actor_indices, ring_indices)); targets_alpha = torch.cat((torch.ones(len(actor_indices)), torch.zeros(len(ring_indices)))).to(device)
        uv = _local_coordinates(indices, width=width, bounds=frame["bounds"]).to(device); target_rgb = frame["rgb_tensor"][actor_indices].to(device)
        time = torch.full((len(indices),), frame["time"], device=device); camera = torch.full((len(indices),), camera_index[frame["camera_id"]], dtype=torch.long, device=device)
        predicted_rgb, predicted_alpha, deformation = model(uv, time, camera)
        rgb_loss = functional.l1_loss(predicted_rgb[:len(actor_indices)], target_rgb)
        alpha_loss = functional.binary_cross_entropy(predicted_alpha, targets_alpha)
        # BCE samples a ring but does not state that its *mean* opacity must
        # remain low.  This term prevents a visually soft halo from trading
        # against a slightly better foreground fit.
        ring_leak_loss = predicted_alpha[len(actor_indices):].mean()
        deformation_loss = deformation.square().mean()
        loss = rgb_loss + .75 * alpha_loss + options.ring_loss_weight * ring_leak_loss + .002 * deformation_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if iteration == 1 or iteration % options.validation_every == 0 or iteration == options.iterations:
            validation = _evaluate(model, frames, cameras, height=height, width=width, device=str(device))
            values = validation["metrics"]
            # A completely transparent prediction has superficially low ring
            # leakage.  Area is therefore compared in log space so selecting
            # a checkpoint cannot silently prefer the no-person baseline.
            score = values["rgb_mae"] + .50 * (1.0 - values["alpha_iou_0_5"]) + .25 * abs(float(np.log(max(values["alpha_area_ratio"], 1.0e-4)))) + .50 * values["ring_alpha"]
            print(f"Deformable plane [{iteration:05d}/{options.iterations}] loss={float(loss):.6f} rgb={float(rgb_loss):.6f} alpha={float(alpha_loss):.6f} ring_loss={float(ring_leak_loss):.6f} holdout_rgb={values['rgb_mae']:.6f} iou={values['alpha_iou_0_5']:.4f} ring={values['ring_alpha']:.4f}", flush=True)
            if best is None or score < best[0]:
                best = (score, iteration, validation)
                torch.save({"schema": SCHEMA, "version": VERSION, "track_id": track_id, "camera_ids": cameras, "width": width, "height": height, "atlas_size": options.atlas_size, "hidden_dim": options.hidden_dim, "iteration": iteration, "model": model.state_dict()}, output / "best.pt")
    assert best is not None
    metrics = best[2]["metrics"]
    passed = bool(metrics["rgb_mae"] <= options.max_holdout_rgb_mae and metrics["alpha_iou_0_5"] >= options.min_holdout_alpha_iou and options.min_holdout_alpha_area_ratio <= metrics["alpha_area_ratio"] <= options.max_holdout_alpha_area_ratio and metrics["ring_alpha"] <= options.max_holdout_ring_alpha)
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "track_id": track_id, "datasets": [str(path.resolve()) for path in options.datasets], "camera_ids": cameras, "representation": {"actor_local_bounds": True, "time_axis": "all-input physical timestamps normalized jointly", "camera_conditioned": True, "soft_ring_loss": options.ring_loss_weight}, "best_iteration": best[1], "checkpoint": str((output / "best.pt").resolve()), "holdout": best[2], "passed_source_holdout": passed, "thresholds": {"max_rgb_mae": options.max_holdout_rgb_mae, "min_alpha_iou": options.min_holdout_alpha_iou, "alpha_area_ratio": [options.min_holdout_alpha_area_ratio, options.max_holdout_alpha_area_ratio], "max_ring_alpha": options.max_holdout_ring_alpha}, "limitations": ["This is a source-view, actor-local, time/camera-conditioned 2-D proxy, not 3-D geometry or a target-L4 renderer.", "Actor-local bounds are source-frame metadata; a passing source holdout is necessary but not sufficient for target-view composition.", "No static background, static PLY depth or generated-person prior is used in this training objective."]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote deformable dynamic-plane checkpoint: {output / 'best.pt'} (source_holdout_passed={passed})", flush=True)
    return output / "best.pt"
