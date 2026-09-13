"""Render a static official-2DGS checkpoint through this project's target rig.

This is deliberately a diagnostic bridge.  Official 2DGS is a static,
global-shutter pinhole renderer and has no dynamic actor layer, native FTheta,
or rolling-shutter model.  It is useful for making a target-rig sequence that
reveals baked dynamic-object ghosts, but never produces sensor ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .camera import CameraFrame
from .pose_sources import CameraPathSelection, load_camera_path_set
from .sky import load_sky_asset, sample_sky_asset


@dataclass(frozen=True)
class TwoDgsTargetRenderOptions:
    two_dgs_root: Path
    source_path: Path
    model_path: Path
    camera_config: Path
    output: Path
    iteration: int = -1
    camera_ids: tuple[str, ...] = ()
    chunk_index: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    width: int = 480
    height: int = 270
    device: str = "cuda"
    sky_cubemap: Path | None = None
    sky_confidence_threshold: float = 0.5
    sky_min_sample_count: int = 1
    write_depth: bool = False
    depth_quantization_m: float = .01

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.stride <= 0:
            raise ValueError("width, height and stride must be positive")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if not 0.0 <= self.sky_confidence_threshold <= 1.0 or self.sky_min_sample_count <= 0:
            raise ValueError("sky sampling thresholds are invalid")
        if not 0.001 <= self.depth_quantization_m <= 1.0:
            raise ValueError("depth_quantization_m must be in [0.001, 1.0]")


def _import_official_two_dgs(root: Path) -> tuple[object, object]:
    root = root.resolve()
    if not (root / "scene" / "gaussian_model.py").is_file():
        raise FileNotFoundError(f"not an official 2DGS checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from scene import GaussianModel
        from scene.cameras import Camera
        from gaussian_renderer import render
    except ImportError as exc:  # pragma: no cover - external environment
        raise RuntimeError("could not import official 2DGS; use its checkout as PYTHONPATH or --two-dgs-root") from exc
    return GaussianModel, (Camera, render)


def _checkpoint_ply(model_path: Path, iteration: int) -> tuple[Path, int]:
    """Resolve an official-2DGS point-cloud checkpoint without loading images."""
    point_root = model_path / "point_cloud"
    if iteration == -1:
        candidates: list[tuple[int, Path]] = []
        for directory in point_root.glob("iteration_*"):
            try:
                value = int(directory.name.removeprefix("iteration_"))
            except ValueError:
                continue
            candidate = directory / "point_cloud.ply"
            if candidate.is_file():
                candidates.append((value, candidate))
        if not candidates:
            raise FileNotFoundError(f"no official 2DGS point-cloud checkpoint in {point_root}")
        # ``candidates`` is stored as (iteration, path), while this public
        # helper promises (path, iteration).  Keep the contract identical to
        # the explicit-iteration branch below.
        value, candidate = max(candidates, key=lambda item: item[0])
        return candidate, value
    candidate = point_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate, int(iteration)


def _camera_for_frame(frame: CameraFrame, camera_cls: object, torch: object) -> object:
    intrinsics = frame.intrinsics
    fov_x = 2.0 * math.atan(intrinsics.width / (2.0 * intrinsics.fx))
    fov_y = 2.0 * math.atan(intrinsics.height / (2.0 * intrinsics.fy))
    w2c = np.asarray(frame.world_to_camera, dtype=np.float64)
    image = torch.zeros((3, intrinsics.height, intrinsics.width), dtype=torch.float32)
    # Official Camera's R is stored transposed; it reconstructs W2C internally.
    return camera_cls(
        0, w2c[:3, :3].T, w2c[:3, 3], fov_x, fov_y, image, None, frame.frame_id, 0,
        data_device="cuda",
    )


def composite_two_dgs_sky(
    rendered_rgb: np.ndarray,
    rendered_alpha: np.ndarray,
    sky_rgb: np.ndarray,
    sky_valid: np.ndarray,
) -> np.ndarray:
    """Composite a direction-domain sky behind the official 2DGS surface layer.

    The official rasterizer uses the supplied black background, hence RGB is
    already premultiplied by its accumulated alpha.  This exactly mirrors the
    root renderer's ``C + (1-alpha) * sky`` contract without changing 2DGS
    alpha/depth semantics.
    """
    rgb = np.asarray(rendered_rgb, dtype=np.float32)
    alpha = np.asarray(rendered_alpha, dtype=np.float32)
    sky = np.asarray(sky_rgb, dtype=np.float32)
    valid = np.asarray(sky_valid, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or alpha.shape != rgb.shape[:2] or sky.shape != rgb.shape or valid.shape != rgb.shape[:2]:
        raise ValueError("2DGS RGB/alpha and sky tensors must have matching HxW shapes")
    return np.clip(rgb + (1.0 - np.clip(alpha, 0.0, 1.0))[..., None] * np.clip(valid, 0.0, 1.0)[..., None] * sky, 0.0, 1.0)


def render_two_dgs_target_sequence(options: TwoDgsTargetRenderOptions) -> Path:
    import torch

    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    GaussianModel, external = _import_official_two_dgs(options.two_dgs_root)
    Camera, render = external
    output.mkdir(parents=True)
    selection = CameraPathSelection(
        chunk_index=options.chunk_index, frame_start=options.frame_start,
        frame_end=options.frame_end, stride=options.stride,
    )
    requested = set(options.camera_ids) if options.camera_ids else None
    camera_set = load_camera_path_set(options.camera_config, selection=selection, camera_ids=requested)
    model = GaussianModel(3)
    checkpoint_ply, loaded_iteration = _checkpoint_ply(options.model_path.resolve(), options.iteration)
    print(f"Loading official 2DGS point cloud at iteration {loaded_iteration}: {checkpoint_ply}", flush=True)
    model.load_ply(str(checkpoint_ply))
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device=options.device)
    sky_asset = None if options.sky_cubemap is None else load_sky_asset(options.sky_cubemap)
    counts: dict[str, int] = {}
    for camera_id, path in camera_set.paths.items():
        camera_dir = output / camera_id / "rgb"
        camera_dir.mkdir(parents=True)
        if options.write_depth:
            (output / camera_id / "depth").mkdir(parents=True)
        if sky_asset is not None:
            (output / camera_id / "alpha").mkdir(parents=True)
            (output / camera_id / "sky_rgb").mkdir(parents=True)
            (output / camera_id / "sky_valid").mkdir(parents=True)
        for index, source_frame in enumerate(path.frames):
            frame = source_frame if (source_frame.intrinsics.width == options.width and source_frame.intrinsics.height == options.height) else CameraFrame(
                frame_id=source_frame.frame_id,
                intrinsics=source_frame.intrinsics.resized(options.width, options.height),
                world_to_camera=source_frame.world_to_camera, timestamp_us=source_frame.timestamp_us,
            )
            camera = _camera_for_frame(frame, Camera, torch)
            with torch.inference_mode():
                result = render(camera, model, pipeline, background)
                rendered = torch.clamp(result["render"], 0.0, 1.0)
                alpha = torch.clamp(result["rend_alpha"][0], 0.0, 1.0)
                depth = torch.nan_to_num(result["surf_depth"][0], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            rgb_float = rendered.permute(1, 2, 0).cpu().numpy()
            if sky_asset is not None:
                alpha_float = alpha.cpu().numpy()
                sky_rgb, sky_valid = sample_sky_asset(
                    sky_asset,
                    frame,
                    confidence_threshold=options.sky_confidence_threshold,
                    min_sample_count=options.sky_min_sample_count,
                )
                rgb_float = composite_two_dgs_sky(rgb_float, alpha_float, sky_rgb, sky_valid)
                Image.fromarray((alpha_float * 255.0 + .5).astype(np.uint8), mode="L").save(output / camera_id / "alpha" / f"{index:06d}.png")
                Image.fromarray((sky_rgb * 255.0 + .5).astype(np.uint8), mode="RGB").save(output / camera_id / "sky_rgb" / f"{index:06d}.png")
                Image.fromarray((sky_valid * 255.0 + .5).astype(np.uint8), mode="L").save(output / camera_id / "sky_valid" / f"{index:06d}.png")
            rgb = (rgb_float * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(camera_dir / f"{index:06d}.png")
            if options.write_depth:
                depth_u16 = np.clip(depth.cpu().numpy() / options.depth_quantization_m, 0.0, 65535.0)
                Image.fromarray((depth_u16 + .5).astype(np.uint16), mode="I;16").save(output / camera_id / "depth" / f"{index:06d}.png")
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(path.frames):
                print(f"2DGS target [{camera_id}] {index + 1}/{len(path.frames)} frame={frame.frame_id}", flush=True)
        counts[camera_id] = len(path.frames)
    manifest = {
        "schema": "nurec-2dgs-target-render-diagnostic", "version": 1,
        "two_dgs_root": str(options.two_dgs_root.resolve()), "source_path": str(options.source_path.resolve()),
        "model_path": str(options.model_path.resolve()), "iteration": loaded_iteration,
        "checkpoint_ply": str(checkpoint_ply),
        "camera_config": str(options.camera_config.resolve()), "counts": counts,
        "output_size": [options.width, options.height],
        "sky_cubemap": None if options.sky_cubemap is None else str(options.sky_cubemap.resolve()),
        "sky_confidence_threshold": options.sky_confidence_threshold,
        "sky_min_sample_count": options.sky_min_sample_count,
        "depth": None if not options.write_depth else {"directory": "depth", "quantization_m": options.depth_quantization_m, "encoding": "uint16_png"},
        "limitations": [
            "Official 2DGS is a static global-shutter pinhole representation.",
            "Target poses are pinhole midpoint poses; FTheta and rolling shutter are not represented.",
            "Dynamic objects in this output are baked static surfaces, not recovered actors.",
            "When supplied, the sky cubemap is composited only behind 2DGS alpha; it does not repair opaque geometry.",
            "This diagnostic output is not sensor truth, a training input, or an L4 7V production sequence.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote diagnostic 2DGS target sequence: {output}", flush=True)
    return output
