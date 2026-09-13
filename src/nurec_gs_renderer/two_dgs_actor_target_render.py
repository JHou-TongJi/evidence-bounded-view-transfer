"""Render camera-gated canonical 2DGS actor experts over a static RGB sequence.

The official 2DGS rasterizer has a static, global-shutter pinhole camera model.
This bridge therefore does *not* claim native NCore FTheta/rolling-shutter
output.  Its narrow purpose is to test whether a separately trained actor
expert can improve a compatible target view without blending incompatible
source-camera surfaces into a ghosted object.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .camera import CameraFrame
from .pose_sources import CameraPathSelection, load_camera_path_set
from .street_dataset import _interpolate_track_pose
from .two_dgs_target_render import _camera_for_frame, _checkpoint_ply, _import_official_two_dgs


@dataclass(frozen=True)
class ActorViewExpert:
    """One actor-local 2DGS asset, trusted only near its source observation view."""

    name: str
    actor_dataset: Path
    model_path: Path
    iteration: int = -1

    def __post_init__(self) -> None:
        if not self.name or any(value in self.name for value in "/\\"):
            raise ValueError("expert name must be a simple non-empty label")


@dataclass(frozen=True)
class TwoDgsActorTargetRenderOptions:
    two_dgs_root: Path
    camera_config: Path
    output: Path
    experts: tuple[ActorViewExpert, ...]
    background_rgb_dir: Path | None = None
    background_depth_dir: Path | None = None
    background_depth_quantization_m: float = .01
    static_occlusion_margin_m: float = .50
    camera_ids: tuple[str, ...] = ()
    chunk_index: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    width: int = 480
    height: int = 270
    device: str = "cuda"
    max_view_angle_deg: float = 30.0
    min_distance_ratio: float = .55
    max_distance_ratio: float = 1.85
    temporal_fade_us: int = 100_000

    def __post_init__(self) -> None:
        if not self.experts or len({item.name for item in self.experts}) != len(self.experts):
            raise ValueError("at least one uniquely named actor expert is required")
        if self.width <= 0 or self.height <= 0 or self.stride <= 0:
            raise ValueError("output dimensions and stride must be positive")
        if not 0.0 < self.max_view_angle_deg < 90.0:
            raise ValueError("max_view_angle_deg must be in (0, 90)")
        if not 0.0 < self.min_distance_ratio <= self.max_distance_ratio:
            raise ValueError("distance ratio bounds are invalid")
        if self.temporal_fade_us < 0:
            raise ValueError("temporal_fade_us must be non-negative")
        if not 0.001 <= self.background_depth_quantization_m <= 1.0:
            raise ValueError("background_depth_quantization_m must be in [0.001, 1.0]")
        if not 0.0 <= self.static_occlusion_margin_m <= 5.0:
            raise ValueError("static_occlusion_margin_m must be in [0, 5]")


@dataclass(frozen=True)
class _LoadedExpert:
    spec: ActorViewExpert
    model: Any
    track: dict[str, Any]
    track_id: int
    prototype_direction: np.ndarray
    prototype_distance: float
    timestamp_start_us: int
    timestamp_end_us: int
    checkpoint: Path
    iteration: int


def accepted_experts_from_batch_manifest(path: Path) -> tuple[ActorViewExpert, ...]:
    """Load only independently accepted rigid experts from a batch manifest.

    The batch manifest is an explicit quality gate.  In particular, this
    helper deliberately does not fall back to a merely trained, skipped, or
    rejected actor: target rendering must fail closed to the static sequence.
    """
    manifest = _read_json(path.resolve())
    if manifest.get("schema") != "ncore-2dgs-rigid-actor-batch":
        raise ValueError("batch manifest has an unexpected schema")
    experts: list[ActorViewExpert] = []
    names: set[str] = set()
    for record in manifest.get("train", []):
        if not isinstance(record, dict) or record.get("status") != "accepted":
            continue
        name = str(record.get("name", ""))
        dataset = record.get("dataset")
        model = record.get("model")
        if not name or not dataset or not model:
            raise ValueError(f"accepted batch record is incomplete: {name or '<unnamed>'}")
        if name in names:
            raise ValueError(f"duplicate accepted expert name in batch manifest: {name}")
        names.add(name)
        # Older manifests predate the explicit iteration field.  ``-1`` is
        # safe here because ``_checkpoint_ply`` chooses the latest saved
        # point cloud within this isolated model directory.
        experts.append(ActorViewExpert(name, Path(str(dataset)), Path(str(model)), int(record.get("iteration", -1))))
    if not experts:
        raise ValueError("batch manifest contains no accepted actor experts")
    return tuple(experts)


def experts_from_registry(path: Path) -> tuple[ActorViewExpert, ...]:
    """Load a clip-agnostic, source-holdout validated expert registry."""
    manifest = _read_json(path.resolve())
    if manifest.get("schema") != "ncore-2dgs-actor-expert-registry" or manifest.get("status") != "complete":
        raise ValueError("expert registry is incomplete or has an unexpected schema")
    experts: list[ActorViewExpert] = []
    for record in manifest.get("experts", []):
        if not isinstance(record, dict):
            continue
        name = record.get("registry_name")
        dataset = record.get("actor_dataset")
        model = record.get("model_path")
        if not name or not dataset or not model:
            raise ValueError("expert registry contains an incomplete expert")
        experts.append(ActorViewExpert(str(name), Path(str(dataset)), Path(str(model)), int(record.get("iteration", -1))))
    if not experts:
        raise ValueError("expert registry contains no experts")
    return tuple(experts)


def composite_actor_layers_by_depth(
    rgb_layers: np.ndarray,
    alpha_layers: np.ndarray,
    depth_layers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite premultiplied actor layers back-to-front using actor depth.

    This only resolves actor-vs-actor overlap.  The optional static RGB
    background has no compatible depth map in this proxy path, so it is still
    intentionally *not* used as an occluder.  Invalid/transparent samples
    are made infinitely far and have zero opacity.
    """
    rgb = np.asarray(rgb_layers, np.float32)
    alpha = np.asarray(alpha_layers, np.float32)
    depth = np.asarray(depth_layers, np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or alpha.shape != rgb.shape[:3] or depth.shape != alpha.shape:
        raise ValueError("actor layers must be [N,H,W,3], [N,H,W], [N,H,W]")
    count, height, width, _ = rgb.shape
    if count == 0:
        return np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32), np.full((height, width), -1, np.int32)
    valid = np.isfinite(depth) & (depth > 0.0) & (alpha > 1e-5)
    safe_depth = np.where(valid, depth, np.inf)
    safe_alpha = np.where(valid, np.clip(alpha, 0.0, 1.0), 0.0)
    safe_rgb = rgb * safe_alpha[..., None] / np.clip(alpha[..., None], 1e-6, None)
    # Descending depth means each new layer is nearer than the accumulated
    # result, so ordinary premultiplied alpha-over is valid.
    order = np.argsort(safe_depth, axis=0)[::-1]
    composed_rgb = np.zeros((height, width, 3), np.float32)
    composed_alpha = np.zeros((height, width), np.float32)
    for rank in range(count):
        indices = order[rank]
        layer_rgb = np.take_along_axis(safe_rgb, indices[None, ..., None], axis=0)[0]
        layer_alpha = np.take_along_axis(safe_alpha, indices[None, ...], axis=0)[0]
        composed_rgb = layer_rgb + (1.0 - layer_alpha)[..., None] * composed_rgb
        composed_alpha = layer_alpha + (1.0 - layer_alpha) * composed_alpha
    nearest = np.argmin(safe_depth, axis=0).astype(np.int32)
    nearest[~np.isfinite(np.min(safe_depth, axis=0))] = -1
    return np.clip(composed_rgb, 0.0, 1.0), np.clip(composed_alpha, 0.0, 1.0), nearest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def observer_direction_in_actor(actor_to_world: np.ndarray, world_to_camera: np.ndarray) -> tuple[np.ndarray, float]:
    """Return actor-to-camera direction and distance in actor-local coordinates."""
    actor = np.asarray(actor_to_world, dtype=np.float64)
    w2c = np.asarray(world_to_camera, dtype=np.float64)
    if actor.shape != (4, 4) or w2c.shape != (4, 4):
        raise ValueError("actor and camera transforms must be 4x4")
    camera_to_actor = np.linalg.inv(actor) @ np.linalg.inv(w2c)
    position = camera_to_actor[:3, 3]
    distance = float(np.linalg.norm(position))
    if not np.isfinite(position).all() or distance <= 1e-6:
        raise ValueError("camera is coincident with the actor")
    return (position / distance).astype(np.float64), distance


def expert_view_score(
    target_direction: np.ndarray,
    target_distance: float,
    prototype_direction: np.ndarray,
    prototype_distance: float,
    *,
    max_view_angle_deg: float,
    min_distance_ratio: float,
    max_distance_ratio: float,
) -> float | None:
    """Fail closed outside the trusted angular/distance observation envelope."""
    target = np.asarray(target_direction, dtype=np.float64).reshape(3)
    prototype = np.asarray(prototype_direction, dtype=np.float64).reshape(3)
    cosine = float(np.clip(np.dot(target, prototype), -1.0, 1.0))
    minimum_cosine = float(np.cos(np.deg2rad(max_view_angle_deg)))
    ratio = float(target_distance / max(prototype_distance, 1e-6))
    if cosine < minimum_cosine or not min_distance_ratio <= ratio <= max_distance_ratio:
        return None
    # Select angularly closest compatible expert; distance is a weak tie-break.
    return cosine - .01 * abs(np.log(ratio))


def temporal_visibility_weight(timestamp_us: int, start_us: int, end_us: int, fade_us: int) -> float:
    """Fade only within an already trusted observation interval."""
    if timestamp_us < start_us or timestamp_us > end_us:
        return 0.0
    if fade_us <= 0:
        return 1.0
    return float(np.clip(min((timestamp_us - start_us) / fade_us, (end_us - timestamp_us) / fade_us, 1.0), 0.0, 1.0))


def _load_expert(spec: ActorViewExpert, gaussian_cls: Any) -> _LoadedExpert:
    dataset_path = spec.actor_dataset.resolve()
    manifest = _read_json(dataset_path / "ncore_2dgs_actor_manifest.json")
    if manifest.get("schema") != "ncore-2dgs-canonical-rigid-actor-proxy" or manifest.get("status") != "complete":
        raise ValueError(f"expert {spec.name} is not a completed canonical actor dataset")
    dynamic = _read_json(Path(str(manifest["dynamic_dataset_manifest"])))
    track_id = int(manifest["track_id"])
    track = next((value for value in dynamic.get("tracks", []) if int(value.get("track_id", -1)) == track_id), None)
    if not isinstance(track, dict) or track.get("actor_family") != "rigid":
        raise ValueError(f"expert {spec.name} has no rigid source track")
    positions = []
    timestamps: list[int] = []
    for record in manifest.get("records", []):
        matrix = np.asarray(record.get("camera_to_actor"), dtype=np.float64)
        if matrix.shape != (4, 4):
            continue
        point = matrix[:3, 3]
        distance = float(np.linalg.norm(point))
        if distance > 1e-6 and np.isfinite(point).all():
            positions.append(point)
            timestamp = record.get("timestamp_us")
            if timestamp is not None:
                timestamps.append(int(timestamp))
    if not positions or not timestamps:
        raise ValueError(f"expert {spec.name} has no usable actor-local source cameras")
    position = np.median(np.asarray(positions), axis=0)
    distance = float(np.linalg.norm(position))
    checkpoint, iteration = _checkpoint_ply(spec.model_path.resolve(), spec.iteration)
    model = gaussian_cls(3)
    model.load_ply(str(checkpoint))
    return _LoadedExpert(
        spec, model, track, track_id, position / distance, distance,
        min(timestamps), max(timestamps), checkpoint, iteration,
    )


def _resized(frame: CameraFrame, width: int, height: int) -> CameraFrame:
    if frame.intrinsics.width == width and frame.intrinsics.height == height:
        return frame
    return CameraFrame(
        frame_id=frame.frame_id,
        intrinsics=frame.intrinsics.resized(width, height),
        world_to_camera=frame.world_to_camera,
        timestamp_us=frame.timestamp_us,
    )


def _actor_camera(frame: CameraFrame, actor_to_world: np.ndarray) -> CameraFrame:
    return CameraFrame(
        frame_id=frame.frame_id,
        intrinsics=frame.intrinsics,
        world_to_camera=np.asarray(frame.world_to_camera, np.float64) @ np.asarray(actor_to_world, np.float64),
        timestamp_us=frame.timestamp_us,
    )


def background_frame_path(directory: Path, frame: CameraFrame, render_index: int) -> Path:
    """Resolve static RGB by original reference ID before local render ordinal.

    A sliced target path keeps its original NCore reference in ``frame_id``
    (for example ``000100_3376478``).  Static full-clip RGB uses that global
    reference as the filename, whereas the new actor directory always starts
    from ``000000``.  Using the local ordinal for both would silently compose
    different timestamps.
    """
    token = str(frame.frame_id).split("_", 1)[0]
    candidates: list[Path] = []
    if token.isdigit():
        candidates.append(directory / f"{int(token):06d}.png")
        candidates.extend(sorted(directory.glob(f"{int(token):06d}_*.png")))
    candidates.append(directory / f"{render_index:06d}.png")
    candidates.extend(sorted(directory.glob(f"{render_index:06d}_*.png")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"static RGB frame is missing; tried: {candidates}")


def background_rgb_directory(root: Path, camera_id: str) -> Path:
    """Accept either one ``rgb`` directory or a seven-camera render root."""
    candidate = root / camera_id / "rgb"
    return candidate if candidate.is_dir() else root


def load_background_depth(directory: Path, frame: CameraFrame, render_index: int, quantization_m: float) -> np.ndarray:
    """Load camera-z depth written by ``nurec-gs-2dgs-target-render``."""
    source = background_frame_path(directory, frame, render_index)
    with Image.open(source) as opened:
        encoded = np.asarray(opened, dtype=np.uint16)
    if encoded.ndim != 2:
        raise ValueError(f"background depth must be a single-channel PNG: {source}")
    return encoded.astype(np.float32) * float(quantization_m)


def render_two_dgs_actor_target_sequence(options: TwoDgsActorTargetRenderOptions) -> Path:
    """Render the nearest compatible actor expert over optional static RGB frames."""
    import torch

    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    gaussian_cls, external = _import_official_two_dgs(options.two_dgs_root)
    camera_cls, render = external
    experts = tuple(_load_expert(value, gaussian_cls) for value in options.experts)
    track_ids = sorted({value.track_id for value in experts})
    selection = CameraPathSelection(chunk_index=options.chunk_index, frame_start=options.frame_start, frame_end=options.frame_end, stride=options.stride)
    requested = set(options.camera_ids) if options.camera_ids else None
    camera_set = load_camera_path_set(options.camera_config, selection=selection, camera_ids=requested)
    output.mkdir(parents=True)
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device=options.device)
    counts: dict[str, dict[str, int]] = {}
    for camera_id, path in camera_set.paths.items():
        rgb_dir, actor_dir, alpha_dir, depth_dir, owner_dir, reject_dir, expert_dir = (
            output / camera_id / "rgb", output / camera_id / "actor_rgb", output / camera_id / "actor_alpha",
            output / camera_id / "actor_depth", output / camera_id / "actor_owner", output / camera_id / "actor_static_reject", output / camera_id / "expert",
        )
        for directory in (rgb_dir, actor_dir, alpha_dir, depth_dir, owner_dir, reject_dir, expert_dir):
            directory.mkdir(parents=True)
        selected_counts = {value.spec.name: 0 for value in experts}
        skipped = 0
        depth_rejected_pixels = 0
        for index, raw_frame in enumerate(path.frames):
            frame = _resized(raw_frame, options.width, options.height)
            timestamp = frame.timestamp_us
            selected: dict[int, tuple[_LoadedExpert, np.ndarray]] = {}
            if timestamp is not None:
                candidates: dict[int, tuple[float, _LoadedExpert, np.ndarray]] = {}
                for expert in experts:
                    # A canonical asset is spatially static, but it is not
                    # evidence that this actor was visible throughout the
                    # whole clip.  Never extrapolate it beyond its source
                    # observation interval merely because a cuboid pose
                    # happens to exist there.
                    if not expert.timestamp_start_us <= int(timestamp) <= expert.timestamp_end_us:
                        continue
                    pose = _interpolate_track_pose(expert.track, int(timestamp))
                    if pose is None:
                        continue
                    direction, distance = observer_direction_in_actor(pose, frame.world_to_camera)
                    score = expert_view_score(direction, distance, expert.prototype_direction, expert.prototype_distance, max_view_angle_deg=options.max_view_angle_deg, min_distance_ratio=options.min_distance_ratio, max_distance_ratio=options.max_distance_ratio)
                    if score is not None:
                        prior = candidates.get(expert.track_id)
                        if prior is None or score > prior[0]:
                            candidates[expert.track_id] = (score, expert, pose)
                selected = {track_id: (expert, pose) for track_id, (_, expert, pose) in candidates.items()}
            if not selected:
                actor_rgb = np.zeros((options.height, options.width, 3), np.float32)
                actor_alpha = np.zeros((options.height, options.width), np.float32)
                actor_depth = np.zeros((options.height, options.width), np.float32)
                owner = np.full((options.height, options.width), -1, np.int32)
                expert_names: list[str] = []
                skipped += 1
            else:
                layer_rgb: list[np.ndarray] = []
                layer_alpha: list[np.ndarray] = []
                layer_depth: list[np.ndarray] = []
                selected_layers: list[tuple[_LoadedExpert, np.ndarray]] = []
                for track_id in sorted(selected):
                    expert, actor_pose = selected[track_id]
                    camera = _camera_for_frame(_actor_camera(frame, actor_pose), camera_cls, torch)
                    with torch.inference_mode():
                        result = render(camera, expert.model, pipeline, background)
                    rgb = torch.clamp(result["render"], 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    alpha = torch.clamp(result["rend_alpha"][0], 0.0, 1.0).cpu().numpy()
                    depth = result["surf_depth"][0].cpu().numpy().astype(np.float32)
                    fade = temporal_visibility_weight(int(timestamp), expert.timestamp_start_us, expert.timestamp_end_us, options.temporal_fade_us)
                    # Official 2DGS returns RGB premultiplied by black alpha.
                    # Applying the same visibility weight to RGB and alpha keeps
                    # the over operator valid and does not invent pixels outside
                    # the verified observation interval.
                    rgb *= fade
                    alpha *= fade
                    # The direction/distance gate is only a coarse proxy for
                    # the target frustum.  Count an expert as active only
                    # when its actual 2DGS render contains visible alpha.
                    # This prevents coverage statistics from claiming an
                    # off-screen actor was inserted into the target sequence.
                    if float(alpha.max()) <= 1.0 / 255.0:
                        continue
                    layer_rgb.append(rgb)
                    layer_alpha.append(alpha)
                    layer_depth.append(depth)
                    selected_layers.append((expert, depth))
                    selected_counts[expert.spec.name] += 1
                if not layer_rgb:
                    actor_rgb = np.zeros((options.height, options.width, 3), np.float32)
                    actor_alpha = np.zeros((options.height, options.width), np.float32)
                    actor_depth = np.zeros((options.height, options.width), np.float32)
                    owner = np.full((options.height, options.width), -1, np.int32)
                    expert_names = []
                    skipped += 1
                else:
                    actor_rgb, actor_alpha, owner = composite_actor_layers_by_depth(
                        np.stack(layer_rgb), np.stack(layer_alpha), np.stack(layer_depth),
                    )
                    stacked_depth = np.stack(layer_depth)
                    actor_depth = np.zeros((options.height, options.width), np.float32)
                    valid_owner = owner >= 0
                    if np.any(valid_owner):
                        actor_depth[valid_owner] = np.take_along_axis(
                            stacked_depth, owner.clip(min=0)[None, ...], axis=0,
                        )[0][valid_owner]
                    # Store a stable track id, rather than the temporary layer
                    # index, so debugging remains meaningful across frames.
                    owner_track = np.full_like(owner, -1)
                    for layer_index, (expert, _) in enumerate(selected_layers):
                        owner_track[owner == layer_index] = expert.track_id
                    owner = owner_track
                    expert_names = [expert.spec.name for expert, _ in selected_layers]
            rejection = np.zeros((options.height, options.width), dtype=bool)
            if options.background_depth_dir is not None and np.any(actor_alpha > 0.0):
                static_depth = load_background_depth(
                    background_rgb_directory(options.background_depth_dir.resolve(), camera_id), frame, index,
                    options.background_depth_quantization_m,
                )
                if static_depth.shape != actor_alpha.shape:
                    raise ValueError(f"background depth shape {static_depth.shape} does not match actor render {actor_alpha.shape}")
                # Reject only actors that are demonstrably *behind* the static
                # surface. Equal-depth baked dynamic ghosts remain eligible
                # for replacement; a hard front-only gate would preserve the
                # very static blur this actor layer is meant to diagnose.
                rejection = (actor_alpha > 1e-5) & (static_depth > 0.0) & (actor_depth > static_depth + options.static_occlusion_margin_m)
                if np.any(rejection):
                    actor_rgb[rejection] = 0.0
                    actor_alpha[rejection] = 0.0
                    actor_depth[rejection] = 0.0
                    owner[rejection] = -1
                    depth_rejected_pixels += int(rejection.sum())
            if options.background_rgb_dir is None:
                static_rgb = np.zeros_like(actor_rgb)
            else:
                source = background_frame_path(background_rgb_directory(options.background_rgb_dir.resolve(), camera_id), frame, index)
                with Image.open(source) as opened:
                    static_rgb = np.asarray(opened.convert("RGB").resize((options.width, options.height)), np.float32) / 255.0
            final = np.clip(actor_rgb + (1.0 - actor_alpha)[..., None] * static_rgb, 0.0, 1.0)
            Image.fromarray((final * 255.0 + .5).astype(np.uint8), "RGB").save(rgb_dir / f"{index:06d}.png")
            Image.fromarray((actor_rgb * 255.0 + .5).astype(np.uint8), "RGB").save(actor_dir / f"{index:06d}.png")
            Image.fromarray((actor_alpha * 255.0 + .5).astype(np.uint8), "L").save(alpha_dir / f"{index:06d}.png")
            Image.fromarray((np.clip(actor_depth / 80.0, 0.0, 1.0) * 65535.0 + .5).astype(np.uint16), "I;16").save(depth_dir / f"{index:06d}.png")
            Image.fromarray(np.where(owner >= 0, owner + 1, 0).astype(np.uint16), "I;16").save(owner_dir / f"{index:06d}.png")
            Image.fromarray(rejection.astype(np.uint8) * 255, "L").save(reject_dir / f"{index:06d}.png")
            (expert_dir / f"{index:06d}.txt").write_text(",".join(expert_names) if expert_names else "none\n", encoding="utf-8")
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(path.frames):
                print(f"2DGS actor target [{camera_id}] {index + 1}/{len(path.frames)} experts={','.join(expert_names) or 'none'}", flush=True)
        counts[camera_id] = {**selected_counts, "none": skipped, "static_depth_rejected_pixels": depth_rejected_pixels}
    manifest = {
        "schema": "ncore-2dgs-camera-gated-actor-target-diagnostic", "version": 1,
        "track_ids": track_ids, "camera_config": str(options.camera_config.resolve()),
        "background_rgb_dir": None if options.background_rgb_dir is None else str(options.background_rgb_dir.resolve()),
        "background_depth_dir": None if options.background_depth_dir is None else str(options.background_depth_dir.resolve()),
        "background_depth_quantization_m": options.background_depth_quantization_m,
        "static_occlusion_margin_m": options.static_occlusion_margin_m,
        "experts": [{"name": value.spec.name, "track_id": value.track_id, "actor_dataset": str(value.spec.actor_dataset.resolve()), "model_path": str(value.spec.model_path.resolve()), "iteration": value.iteration, "checkpoint": str(value.checkpoint), "prototype_direction_actor": value.prototype_direction.tolist(), "prototype_distance_m": value.prototype_distance, "observation_timestamps_us": [value.timestamp_start_us, value.timestamp_end_us]} for value in experts],
        "counts": counts, "output_size": [options.width, options.height],
        "max_view_angle_deg": options.max_view_angle_deg, "distance_ratio": [options.min_distance_ratio, options.max_distance_ratio],
        "temporal_fade_us": options.temporal_fade_us,
        "limitations": [
            "Actor experts are official 2DGS midpoint-pinhole proxies, not native FTheta or rolling-shutter renderings.",
            "At most one compatible expert is selected per track inside its observed time range; unsupported tracks fail closed to static background.",
            "Actor layers are depth-sorted against one another. If a compatible static 2DGS depth directory is supplied, only actors demonstrably behind that static surface are rejected; equal-depth baked ghosts may still be replaced.",
            "This is not a production dynamic sequence or L4 7V asset.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote camera-gated 2DGS actor target diagnostic: {output}", flush=True)
    return output
