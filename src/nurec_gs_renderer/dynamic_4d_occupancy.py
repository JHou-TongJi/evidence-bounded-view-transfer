"""Actor-local 4D occupancy/transmittance reconstruction from raw NCore data.

Unlike previous actor experiments, this trainer does not render a Gaussian
layer over a static PLY.  A compact neural field represents density in an
actor-local ``(x, y, z, t)`` volume and view-dependent colour.  It is volume
rendered along exact NCore FTheta/rolling-shutter rays, yielding explicit
dynamic transmittance ``T_dyn`` and alpha ``1-T_dyn``.  The static scene is
neither loaded nor used as an occlusion or RGB supervisor.

MASt3R points are frozen, geometrically audited anchors only.  They constrain
observed occupancy; unknown volume is kept transparent by ray/ring evidence
instead of being filled from a contaminated static reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .mast3r_correspondence import SCHEMA as CORRESPONDENCE_SCHEMA
from .pose_sources import _create_ncore_loader
from .roma_multiview_audit import _native_world_rays
from .sky_build import dilate_boolean_mask
from .street_dataset import _interpolate_track_pose


SCHEMA = "ncore-dynamic-4d-occupancy-transmittance"
VERSION = 1
DATASET_SCHEMA = "ncore-dynamic-reconstruction-dataset"


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def _track_instance(record: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    return next((value for value in record["instances"] if int(value["track_id"]) == track_id), None)


def ray_box_intersection(origins: Any, directions: Any, half_extent: Any) -> tuple[Any, Any, Any]:
    """Return positive ray intervals through an actor-local axis-aligned box.

    The implementation is pure Torch so it is also used by CPU unit tests.
    Parallel rays are handled without division-by-zero; a ray parallel and
    outside one slab is invalid.
    """
    import torch

    origins, directions, half_extent = torch.as_tensor(origins), torch.as_tensor(directions), torch.as_tensor(half_extent)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3 or half_extent.shape != (3,):
        raise ValueError("origins/directions must be [N,3] and half_extent [3]")
    epsilon = torch.finfo(origins.dtype).eps
    parallel = directions.abs() <= epsilon
    outside_parallel = parallel & (origins.abs() > half_extent[None])
    inverse = torch.where(parallel, torch.ones_like(directions), directions.reciprocal())
    lower = (-half_extent[None] - origins) * inverse
    upper = (half_extent[None] - origins) * inverse
    near = torch.minimum(lower, upper)
    far = torch.maximum(lower, upper)
    near = torch.where(parallel, torch.full_like(near, -torch.inf), near)
    far = torch.where(parallel, torch.full_like(far, torch.inf), far)
    entry = near.max(dim=1).values.clamp_min(0.0)
    exit_ = far.min(dim=1).values
    valid = (~outside_parallel.any(dim=1)) & torch.isfinite(entry) & torch.isfinite(exit_) & (exit_ > entry + 1e-4)
    return entry, exit_, valid


def volume_weights(density: Any, deltas: Any) -> tuple[Any, Any, Any]:
    """Return alpha samples, weights and final transmittance for one ray batch."""
    import torch

    density, deltas = torch.as_tensor(density), torch.as_tensor(deltas)
    if density.shape != deltas.shape or density.ndim != 2:
        raise ValueError("density and deltas must have matching [N,S] shape")
    alpha = 1.0 - torch.exp(-density.clamp_min(0.0) * deltas.clamp_min(0.0))
    trans_before = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-8), dim=1), dim=1)[:, :-1]
    weights = trans_before * alpha
    transmittance = torch.prod(1.0 - alpha + 1e-8, dim=1)
    return alpha, weights, transmittance


def positional_encoding(value: Any, frequencies: int) -> Any:
    import torch

    value = torch.as_tensor(value)
    encoded = [value]
    for index in range(frequencies):
        scale = float(2 ** index) * np.pi
        encoded.extend((torch.sin(scale * value), torch.cos(scale * value)))
    return torch.cat(encoded, dim=-1)


class Actor4DOccupancyField:  # torch.nn.Module is delayed to keep CPU imports lightweight
    """Factory wrapper; ``module()`` creates the actual Torch module lazily."""

    @staticmethod
    def module(*, position_frequencies: int, time_frequencies: int, hidden_dim: int) -> Any:
        import torch

        position_dim = 3 * (1 + 2 * position_frequencies)
        time_dim = 1 + 2 * time_frequencies

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = torch.nn.Sequential(
                    torch.nn.Linear(position_dim + time_dim, hidden_dim), torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
                )
                self.density = torch.nn.Linear(hidden_dim, 1)
                self.colour = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 3, hidden_dim // 2), torch.nn.SiLU(), torch.nn.Linear(hidden_dim // 2, 3),
                )

            def forward(self, positions: Any, times: Any, directions: Any) -> tuple[Any, Any]:
                encoded = torch.cat((positional_encoding(positions, position_frequencies), positional_encoding(times, time_frequencies)), dim=-1)
                feature = self.trunk(encoded)
                # Start from near-transparent unknown volume.  A neutral
                # softplus bias makes an 8 m cuboid visibly foggy before any
                # evidence arrives, exactly the failure explicit
                # transmittance is meant to avoid.  Masks/LiDAR/correspondence
                # must earn opacity instead.
                density = torch.nn.functional.softplus(self.density(feature)[..., 0] - 5.0)
                colour = torch.sigmoid(self.colour(torch.cat((feature, directions), dim=-1)))
                return density, colour

            def density_only(self, positions: Any, times: Any) -> Any:
                encoded = torch.cat((positional_encoding(positions, position_frequencies), positional_encoding(times, time_frequencies)), dim=-1)
                return torch.nn.functional.softplus(self.density(self.trunk(encoded))[..., 0] - 5.0)

        return _Module()


def render_actor_rays(field: Any, origins: Any, directions: Any, times: Any, half_extent: Any, *, samples: int) -> dict[str, Any]:
    """Volume render one actor field and expose transmittance explicitly."""
    import torch

    entry, exit_, valid = ray_box_intersection(origins, directions, half_extent)
    ratio = (torch.arange(samples, device=origins.device, dtype=origins.dtype) + .5) / samples
    distance = entry[:, None] + (exit_ - entry)[:, None] * ratio[None]
    delta = ((exit_ - entry) / samples)[:, None].expand(-1, samples)
    positions = origins[:, None, :] + distance[..., None] * directions[:, None, :]
    normalized = positions / torch.as_tensor(half_extent, device=origins.device, dtype=origins.dtype)[None, None, :].clamp_min(1e-4)
    sample_time = times[:, None, None].expand(-1, samples, 1)
    sample_direction = directions[:, None, :].expand(-1, samples, -1)
    density, colour = field(normalized.reshape(-1, 3), sample_time.reshape(-1, 1), sample_direction.reshape(-1, 3))
    density, colour = density.reshape(-1, samples), colour.reshape(-1, samples, 3)
    density = torch.where(valid[:, None], density, torch.zeros_like(density))
    alpha_samples, weights, transmittance = volume_weights(density, delta)
    rgb = (weights[..., None] * colour).sum(dim=1)
    alpha = 1.0 - transmittance
    expected_depth = (weights * distance).sum(dim=1) / alpha.clamp_min(1e-5)
    expected_depth = torch.where(valid & (alpha > 1e-4), expected_depth, torch.full_like(expected_depth, torch.nan))
    return {"rgb": rgb, "alpha": alpha, "transmittance": transmittance, "expected_depth": expected_depth, "valid": valid, "weights": weights, "sample_alpha": alpha_samples}


@dataclass
class DynamicObservation:
    reference_index: int
    camera_id: str
    record: dict[str, Any]
    image: np.ndarray
    mask: np.ndarray
    ring: np.ndarray
    lidar_xy: np.ndarray
    lidar_depth: np.ndarray
    native_model: Any


def _load_observations(manifest: dict[str, Any], manifest_path: Path, *, track_id: int, camera_ids: tuple[str, ...] | None, reference_start: int | None, reference_end: int | None, device: str) -> tuple[list[DynamicObservation], dict[str, Any]]:
    dataset_root = manifest_path.parent
    loader = _create_ncore_loader(Path(manifest["ncore_path"]))
    tracks = {int(value["track_id"]): value for value in manifest["tracks"]}
    track = tracks.get(track_id)
    if track is None or track.get("actor_family") != "rigid":
        raise ValueError("requested actor must be a listed rigid track")
    from ncore.impl.sensors.camera import FThetaCameraModel
    models: dict[str, Any] = {}
    observations: list[DynamicObservation] = []
    for frame in manifest["frames"]:
        reference = int(frame["reference_frame_index"])
        if reference_start is not None and reference < reference_start:
            continue
        if reference_end is not None and reference >= reference_end:
            continue
        for record in frame["cameras"]:
            camera_id = str(record["camera_id"])
            if camera_ids is not None and camera_id not in camera_ids:
                continue
            instance = _track_instance(record, track_id)
            if instance is None:
                continue
            sensor = loader.get_camera_sensor(camera_id)
            image = np.asarray(sensor.get_frame_image_array(int(record["source_frame_index"])), np.uint8)
            mask = np.asarray(Image.open(dataset_root / str(instance["mask"])).convert("L"), np.uint8) > 0
            if mask.shape != image.shape[:2] or not mask.any():
                continue
            # A source semantic boundary is not a reliable empty-space label:
            # anti-aliasing and small track/mask disagreement occur exactly
            # there.  Reserve a 2 px uncertainty band, then supervise only a
            # wider exterior ring as transparent.  This is an explicit
            # visibility/unknown decision, not a post-hoc alpha threshold.
            ring = dilate_boolean_mask(mask, 8) & ~dilate_boolean_mask(mask, 2)
            xy, depth = np.empty((0, 2), np.int64), np.empty(0, np.float32)
            lidar_path = instance.get("lidar_depth")
            if isinstance(lidar_path, str):
                with np.load(dataset_root / lidar_path, allow_pickle=False) as value:
                    xy, depth = np.asarray(value["xy"], np.int64), np.asarray(value["depth_m"], np.float32)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < image.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < image.shape[0]) & np.isfinite(depth) & (depth > .1)
                xy, depth = xy[inside], depth[inside]
            native_model = models.setdefault(camera_id, FThetaCameraModel(sensor.model_parameters, device=device))
            observations.append(DynamicObservation(reference, camera_id, record, image, mask, ring, xy, depth, native_model))
    if not observations:
        raise RuntimeError("the requested actor/window has no source observations")
    return observations, track


def _pixel_rays_local(observation: DynamicObservation, pixels: np.ndarray, track: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    origins, directions = _native_world_rays(observation.native_model, pixels, observation.record)
    pose = _interpolate_track_pose(track, int(observation.record["timestamp_midpoint_us"]))
    if pose is None:
        raise RuntimeError(f"track has no pose at observation {observation.reference_index}")
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return ((origins - translation) @ rotation).astype(np.float32), (directions @ rotation).astype(np.float32)


def _sample_pixels(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.nonzero(mask)
    if not len(x):
        return np.empty((0, 2), np.int64)
    choose = rng.choice(len(x), size=count, replace=len(x) < count)
    return np.stack((x[choose], y[choose]), axis=1).astype(np.int64)


@dataclass(frozen=True)
class Dynamic4DTrainOptions:
    dataset_manifest: Path
    correspondence: Path
    output: Path
    track_id: int
    reference_start: int
    reference_end: int
    camera_ids: tuple[str, ...] | None = None
    temporal_even_train_odd_holdout: bool = True
    device: str = "cuda"
    iterations: int = 1_000
    learning_rate: float = 2e-3
    ray_samples: int = 48
    actor_rays_per_step: int = 384
    ring_rays_per_step: int = 192
    anchor_points_per_step: int = 512
    hidden_dim: int = 96
    position_frequencies: int = 5
    time_frequencies: int = 3
    rgb_weight: float = 1.0
    silhouette_weight: float = .5
    ring_transmittance_weight: float = .2
    ring_alpha_target: float = .08
    ring_hard_weight: float = 2.0
    lidar_weight: float = .05
    anchor_occupancy_weight: float = .25
    temporal_anchor_weight: float = .05
    max_holdout_ring_alpha: float = .10
    max_holdout_actor_alpha_error: float = .15
    max_holdout_rgb_mae: float = .17
    validation_every: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output.suffix != ".pt" or self.track_id < 0 or self.reference_start < 0 or self.reference_end <= self.reference_start:
            raise ValueError("output, track or reference window is invalid")
        if self.iterations <= 0 or self.learning_rate <= 0 or self.ray_samples < 8 or min(self.actor_rays_per_step, self.ring_rays_per_step, self.anchor_points_per_step, self.hidden_dim, self.validation_every) <= 0:
            raise ValueError("training budgets are invalid")
        if self.camera_ids is not None and (not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique")
        if not 0 < self.ring_alpha_target < 1 or not 0 < self.max_holdout_ring_alpha < 1 or not 0 < self.max_holdout_actor_alpha_error < 1 or not 0 < self.max_holdout_rgb_mae < 1:
            raise ValueError("ring/holdout acceptance thresholds must be in (0,1)")
        if any(value < 0 for value in (self.rgb_weight, self.silhouette_weight, self.ring_transmittance_weight, self.ring_hard_weight, self.lidar_weight, self.anchor_occupancy_weight, self.temporal_anchor_weight)):
            raise ValueError("loss weights must be non-negative")


def _time_value(reference: int, options: Dynamic4DTrainOptions) -> float:
    return 2.0 * (reference - options.reference_start) / max(options.reference_end - options.reference_start - 1, 1) - 1.0


def _batch_render(field: Any, observation: DynamicObservation, pixels: np.ndarray, track: dict[str, Any], options: Dynamic4DTrainOptions, *, device: Any) -> tuple[dict[str, Any], Any]:
    import torch

    origin, direction = _pixel_rays_local(observation, pixels, track)
    origins = torch.from_numpy(origin).to(device); directions = torch.from_numpy(direction).to(device)
    times = torch.full((len(pixels),), _time_value(observation.reference_index, options), device=device)
    half = torch.as_tensor(np.asarray(track["length_width_height"], np.float32) * .5, device=device)
    rendered = render_actor_rays(field, origins, directions, times, half, samples=options.ray_samples)
    target = torch.from_numpy(observation.image[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0).to(device)
    return rendered, target


def _evaluate(field: Any, observations: list[DynamicObservation], track: dict[str, Any], options: Dynamic4DTrainOptions, *, device: Any, rng: np.random.Generator) -> dict[str, float]:
    import torch

    values: list[tuple[float, float, float]] = []
    with torch.no_grad():
        for observation in observations:
            actor = _sample_pixels(observation.mask, min(1024, options.actor_rays_per_step), rng)
            ring = _sample_pixels(observation.ring, min(512, options.ring_rays_per_step), rng)
            if not len(actor) or not len(ring):
                continue
            actor_render, target = _batch_render(field, observation, actor, track, options, device=device)
            ring_render, _ = _batch_render(field, observation, ring, track, options, device=device)
            mae = (actor_render["rgb"] / actor_render["alpha"].clamp_min(.05)[..., None] - target).abs().mean()
            alpha_error = (actor_render["alpha"] - 1.).abs().mean()
            ring_alpha = ring_render["alpha"].mean()
            values.append((float(mae), float(alpha_error), float(ring_alpha)))
    return {"observation_count": float(len(values)), "rgb_mae": float(np.mean([value[0] for value in values])) if values else float("nan"), "actor_alpha_error": float(np.mean([value[1] for value in values])) if values else float("nan"), "ring_alpha": float(np.mean([value[2] for value in values])) if values else float("nan")}


def train_dynamic_4d_occupancy(options: Dynamic4DTrainOptions) -> Path:
    """Train a bounded raw-NCore 4D field and write strict temporal holdout metrics."""
    import torch

    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite existing 4D checkpoint: {options.output}")
    manifest_path, correspondence_path = options.dataset_manifest.resolve(), options.correspondence.resolve()
    manifest, correspondence = _json(manifest_path), _json(correspondence_path)
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("dataset manifest is not a complete dynamic reconstruction dataset")
    if correspondence.get("schema") != CORRESPONDENCE_SCHEMA or correspondence.get("status") != "complete":
        raise ValueError("correspondence report is invalid")
    if correspondence.get("dataset_manifest_sha256") != _sha256(manifest_path) or int(correspondence.get("track_id", -1)) != options.track_id:
        raise ValueError("correspondence asset is not bound to this dataset manifest/track")
    with np.load(Path(correspondence["asset"]), allow_pickle=False) as data:
        anchors, anchor_references = np.asarray(data["points_actor_local"], np.float32), np.asarray(data["reference_frame_indices"], np.int32)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not len(anchors):
        raise RuntimeError("MASt3R has no accepted native correspondences; refusing to train an unconstrained 4D field")
    if options.temporal_even_train_odd_holdout and np.any(anchor_references % 2):
        raise ValueError("temporal holdout would leak: correspondence asset contains odd reference anchors")
    observations, track = _load_observations(manifest, manifest_path, track_id=options.track_id, camera_ids=options.camera_ids, reference_start=options.reference_start, reference_end=options.reference_end, device=options.device)
    train = [value for value in observations if (value.reference_index % 2 == 0 if options.temporal_even_train_odd_holdout else True)]
    holdout = [value for value in observations if (value.reference_index % 2 == 1 if options.temporal_even_train_odd_holdout else False)]
    if not train or not holdout:
        raise RuntimeError("4D training needs non-empty even train and odd temporal holdout observations")
    torch.manual_seed(options.seed); rng = np.random.default_rng(options.seed)
    field = Actor4DOccupancyField.module(position_frequencies=options.position_frequencies, time_frequencies=options.time_frequencies, hidden_dim=options.hidden_dim).to(options.device)
    optimizer = torch.optim.Adam(field.parameters(), lr=options.learning_rate)
    anchor_tensor = torch.from_numpy(anchors).to(options.device)
    anchor_time = torch.from_numpy(np.asarray([_time_value(int(value), options) for value in anchor_references], np.float32)).to(options.device)
    history: list[dict[str, float]] = []
    best_score, best_state = float("inf"), None
    for step in range(options.iterations):
        observation = train[step % len(train)]
        actor_pixels = _sample_pixels(observation.mask, options.actor_rays_per_step, rng)
        ring_pixels = _sample_pixels(observation.ring, options.ring_rays_per_step, rng)
        if not len(actor_pixels) or not len(ring_pixels):
            continue
        actor, target = _batch_render(field, observation, actor_pixels, track, options, device=options.device)
        ring, _ = _batch_render(field, observation, ring_pixels, track, options, device=options.device)
        # ``rgb`` is premultiplied dynamic colour.  Source instance RGB is a
        # foreground colour observation, so supervise the unpremultiplied
        # colour and retain silhouette/transmittance as independent terms.
        rgb = (actor["rgb"] / actor["alpha"].clamp_min(.05)[..., None] - target).abs().mean()
        silhouette = torch.nn.functional.binary_cross_entropy(actor["alpha"].clamp(1e-5, 1 - 1e-5), torch.ones_like(actor["alpha"]))
        ring_loss = torch.nn.functional.binary_cross_entropy(ring["transmittance"].clamp(1e-5, 1 - 1e-5), torch.ones_like(ring["transmittance"]))
        ring_alpha = ring["alpha"].mean()
        # A soft background BCE can be traded away against foreground colour.
        # Enforce an explicit visibility budget as well: once a safe ray has
        # more than the allowed actor alpha, its excess must be paid down.
        ring_hard = torch.relu(ring_alpha - options.ring_alpha_target).square()
        lidar = rgb.new_zeros(())
        if len(observation.lidar_xy):
            lidar_indices = rng.choice(len(observation.lidar_xy), size=min(len(observation.lidar_xy), options.actor_rays_per_step), replace=len(observation.lidar_xy) < options.actor_rays_per_step)
            lidar_pixels = observation.lidar_xy[lidar_indices]
            lidar_render, _ = _batch_render(field, observation, lidar_pixels, track, options, device=options.device)
            depth = torch.from_numpy(observation.lidar_depth[lidar_indices]).to(options.device)
            finite = torch.isfinite(lidar_render["expected_depth"]) & torch.isfinite(depth)
            if bool(finite.any()):
                lidar = (lidar_render["expected_depth"][finite] - depth[finite]).abs().mean()
        selection = torch.randint(len(anchor_tensor), (min(len(anchor_tensor), options.anchor_points_per_step),), device=options.device)
        anchor_positions, anchor_times = anchor_tensor[selection], anchor_time[selection, None]
        occupancy = 1.0 - torch.exp(-field.density_only(anchor_positions / (torch.as_tensor(track["length_width_height"], device=options.device) * .5), anchor_times) * .10)
        anchor_loss = torch.nn.functional.binary_cross_entropy(occupancy.clamp(1e-5, 1 - 1e-5), torch.ones_like(occupancy))
        time_offset = torch.full_like(anchor_times, .05)
        temporal = (field.density_only(anchor_positions / (torch.as_tensor(track["length_width_height"], device=options.device) * .5), (anchor_times + time_offset).clamp(-1., 1.)) - field.density_only(anchor_positions / (torch.as_tensor(track["length_width_height"], device=options.device) * .5), (anchor_times - time_offset).clamp(-1., 1.))).abs().mean()
        loss = options.rgb_weight * rgb + options.silhouette_weight * silhouette + options.ring_transmittance_weight * ring_loss + options.ring_hard_weight * ring_hard + options.lidar_weight * lidar + options.anchor_occupancy_weight * anchor_loss + options.temporal_anchor_weight * temporal
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 0 or (step + 1) % options.validation_every == 0 or step + 1 == options.iterations:
            # Keep the holdout ray subset fixed across checkpoints.  A moving
            # random sample would make early stopping appear to improve merely
            # because a different silhouette slice was evaluated.
            field.eval(); validation = _evaluate(field, holdout, track, options, device=options.device, rng=np.random.default_rng(options.seed + 10_003)); field.train()
            excess = max(0.0, validation["ring_alpha"] - options.max_holdout_ring_alpha)
            score = validation["rgb_mae"] + validation["actor_alpha_error"] + validation["ring_alpha"] + 5.0 * excess
            history.append({"step": float(step + 1), "loss": float(loss.detach()), "rgb": float(rgb.detach()), "silhouette": float(silhouette.detach()), "ring": float(ring_loss.detach()), "ring_alpha": float(ring_alpha.detach()), "ring_hard": float(ring_hard.detach()), "lidar": float(lidar.detach()), "anchor": float(anchor_loss.detach()), "temporal_anchor": float(temporal.detach()), "holdout_score": float(score), **{f"holdout_{key}": value for key, value in validation.items()}})
            if score < best_score:
                best_score, best_state = float(score), {key: value.detach().cpu().clone() for key, value in field.state_dict().items()}
            print(f"4D occupancy [{step + 1:05d}/{options.iterations}] loss={float(loss):.6f} rgb={float(rgb):.6f} silhouette={float(silhouette):.6f} ring={float(ring_loss):.6f} anchor={float(anchor_loss):.6f} holdout={float(score):.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("4D trainer did not complete an evaluable optimisation step")
    field.load_state_dict(best_state); field.eval()
    best_holdout = _evaluate(field, holdout, track, options, device=options.device, rng=np.random.default_rng(options.seed + 10_003))
    acceptance = {
        "passed": bool(best_holdout["rgb_mae"] <= options.max_holdout_rgb_mae and best_holdout["actor_alpha_error"] <= options.max_holdout_actor_alpha_error and best_holdout["ring_alpha"] <= options.max_holdout_ring_alpha),
        "thresholds": {"max_rgb_mae": options.max_holdout_rgb_mae, "max_actor_alpha_error": options.max_holdout_actor_alpha_error, "max_ring_alpha": options.max_holdout_ring_alpha},
        "measured": best_holdout,
    }
    payload = {"schema": SCHEMA, "version": VERSION, "state_dict": best_state, "model": {"hidden_dim": options.hidden_dim, "position_frequencies": options.position_frequencies, "time_frequencies": options.time_frequencies}, "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": _sha256(manifest_path), "correspondence": str(correspondence_path), "track_id": options.track_id, "track": track, "reference_window": [options.reference_start, options.reference_end], "temporal_even_train_odd_holdout": options.temporal_even_train_odd_holdout, "static_ply_used": False, "best_holdout_score": best_score, "acceptance": acceptance}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, options.output)
    report = {"schema": SCHEMA, "version": VERSION, "status": "complete", "checkpoint": str(options.output.resolve()), "parameters": {key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value) for key, value in vars(options).items()}, "input": {"dataset_manifest_sha256": _sha256(manifest_path), "correspondence": str(correspondence_path), "accepted_correspondence_points": int(len(anchors)), "static_ply_used": False}, "train_observations": len(train), "temporal_holdout_observations": len(holdout), "history": history, "best_holdout_score": best_score, "acceptance": acceptance, "limitations": ["This field contains only a rigid actor-local dynamic layer; it does not repair static PLY leakage.", "All RGB and silhouette supervision comes from original source masks/images; static RGB/alpha/depth and old dynamic Gaussians are excluded.", "Odd reference frames are a strict temporal holdout.  Do not render a target L4 sequence unless this gate and a separate target-view compositing audit pass.", "Unobserved actor volume remains uncertain; no generated inpainting or full-cuboid completion is used."]}
    _atomic_json(options.output.with_suffix(".report.json"), report)
    print(f"Wrote 4D occupancy/transmittance checkpoint: {options.output}", flush=True)
    return options.output
