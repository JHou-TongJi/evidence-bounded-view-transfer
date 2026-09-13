"""Conservative bidirectional SAM2 video propagation for dynamic actor masks.

The source masks remain authoritative observations.  SAM2 proposals are used
only to bridge temporal gaps after forward/reverse agreement and exact NCore
FTheta cuboid containment checks; this module never changes the source dataset
or a Gaussian asset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .actor_audit import _ftheta_cuboid_mask
from .camera import compose_ncore_camera_pose
from .dynamic_reconstruction_dataset import sparse_depth_inside_mask
from .pose_sources import _create_ncore_loader
from .static_leakage import _project_lidar_depth_ftheta

SCHEMA = "ncore-dynamic-reconstruction-mask-refinement"
VERSION = 1


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    if first.shape != second.shape:
        raise ValueError("mask shapes must match")
    union = np.logical_or(first, second).sum()
    return 1.0 if not union else float(np.logical_and(first, second).sum() / union)


def select_anchor_positions(positions: Iterable[int], stride: int) -> list[int]:
    values = sorted({int(item) for item in positions})
    if stride <= 0:
        raise ValueError("anchor stride must be positive")
    if not values:
        return []
    output = values[::stride]
    if output[-1] != values[-1]:
        output.append(values[-1])
    return output


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


@dataclass(frozen=True)
class MaskDecision:
    mask: np.ndarray
    status: str
    directional_iou: float
    seed_iou: float | None
    confidence: float


def refine_bidirectional_mask(
    forward_logits: np.ndarray, backward_logits: np.ndarray, cuboid_mask: np.ndarray, *, source_mask: np.ndarray | None,
    probability_threshold: float, minimum_directional_iou: float, minimum_seed_iou: float,
    minimum_mask_pixels: int, maximum_cuboid_fill_fraction: float,
) -> MaskDecision:
    """Accept only an F/B-consistent proposal; keep a valid source mask on failure."""
    forward, backward, cuboid = (np.asarray(value, np.float32) for value in (forward_logits, backward_logits, cuboid_mask))
    if forward.shape != backward.shape or forward.shape != cuboid.shape:
        raise ValueError("SAM logits and cuboid mask must have matching shapes")
    cuboid = cuboid.astype(bool)
    f_prob, b_prob = _sigmoid(forward), _sigmoid(backward)
    f_mask, b_mask = (f_prob >= probability_threshold) & cuboid, (b_prob >= probability_threshold) & cuboid
    directional_iou = mask_iou(f_mask, b_mask)
    probability = (f_prob + b_prob) * 0.5
    candidate = (probability >= probability_threshold) & cuboid
    seed = None if source_mask is None else np.asarray(source_mask, bool) & cuboid
    seed_iou = None if seed is None else mask_iou(candidate, seed)
    pixels, cuboid_pixels = int(candidate.sum()), int(cuboid.sum())
    valid = (
        directional_iou >= minimum_directional_iou and pixels >= minimum_mask_pixels
        and (cuboid_pixels == 0 or pixels / cuboid_pixels <= maximum_cuboid_fill_fraction)
        and (seed is None or seed_iou >= minimum_seed_iou)
    )
    confidence = float(probability[candidate].mean()) if pixels else 0.0
    if valid:
        return MaskDecision(candidate, "accepted_bidirectional", directional_iou, seed_iou, confidence)
    if seed is not None and seed.sum() >= minimum_mask_pixels:
        return MaskDecision(seed, "source_fallback", directional_iou, seed_iou, 1.0)
    return MaskDecision(np.zeros_like(cuboid), "rejected", directional_iou, seed_iou, confidence)


def resolve_instance_overlaps(masks: dict[int, np.ndarray], scores: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Prevent a pixel from supervising two actor IDs."""
    if not masks:
        return {}
    ids = sorted(masks)
    shape = np.asarray(masks[ids[0]], bool).shape
    if any(np.asarray(value, bool).shape != shape for value in masks.values()):
        raise ValueError("all actor masks must have the same shape")
    stacked = np.stack([np.where(masks[track_id], scores[track_id], -np.inf) for track_id in ids])
    owner = np.argmax(stacked, axis=0)
    return {track_id: np.asarray(masks[track_id], bool) & (owner == index) for index, track_id in enumerate(ids)}


def merge_source_priority_masks(
    source_masks: dict[int, np.ndarray], proposed_masks: dict[int, np.ndarray], proposed_scores: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Keep every original observation byte-for-byte; arbitrate only novel pixels.

    Source masks can overlap in a conservative semantic dataset.  They must
    remain available to a later depth-aware trainer rather than being deleted
    by a 2-D actor arbitration pass.  A SAM2-only proposal cannot occupy any
    source pixel belonging to another actor.
    """
    if not source_masks and not proposed_masks:
        return {}
    all_masks = list(source_masks.values()) + list(proposed_masks.values())
    shape = np.asarray(all_masks[0], bool).shape
    source_union = np.zeros(shape, bool)
    for value in source_masks.values():
        source_union |= np.asarray(value, bool)
    novel = {
        track_id: np.asarray(mask, bool) & ~source_union
        for track_id, mask in proposed_masks.items()
        if track_id not in source_masks
    }
    novel = resolve_instance_overlaps(novel, {track_id: proposed_scores[track_id] for track_id in novel})
    return {**{track_id: np.asarray(mask, bool) for track_id, mask in source_masks.items()}, **novel}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L"), np.uint8) > 0
    if value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match {shape}: {path}")
    return value


def _camera_record(frame: dict[str, Any], camera_id: str) -> dict[str, Any]:
    for value in frame["cameras"]:
        if value["camera_id"] == camera_id:
            return value
    raise ValueError(f"source manifest frame has no camera={camera_id}")


def _instances(frame: dict[str, Any], camera_id: str) -> dict[int, dict[str, Any]]:
    return {int(value["track_id"]): value for value in _camera_record(frame, camera_id).get("instances", [])}


def _predictions(value: Any, count: int, shape: tuple[int, int]) -> np.ndarray:
    # Transformers returns a CUDA tensor when the inference session is on GPU.
    # The following confidence gates/write path are intentionally CPU/Numpy.
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, np.float32)
    while value.ndim > 3 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 2:
        value = value[None]
    if value.shape != (count, *shape):
        raise ValueError(f"unexpected SAM2 mask shape {value.shape}, expected {(count, *shape)}")
    return value


def _load_sam2(model_dir: Path, device: str):
    try:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("requires transformers==4.57.1 with SAM2 video support") from exc
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SAM2 Transformers model is missing: {model_dir}")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = Sam2VideoModel.from_pretrained(str(model_dir), local_files_only=True).to(device, dtype=dtype).eval()
    return torch, model, Sam2VideoProcessor.from_pretrained(str(model_dir), local_files_only=True), dtype


def _session(processor: Any, images: list[Image.Image], device: str, dtype: Any, cache_size: int) -> Any:
    return processor.init_video_session(
        video=images, inference_device=device, video_storage_device="cpu",
        max_vision_features_cache_size=cache_size, dtype=dtype,
    )


def _add_prompts(model: Any, processor: Any, session: Any, anchors: dict[int, dict[int, np.ndarray]]) -> None:
    for position, values in anchors.items():
        ids = sorted(values)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=position, obj_ids=ids,
            input_masks=[values[track_id].astype(np.uint8) for track_id in ids],
        )
        # ``add_inputs`` only queues a prompt.  Materialise its conditional
        # output before propagation, as required by the video-session API;
        # otherwise a reverse pass can start from an empty memory state.
        model(session, frame_idx=position)


def _propagate(
    model: Any, processor: Any, session: Any, *, reverse: bool, count: int, shape: tuple[int, int], start: int,
) -> dict[int, dict[int, np.ndarray]]:
    """Propagate from an actual prompt frame, never from an unconditioned edge."""
    if not 0 <= start < count:
        raise ValueError("SAM2 propagation start must be a video-frame position")
    output: dict[int, dict[int, np.ndarray]] = {}
    for prediction in model.propagate_in_video_iterator(session, start_frame_idx=start, max_frame_num_to_track=count, reverse=reverse):
        ids = [int(value) for value in session.obj_ids]
        masks = processor.post_process_masks([prediction.pred_masks], original_sizes=[[shape[0], shape[1]]], binarize=False)[0]
        output[int(prediction.frame_idx)] = dict(zip(ids, _predictions(masks, len(ids), shape)))
    return output


def _midpoint_pose(loader: Any, sensor: Any, timestamp_us: int) -> np.ndarray:
    rig = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([timestamp_us], dtype=np.uint64)))[0]
    return compose_ncore_camera_pose(rig, np.asarray(sensor.T_sensor_rig))


def _cuboid(track: dict[str, Any], pose: np.ndarray, timestamp_us: int, model: Any, shape: tuple[int, int], margin: int, device: str) -> np.ndarray:
    height, width = shape
    return _ftheta_cuboid_mask(
        [track], timestamp_us, pose, model, height=height, width=width, source_height=height, source_width=width,
        margin_pixels=margin, device=device,
    ).astype(bool)


def _preview(image: np.ndarray, source: np.ndarray, refined: np.ndarray) -> np.ndarray:
    canvas = np.asarray(image, np.float32).copy()
    canvas[source] = canvas[source] * .35 + np.asarray((255, 40, 40), np.float32) * .65
    canvas[refined] = canvas[refined] * .35 + np.asarray((40, 255, 40), np.float32) * .65
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class DynamicMaskRefinementOptions:
    dataset_manifest: Path
    model_dir: Path
    output: Path
    ncore_path: Path | None = None
    camera_ids: tuple[str, ...] | None = None
    track_ids: tuple[int, ...] | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    device: str = "cuda"
    anchor_stride: int = 15
    probability_threshold: float = .50
    minimum_directional_iou: float = .35
    minimum_seed_iou: float = .05
    minimum_mask_pixels: int = 24
    maximum_cuboid_fill_fraction: float = .92
    cuboid_margin_pixels: int | None = None
    max_vision_features_cache_size: int = 2
    write_previews: bool = False
    preview_stride: int = 15
    write_lidar_depth: bool = True
    max_lidar_timestamp_delta_us: int = 100_000
    resume: bool = False

    def __post_init__(self) -> None:
        if self.anchor_stride <= 0 or self.minimum_mask_pixels <= 0 or self.preview_stride <= 0 or self.max_vision_features_cache_size <= 0:
            raise ValueError("strides, mask pixels and cache size must be positive")
        if self.max_lidar_timestamp_delta_us <= 0:
            raise ValueError("max_lidar_timestamp_delta_us must be positive")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_start is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must exceed frame_start")
        if self.camera_ids is not None and not self.camera_ids:
            raise ValueError("camera_ids must be non-empty")
        if self.track_ids is not None and (not self.track_ids or len(set(self.track_ids)) != len(self.track_ids)):
            raise ValueError("track_ids must be unique and non-empty")
        if not 0 < self.probability_threshold < 1 or not 0 <= self.minimum_directional_iou <= 1 or not 0 <= self.minimum_seed_iou <= 1:
            raise ValueError("invalid confidence/IoU threshold")
        if not 0 <= self.maximum_cuboid_fill_fraction <= 1 or (self.cuboid_margin_pixels is not None and self.cuboid_margin_pixels < 0):
            raise ValueError("invalid cuboid gate")


def refine_dynamic_instance_masks(options: DynamicMaskRefinementOptions) -> Path:
    """Write an independent mask sidecar; data and original masks stay immutable."""
    source_path = options.dataset_manifest.resolve()
    source = _read_json(source_path)
    if source.get("schema") != "ncore-dynamic-reconstruction-dataset" or source.get("status") != "complete":
        raise ValueError("dataset-manifest must be a complete dynamic reconstruction dataset")
    root, output = source_path.parent, options.output.resolve()
    ncore_path = (options.ncore_path or Path(source["ncore_path"])).resolve()
    model_dir = options.model_dir.resolve()
    cameras = tuple(source["camera_ids"] if options.camera_ids is None else options.camera_ids)
    if set(cameras).difference(source["camera_ids"]):
        raise ValueError("a requested camera is not present in source dataset")
    tracks = {int(value["track_id"]): value for value in source["tracks"]}
    ids = tuple(sorted(tracks if options.track_ids is None else options.track_ids))
    if set(ids).difference(tracks):
        raise ValueError("a requested track is not present in source dataset")
    frames = list(source["frames"])[slice(options.frame_start, options.frame_end)]
    if not frames:
        raise ValueError("frame window is empty")
    margin = int(source["parameters"]["cuboid_margin_pixels"] if options.cuboid_margin_pixels is None else options.cuboid_margin_pixels)
    partial_path, final_path = output / "manifest.partial.json", output / "manifest.json"
    if output.exists() and any(output.iterdir()) and not options.resume:
        raise FileExistsError(f"output directory is not empty: {output}; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    source_hash = _sha256(source_path)
    if options.resume:
        partial = _read_json(partial_path)
        if partial.get("source_manifest_sha256") != source_hash or partial.get("model_dir") != str(model_dir):
            raise ValueError("partial output does not match source manifest/model")
        completed = {item["camera_id"]: item for item in partial.get("cameras", [])}

    parameters = asdict(options)
    for key, value in tuple(parameters.items()):
        if isinstance(value, Path): parameters[key] = str(value.resolve())
        if isinstance(value, tuple): parameters[key] = list(value)
    parameters["effective_cuboid_margin_pixels"] = margin
    def manifest(status: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "version": VERSION, "status": status, "source_dataset": str(root),
            "source_manifest": str(source_path), "source_manifest_sha256": source_hash, "ncore_path": str(ncore_path),
            "model_dir": str(model_dir), "model_loader": "transformers SAM2 local_files_only", "camera_ids": list(cameras),
            "track_ids": list(ids), "source_reference_frame_indices": [int(frame["reference_frame_index"]) for frame in frames],
            "parameters": parameters, "cameras": [completed[camera] for camera in cameras if camera in completed],
            "limitations": ["Training supervision only, not a dynamic Gaussian asset.", "Masks pass bidirectional, cuboid and source-consistency gates."],
        }

    loader = _create_ncore_loader(ncore_path)
    sensors = {camera: loader.get_camera_sensor(camera) for camera in cameras}
    lidar_sensor = loader.get_lidar_sensor("lidar_top_360fov") if options.write_lidar_depth else None
    from ncore.impl.sensors.camera import FThetaCameraModel
    ftheta = {camera: FThetaCameraModel(sensor.model_parameters, device=options.device) for camera, sensor in sensors.items()}
    torch, sam, processor, dtype = _load_sam2(model_dir, options.device)
    print(f"Refining masks: frames={len(frames)} cameras={len(cameras)} tracks={len(ids)}", flush=True)
    for ordinal, camera in enumerate(cameras, 1):
        if camera in completed:
            print(f"[{ordinal}/{len(cameras)}] camera={camera} resumed", flush=True); continue
        started = time.perf_counter()
        records = [_camera_record(frame, camera) for frame in frames]
        width, height = (int(value) for value in records[0]["source_resolution"])
        shape = (height, width)
        if any(tuple(record["source_resolution"]) != (width, height) for record in records):
            raise ValueError(f"camera {camera} changes resolution")
        images = [Image.fromarray(np.asarray(sensors[camera].get_frame_image_array(int(record["source_frame_index"])), np.uint8), mode="RGB") for record in records]
        original = [_instances(frame, camera) for frame in frames]
        available = {track: [index for index, values in enumerate(original) if track in values] for track in ids}
        active = [track for track in ids if available[track]]
        if not active:
            completed[camera] = {"camera_id": camera, "status": "no_source_observations", "frame_count": len(frames), "observations": []}
            _atomic_json(partial_path, manifest("partial")); continue
        anchors: dict[int, dict[int, np.ndarray]] = {}
        for track in active:
            for position in select_anchor_positions(available[track], options.anchor_stride):
                anchors.setdefault(position, {})[track] = _mask(root / original[position][track]["mask"], shape)
        with torch.inference_mode():
            session = _session(processor, images, options.device, dtype, options.max_vision_features_cache_size)
            _add_prompts(sam, processor, session, anchors)
            first_anchor, last_anchor = min(anchors), max(anchors)
            forward = _propagate(
                sam, processor, session, reverse=False, count=len(frames), shape=shape, start=first_anchor,
            )
            del session
            if str(options.device).startswith("cuda"): torch.cuda.empty_cache()
            session = _session(processor, images, options.device, dtype, options.max_vision_features_cache_size)
            _add_prompts(sam, processor, session, anchors)
            backward = _propagate(
                sam, processor, session, reverse=True, count=len(frames), shape=shape, start=last_anchor,
            )
            del session
            if str(options.device).startswith("cuda"): torch.cuda.empty_cache()
        observations, status_counts = [], {}
        for position, (frame, record, sources) in enumerate(zip(frames, records, original)):
            reference, timestamp = int(frame["reference_frame_index"]), int(record["timestamp_midpoint_us"])
            pose = _midpoint_pose(loader, sensors[camera], timestamp)
            source_masks, proposed_masks, scores, details = {}, {}, {}, {}
            source_union = np.zeros(shape, bool)
            for track in active:
                source_mask = _mask(root / sources[track]["mask"], shape) if track in sources else None
                if source_mask is not None:
                    source_union |= source_mask
                    source_masks[track] = source_mask
                # Outside the interval bracketed by actual prompts there is no
                # bidirectional evidence.  Preserve an observation, but never
                # extrapolate a new actor mask past that boundary.
                if (
                    position not in forward or position not in backward
                    or track not in forward[position] or track not in backward[position]
                ):
                    if source_mask is not None and source_mask.sum() >= options.minimum_mask_pixels:
                        details[track] = MaskDecision(source_mask, "source_fallback", 0.0, None, 1.0)
                    else:
                        details[track] = MaskDecision(np.zeros(shape, bool), "rejected_unbracketed", 0.0, None, 0.0)
                    continue
                decision = refine_bidirectional_mask(
                    forward[position][track], backward[position][track], _cuboid(tracks[track], pose, timestamp, ftheta[camera], shape, margin, options.device),
                    source_mask=source_mask, probability_threshold=options.probability_threshold,
                    minimum_directional_iou=options.minimum_directional_iou, minimum_seed_iou=options.minimum_seed_iou,
                    minimum_mask_pixels=options.minimum_mask_pixels, maximum_cuboid_fill_fraction=options.maximum_cuboid_fill_fraction,
                )
                details[track] = decision
                if decision.mask.any() and source_mask is None:
                    proposed_masks[track] = decision.mask
                    scores[track] = (_sigmoid(forward[position][track]) + _sigmoid(backward[position][track])) * .5
            final_masks = merge_source_priority_masks(source_masks, proposed_masks, scores)
            lidar_depth = None
            if lidar_sensor is not None:
                reference_time = int(frame["reference_timestamp_end_us"])
                lidar_index = int(lidar_sensor.get_closest_frame_index(reference_time))
                lidar_time = int(lidar_sensor.get_frame_timestamp_us(lidar_index))
                if abs(lidar_time - reference_time) <= options.max_lidar_timestamp_delta_us:
                    cloud = lidar_sensor.get_frame_point_cloud(lidar_index, False, False).xyz_m_end
                    lidar_to_world = np.asarray(lidar_sensor.get_frames_T_sensor_target("world", lidar_index), dtype=np.float64)
                    lidar_world = np.asarray(cloud, dtype=np.float32) @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
                    lidar_depth = _project_lidar_depth_ftheta(
                        lidar_world, np.linalg.inv(pose), ftheta[camera], source_width=width, source_height=height,
                        width=width, height=height, device=options.device,
                    )
            refined_union = np.zeros(shape, bool)
            for track, decision in details.items():
                result = final_masks.get(track, np.zeros(shape, bool))
                status = decision.status
                mask_origin = "source" if track in source_masks else ("sam2" if result.any() else "none")
                if mask_origin == "sam2" and status == "accepted_bidirectional" and result.sum() < options.minimum_mask_pixels:
                    status = "rejected_overlap"
                status_counts[status] = status_counts.get(status, 0) + 1
                entry = {"reference_frame_index": reference, "source_frame_index": int(record["source_frame_index"]), "track_id": track,
                         "status": status, "source_mask_pixels": int(sources[track]["mask_pixels"]) if track in sources else 0,
                         "refined_mask_pixels": int(result.sum()), "mask_origin": mask_origin,
                         "directional_iou": decision.directional_iou,
                         "seed_iou": decision.seed_iou, "mean_confidence": decision.confidence}
                if result.any() and status != "rejected_overlap":
                    relative = Path("instance_masks") / camera / f"{reference:06d}" / f"track_{track}.png"
                    target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(result.astype(np.uint8) * 255, mode="L").save(target)
                    entry["refined_mask"] = relative.as_posix(); refined_union |= result
                    if lidar_depth is not None:
                        xy, depth = sparse_depth_inside_mask(lidar_depth, result)
                        entry["lidar_pixels"] = int(len(depth))
                        if len(depth):
                            depth_relative = Path("lidar_depth") / camera / f"{reference:06d}" / f"track_{track}.npz"
                            depth_target = output / depth_relative; depth_target.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(depth_target, xy=xy, depth_m=depth)
                            entry["lidar_depth"] = depth_relative.as_posix()
                    elif options.write_lidar_depth:
                        entry["lidar_pixels"] = 0
                observations.append(entry)
            if options.write_previews and position % options.preview_stride == 0:
                target = output / "previews" / camera / f"{reference:06d}.png"; target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_preview(np.asarray(images[position]), source_union, refined_union), mode="RGB").save(target)
        completed[camera] = {"camera_id": camera, "status": "complete", "frame_count": len(frames), "track_ids": active,
                             "anchor_counts": {str(track): len(select_anchor_positions(available[track], options.anchor_stride)) for track in active},
                             "status_counts": status_counts, "observations": observations, "seconds": time.perf_counter() - started}
        _atomic_json(partial_path, manifest("partial"))
        accepted = sum(item["refined_mask_pixels"] > 0 for item in observations)
        print(f"[{ordinal}/{len(cameras)}] camera={camera} accepted={accepted}/{len(observations)} {status_counts}", flush=True)
    _atomic_json(final_path, manifest("complete")); partial_path.unlink(missing_ok=True)
    print(f"Wrote bidirectionally refined masks: {final_path}", flush=True)
    return final_path
