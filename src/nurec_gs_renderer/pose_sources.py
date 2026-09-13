from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .camera import (
    CameraFrame,
    CameraIntrinsics,
    CameraPath,
    FThetaCameraParameters,
    camera_pose_to_opencv_w2c,
    compose_ncore_camera_pose,
)
from .chunking import chunk_time_range, sequence_sampling_interval


@dataclass(frozen=True)
class CameraPathSelection:
    chunk_index: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    stride: int = 1
    sample_count: int | None = None
    timestamp_start_us: int | None = None
    timestamp_end_us: int | None = None
    n_frames_per_chunk: int = 18
    max_frame_gap_timestamp_us: int = 750_000

    def __post_init__(self) -> None:
        if self.chunk_index is not None and self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_end is not None and self.frame_end < 0:
            raise ValueError("frame_end must be non-negative")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end <= self.frame_start:
            raise ValueError("frame_end must be greater than frame_start")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if (
            self.timestamp_start_us is not None
            and self.timestamp_end_us is not None
            and self.timestamp_end_us <= self.timestamp_start_us
        ):
            raise ValueError("timestamp_end_us must be greater than timestamp_start_us")
        if self.n_frames_per_chunk <= 0 or self.max_frame_gap_timestamp_us <= 0:
            raise ValueError("chunk sampling parameters must be positive")

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.chunk_index,
                self.frame_start,
                self.frame_end,
                self.sample_count,
                self.timestamp_start_us,
                self.timestamp_end_us,
            )
        ) or self.stride != 1


@dataclass(frozen=True)
class CameraPathSet:
    """One legacy camera path or a named target-camera rig."""

    paths: dict[str, CameraPath]
    rig_id: str | None = None

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("camera path set must contain at least one camera")

    @property
    def is_rig(self) -> bool:
        return self.rig_id is not None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _output_intrinsics(config: dict[str, Any]) -> CameraIntrinsics:
    intrinsics = CameraIntrinsics.from_dict(config["intrinsics"])
    output = config.get("output", {})
    return intrinsics.resized(output.get("width"), output.get("height"))


def load_world_camera_path(config: dict[str, Any]) -> CameraPath:
    intrinsics = _output_intrinsics(config)
    matrix_type = str(config.get("matrix_type", "camera_to_world"))
    camera_axes = str(config.get("camera_axes", "opencv"))
    raw_frames = config.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("world pose config requires a non-empty 'frames' list")

    frames: list[CameraFrame] = []
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, dict) or "matrix" not in raw:
            raise ValueError(f"frames[{index}] must be an object containing 'matrix'")
        frame_id = str(raw.get("frame_id", f"{index:06d}"))
        w2c = camera_pose_to_opencv_w2c(
            np.asarray(raw["matrix"], dtype=np.float64),
            matrix_type=str(raw.get("matrix_type", matrix_type)),
            camera_axes=str(raw.get("camera_axes", camera_axes)),
        )
        timestamp = raw.get("timestamp_us")
        frames.append(
            CameraFrame(
                frame_id=frame_id,
                intrinsics=intrinsics,
                world_to_camera=w2c,
                timestamp_us=None if timestamp is None else int(timestamp),
            )
        )
    return CameraPath.from_frames(frames, source="world")


def _parse_ncore_meta(ncore_path: Path) -> list[object]:
    """Resolve NCore component stores relative to the sequence JSON."""
    try:
        from upath import UPath
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("NCore mode requires: pip install 'instant-nurec-gs-renderer[ncore]'") from exc
    meta_path = UPath(str(ncore_path))
    with meta_path.open("r") as handle:
        meta = json.load(handle)
    if not str(meta.get("version", "")).startswith("v4"):
        raise ValueError(f"{ncore_path} is not an NCore V4 sequence JSON")
    return [meta_path.parent / item["path"] for item in meta["component_stores"]]


def _create_ncore_loader(ncore_path: Path):
    try:
        import ncore.data.v4
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("NCore mode requires nvidia-ncore") from exc

    paths = _parse_ncore_meta(ncore_path)
    reader = ncore.data.v4.SequenceComponentGroupsReader(paths, open_consolidated=True)
    return ncore.data.v4.SequenceLoaderV4(
        reader,
        poses_component_group_name="default",
        intrinsics_component_group_name="default",
        masks_component_group_name="default",
        cuboids_component_group_name="default",
    )


def _enum_name(value: object, field: str) -> str:
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        raise ValueError(f"NCore {field} must be an enum with a name")
    return name


def _ftheta_from_ncore(model: object) -> tuple[CameraIntrinsics, FThetaCameraParameters]:
    """Convert an NCore FTheta model without importing its concrete class."""
    required = (
        "resolution",
        "principal_point",
        "reference_poly",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
        "max_angle",
        "linear_cde",
        "shutter_type",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"NCore source camera is not an FTheta model; missing {missing}")
    if getattr(model, "external_distortion_parameters", None) is not None:
        raise ValueError("FTheta external distortion parameters are not supported by gsplat")

    resolution = np.asarray(getattr(model, "resolution"), dtype=np.int64)
    principal = np.asarray(getattr(model, "principal_point"), dtype=np.float64)
    forward = tuple(float(x) for x in np.asarray(getattr(model, "angle_to_pixeldist_poly")).tolist())
    if resolution.shape != (2,) or principal.shape != (2,):
        raise ValueError("invalid NCore FTheta resolution or principal point")
    if len(forward) != 6:
        raise ValueError("invalid NCore FTheta forward polynomial")
    # fx/fy are unused by gsplat's FTheta projection; the first-order forward
    # coefficient is retained here to make K meaningful for inspection.
    nominal_focal = abs(forward[1])
    intrinsics = CameraIntrinsics(
        fx=nominal_focal,
        fy=nominal_focal,
        cx=float(principal[0]),
        cy=float(principal[1]),
        width=int(resolution[0]),
        height=int(resolution[1]),
    )
    ftheta = FThetaCameraParameters(
        reference_poly=_enum_name(getattr(model, "reference_poly"), "reference_poly"),
        pixeldist_to_angle_poly=tuple(
            float(x) for x in np.asarray(getattr(model, "pixeldist_to_angle_poly")).tolist()
        ),
        angle_to_pixeldist_poly=forward,
        max_angle=float(getattr(model, "max_angle")),
        linear_cde=tuple(float(x) for x in np.asarray(getattr(model, "linear_cde")).tolist()),
        shutter_type=_enum_name(getattr(model, "shutter_type"), "shutter_type"),
    )
    return intrinsics, ftheta


def _select_frame_indices(frame_count: int, value: object) -> np.ndarray:
    if value is None:
        return np.arange(frame_count, dtype=np.int64)
    indices = np.asarray(value, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("frame_indices must be a non-empty one-dimensional list")
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise ValueError("frame_indices contains an out-of-range index")
    return indices


def _ncore_exposure_timestamps(sensor: object) -> np.ndarray:
    timestamps = np.asarray(getattr(sensor, "frames_timestamps_us"), dtype=np.uint64)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError("NCore camera must provide [start, end] exposure timestamps")
    return timestamps


def _select_ncore_reference_frames(
    config: dict[str, Any],
    loader: object,
    camera_id: str,
    selection: CameraPathSelection | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select reference-camera frames and retain their START/END timestamps."""
    sensor = loader.get_camera_sensor(camera_id)
    all_exposure_timestamps = _ncore_exposure_timestamps(sensor)
    selection = CameraPathSelection() if selection is None else selection
    selection_metadata: dict[str, object] = {"reference_camera_id": camera_id}
    if selection.chunk_index is None:
        source_indices = _select_frame_indices(
            len(all_exposure_timestamps), config.get("frame_indices")
        )
    else:
        camera_ids = tuple(sorted(str(value) for value in getattr(loader, "camera_ids")))
        sensors = {value: loader.get_camera_sensor(value) for value in camera_ids}
        end_timestamps = {
            value: _ncore_exposure_timestamps(value_sensor)[:, 1].astype(np.int64)
            for value, value_sensor in sensors.items()
        }
        sequence_start_us, sequence_end_us = sequence_sampling_interval(
            loader, sensors, end_timestamps
        )
        layout = chunk_time_range(
            sequence_start_us,
            sequence_end_us,
            selection.chunk_index,
            n_frames_per_chunk=selection.n_frames_per_chunk,
            max_frame_gap_timestamp_us=selection.max_frame_gap_timestamp_us,
        )
        source_end = all_exposure_timestamps[:, 1].astype(np.int64)
        end_inclusive = layout.chunk_index + 1 == layout.chunk_count
        in_chunk = (source_end >= layout.start_us) & (
            source_end <= layout.end_us if end_inclusive else source_end < layout.end_us
        )
        source_indices = np.flatnonzero(in_chunk).astype(np.int64)
        selection_metadata["chunk"] = {
            "index": layout.chunk_index,
            "count": layout.chunk_count,
            "sequence_start_us": layout.sequence_start_us,
            "sequence_end_us": layout.sequence_end_us,
            "start_us": layout.start_us,
            "end_us": layout.end_us,
            "end_inclusive": end_inclusive,
            "n_frames_per_sample": selection.n_frames_per_chunk,
            "max_frame_gap_timestamp_us": selection.max_frame_gap_timestamp_us,
            "reference_timestamps_us": list(layout.reference_timestamps_us),
            "camera_ids": list(camera_ids),
        }

    selected_end = all_exposure_timestamps[source_indices, 1].astype(np.int64)
    keep = np.ones(len(source_indices), dtype=bool)
    if selection.timestamp_start_us is not None:
        keep &= selected_end >= selection.timestamp_start_us
    if selection.timestamp_end_us is not None:
        keep &= selected_end < selection.timestamp_end_us
    source_indices = source_indices[keep]
    source_indices = source_indices[slice(selection.frame_start, selection.frame_end, selection.stride)]
    if selection.sample_count is not None and len(source_indices) > selection.sample_count:
        positions = np.linspace(0, len(source_indices) - 1, selection.sample_count)
        source_indices = source_indices[np.round(positions).astype(np.int64)]
    if len(source_indices) == 0:
        raise ValueError("camera path selection resolved to no NCore reference frames")

    exposure_timestamps = all_exposure_timestamps[source_indices]
    end_values = exposure_timestamps[:, 1].astype(np.int64)
    gaps = np.diff(end_values)
    selection_metadata["selection"] = {
        "source_frame_indices": [int(value) for value in source_indices],
        "frame_start": selection.frame_start,
        "frame_end": selection.frame_end,
        "stride": selection.stride,
        "sample_count": selection.sample_count,
        "timestamp_start_us": selection.timestamp_start_us,
        "timestamp_end_us": selection.timestamp_end_us,
        "selected_frame_count": len(source_indices),
        "selected_timestamp_start_us": int(end_values[0]),
        "selected_timestamp_end_us": int(end_values[-1]),
        "frame_gap_us": None
        if len(gaps) == 0
        else {
            "minimum": int(gaps.min()),
            "median": float(np.median(gaps)),
            "maximum": int(gaps.max()),
        },
    }
    return source_indices, exposure_timestamps, selection_metadata


def _load_ncore_source_camera_path(
    config: dict[str, Any],
    loader: object,
    camera_id: str,
    selection: CameraPathSelection | None = None,
) -> CameraPath:
    """Reproduce an NCore source camera, including FTheta and rolling shutter."""
    if config.get("timestamps_us") is not None:
        raise ValueError("ncore_camera_id mode uses source frame_indices, not timestamps_us")
    sensor = loader.get_camera_sensor(camera_id)
    intrinsics, ftheta = _ftheta_from_ncore(sensor.model_parameters)
    source_indices, exposure_timestamps, selection_metadata = _select_ncore_reference_frames(
        config, loader, camera_id, selection
    )

    flat_timestamps = exposure_timestamps.reshape(-1)
    poses = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", flat_timestamps))
    if poses.shape != (len(flat_timestamps), 4, 4):
        raise ValueError("NCore pose graph returned an unexpected pose array")
    poses = poses.reshape(len(exposure_timestamps), 2, 4, 4)
    T_camera_rig = np.asarray(sensor.T_sensor_rig, dtype=np.float64)

    frames: list[CameraFrame] = []
    for frame_index, timestamps, (T_rig_world_start, T_rig_world_end) in zip(
        source_indices, exposure_timestamps, poses, strict=True
    ):
        start_us, end_us = (int(timestamps[0]), int(timestamps[1]))
        start_c2w = compose_ncore_camera_pose(T_rig_world_start, T_camera_rig)
        end_c2w = compose_ncore_camera_pose(T_rig_world_end, T_camera_rig)
        frames.append(
            CameraFrame(
                frame_id=f"{int(frame_index):06d}_{end_us}",
                intrinsics=intrinsics,
                world_to_camera=camera_pose_to_opencv_w2c(start_c2w),
                timestamp_us=end_us,
                camera_model="ftheta",
                ftheta=ftheta,
                world_to_camera_end=camera_pose_to_opencv_w2c(end_c2w),
                timestamp_start_us=start_us,
            )
        )
    path = CameraPath.from_frames(
        frames,
        source=f"ncore:{camera_id}",
        metadata=selection_metadata,
    )
    output = config.get("output", {})
    return path.resized(output.get("width"), output.get("height"))


def _load_ncore_camera_path(
    config: dict[str, Any],
    loader: object,
    selection: CameraPathSelection | None = None,
) -> CameraPath:
    """Build target-camera poses from NCore rig motion and installation extrinsics."""
    source_camera_id = config.get("ncore_camera_id")
    if source_camera_id is not None:
        return _load_ncore_source_camera_path(
            config, loader, str(source_camera_id), selection=selection
        )
    intrinsics = _output_intrinsics(config)

    timestamps_value = config.get("timestamps_us")
    if timestamps_value is not None:
        if selection is not None and selection.active:
            raise ValueError("sequence selection cannot be combined with explicit timestamps_us")
        timestamps = np.asarray(timestamps_value, dtype=np.uint64)
        source_indices: np.ndarray | None = None
        selection_metadata: dict[str, object] = {}
    else:
        reference_camera_id = str(
            config.get("timestamp_camera_id", "camera_front_wide_120fov")
        )
        source_indices, exposure_timestamps, selection_metadata = _select_ncore_reference_frames(
            config, loader, reference_camera_id, selection
        )
        pose_timepoint = str(config.get("pose_timepoint", "end"))
        if pose_timepoint == "start":
            timestamps = exposure_timestamps[:, 0]
        elif pose_timepoint == "midpoint":
            timestamps = (
                exposure_timestamps[:, 0].astype(np.uint64)
                + (exposure_timestamps[:, 1] - exposure_timestamps[:, 0]) // 2
            )
        elif pose_timepoint == "end":
            timestamps = exposure_timestamps[:, 1]
        else:
            raise ValueError("pose_timepoint must be 'start', 'midpoint' or 'end'")
        selection_metadata["target_pose_timepoint"] = pose_timepoint

    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("NCore camera path resolved to no timestamps")

    T_camera_rig = np.asarray(config["T_camera_rig"], dtype=np.float64)
    T_rig_worlds = loader.pose_graph.evaluate_poses("rig", "world", timestamps)
    frames = []
    for index, (timestamp, T_rig_world) in enumerate(zip(timestamps, T_rig_worlds, strict=True)):
        c2w = compose_ncore_camera_pose(T_rig_world, T_camera_rig)
        source_index = index if source_indices is None else int(source_indices[index])
        frames.append(
            CameraFrame(
                frame_id=f"{source_index:06d}_{int(timestamp)}",
                intrinsics=intrinsics,
                world_to_camera=camera_pose_to_opencv_w2c(c2w),
                timestamp_us=int(timestamp),
            )
        )
    return CameraPath.from_frames(
        frames,
        source="ncore",
        metadata=selection_metadata,
    )


def load_ncore_camera_path(
    config: dict[str, Any],
    config_path: Path,
    selection: CameraPathSelection | None = None,
) -> CameraPath:
    raw_ncore_path = Path(str(config["ncore_path"]))
    ncore_path = raw_ncore_path if raw_ncore_path.is_absolute() else config_path.parent / raw_ncore_path
    loader = _create_ncore_loader(ncore_path.resolve())
    return _load_ncore_camera_path(config, loader, selection=selection)


_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def load_camera_path_set(
    path: str | Path,
    selection: CameraPathSelection | None = None,
    camera_ids: set[str] | None = None,
) -> CameraPathSet:
    """Load a legacy single-camera config or a named NCore target-camera rig."""
    path = Path(path)
    config = _read_json(path)
    raw_cameras = config.get("cameras")
    if raw_cameras is None:
        if camera_ids:
            raise ValueError("--camera-id is only valid for a multi-camera rig config")
        return CameraPathSet(paths={"camera": load_camera_path(path, selection=selection)})
    if str(config.get("pose_source", "world")) != "ncore":
        raise ValueError("multi-camera rig configs currently require pose_source='ncore'")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("multi-camera rig config requires a non-empty 'cameras' list")

    raw_ncore_path = Path(str(config["ncore_path"]))
    ncore_path = raw_ncore_path if raw_ncore_path.is_absolute() else path.parent / raw_ncore_path
    loader = _create_ncore_loader(ncore_path.resolve())
    rig_id = str(config.get("rig_id", path.stem))
    shared = {key: value for key, value in config.items() if key != "cameras"}
    paths: dict[str, CameraPath] = {}
    available_ids: set[str] = set()
    for index, raw_camera in enumerate(raw_cameras):
        if not isinstance(raw_camera, dict):
            raise ValueError(f"cameras[{index}] must be an object")
        camera_id = str(raw_camera.get("camera_id", ""))
        if not _CAMERA_ID_PATTERN.fullmatch(camera_id):
            raise ValueError(
                f"cameras[{index}].camera_id must contain only letters, digits, '_' or '-'"
            )
        if camera_id in available_ids:
            raise ValueError(f"duplicate camera_id: {camera_id}")
        available_ids.add(camera_id)
        if camera_ids is not None and camera_id not in camera_ids:
            continue
        camera_config = dict(shared)
        camera_config.update(raw_camera)
        camera_path = _load_ncore_camera_path(camera_config, loader, selection=selection)
        metadata = dict(camera_path.metadata or {})
        metadata["target_rig"] = {
            "rig_id": rig_id,
            "camera_id": camera_id,
            "T_camera_rig": camera_config["T_camera_rig"],
        }
        paths[camera_id] = CameraPath.from_frames(
            camera_path.frames,
            source=f"ncore-target:{rig_id}:{camera_id}",
            metadata=metadata,
        )
    if camera_ids is not None:
        missing = camera_ids - available_ids
        if missing:
            raise ValueError(f"unknown camera_id(s): {', '.join(sorted(missing))}")
    if not paths:
        raise ValueError("camera selection resolved to no target cameras")
    return CameraPathSet(paths=paths, rig_id=rig_id)


def load_camera_path(
    path: str | Path,
    selection: CameraPathSelection | None = None,
) -> CameraPath:
    path = Path(path)
    config = _read_json(path)
    if "cameras" in config:
        raise ValueError("multi-camera rig config must be loaded with load_camera_path_set()")
    pose_source = str(config.get("pose_source", "world"))
    if pose_source == "world":
        if selection is not None and selection.active:
            raise ValueError("sequence selection currently requires an NCore source camera")
        return load_world_camera_path(config)
    if pose_source == "ncore":
        return load_ncore_camera_path(config, path, selection=selection)
    raise ValueError("pose_source must be 'world' or 'ncore'")
