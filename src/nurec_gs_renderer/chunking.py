from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ChunkTimeRange:
    chunk_index: int
    chunk_count: int
    sequence_start_us: int
    sequence_end_us: int
    start_us: int
    end_us: int
    reference_timestamps_us: tuple[int, ...]


def chunk_time_range(
    sequence_start_us: int,
    sequence_end_us: int,
    chunk_index: int,
    *,
    n_frames_per_chunk: int = 18,
    max_frame_gap_timestamp_us: int = 750_000,
) -> ChunkTimeRange:
    """Return the equal-time chunk layout used by InstantNuRec sampling."""
    if n_frames_per_chunk <= 0 or max_frame_gap_timestamp_us <= 0:
        raise ValueError("chunk sampling parameters must be positive")
    total_timespan = int(sequence_end_us) - int(sequence_start_us)
    if total_timespan <= 0:
        raise ValueError("sequence time interval must be non-empty")
    max_chunk_timespan = max_frame_gap_timestamp_us * n_frames_per_chunk
    chunk_count = max(1, int(math.ceil(total_timespan / max_chunk_timespan)))
    if not 0 <= chunk_index < chunk_count:
        raise ValueError(f"chunk_index {chunk_index} is out of range for {chunk_count} chunk(s)")
    frame_gap = total_timespan / (chunk_count * n_frames_per_chunk)
    first_reference = chunk_index * n_frames_per_chunk
    references = tuple(
        int(sequence_start_us + (first_reference + offset) * frame_gap)
        for offset in range(n_frames_per_chunk)
    )
    start = int(sequence_start_us + chunk_index * n_frames_per_chunk * frame_gap)
    end = (
        int(sequence_end_us)
        if chunk_index + 1 == chunk_count
        else int(sequence_start_us + (chunk_index + 1) * n_frames_per_chunk * frame_gap)
    )
    return ChunkTimeRange(
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        sequence_start_us=int(sequence_start_us),
        sequence_end_us=int(sequence_end_us),
        start_us=start,
        end_us=end,
        reference_timestamps_us=references,
    )


def sample_chunk_frame_indices(
    camera_timestamps_us: dict[str, np.ndarray],
    sequence_start_us: int,
    sequence_end_us: int,
    chunk_index: int,
    *,
    n_frames_per_chunk: int = 18,
    max_frame_gap_timestamp_us: int = 750_000,
) -> tuple[dict[str, list[int]], list[int], int]:
    """Match InstantNuRec's AdaptiveSequentialFrameBatchSampler."""
    if not camera_timestamps_us:
        raise ValueError("no camera timestamps were provided")
    layout = chunk_time_range(
        sequence_start_us,
        sequence_end_us,
        chunk_index,
        n_frames_per_chunk=n_frames_per_chunk,
        max_frame_gap_timestamp_us=max_frame_gap_timestamp_us,
    )
    sampled: dict[str, list[int]] = {}
    for camera_id, timestamps in camera_timestamps_us.items():
        values = np.asarray(timestamps, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"camera {camera_id} has no one-dimensional timestamps")
        sampled[camera_id] = [
            int(np.abs(values - reference).argmin())
            for reference in layout.reference_timestamps_us
        ]
    return sampled, list(layout.reference_timestamps_us), layout.chunk_count


def sequence_sampling_interval(
    loader: object,
    sensors: dict[str, object],
    timestamps: dict[str, np.ndarray],
) -> tuple[int, int]:
    edge = loader.pose_graph.get_edge("rig", "world")
    pose_timestamps = np.asarray(edge.timestamps_us, dtype=np.int64)
    if pose_timestamps.ndim != 1 or len(pose_timestamps) == 0:
        raise ValueError("NCore rig-to-world pose edge has no timestamps")
    start = int(pose_timestamps.min())
    end = int(pose_timestamps.max())
    for camera_id in sensors:
        values = np.asarray(timestamps[camera_id], dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"camera {camera_id} has no timestamps")
        start = max(start, int(values.min()) - 100_000)
        end = min(end, int(values.max()) + 100_000)
    if end <= start:
        raise ValueError("NCore camera and pose time ranges do not overlap")
    return start, end
