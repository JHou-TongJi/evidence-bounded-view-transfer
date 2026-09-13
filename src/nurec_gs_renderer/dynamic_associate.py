from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color_correction import file_sha256
from .dynamic import DynamicGaussianScene, infer_dynamic_source_slots
from .dynamic_association import DynamicActorAssociation
from .sky_build import _create_ncore_loader


@dataclass(frozen=True)
class DynamicAssociateOptions:
    dynamic_path: Path
    ncore_path: Path
    output: Path
    query_margin_us: int = 1_000_000
    minimum_rig_distance_m: float = 3.0
    minimum_travel_distance_m: float = 1.5
    cuboid_padding_m: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.query_margin_us < 0:
            raise ValueError("query_margin_us must be non-negative")
        if self.minimum_rig_distance_m < 0.0 or self.minimum_travel_distance_m < 0.0:
            raise ValueError("track distance thresholds must be non-negative")
        if len(self.cuboid_padding_m) != 3 or any(x < 0.0 for x in self.cuboid_padding_m):
            raise ValueError("cuboid_padding_m must contain three non-negative values")


def build_dynamic_actor_association(
    scene: DynamicGaussianScene,
    options: DynamicAssociateOptions,
) -> DynamicActorAssociation:
    """Recover the actor IDs discarded by the v1 InstantNuRec dynamic export.

    This intentionally imports InstantNuRec only in the offline builder.  The
    resulting sidecar is self-contained and rendering does not depend on the
    ignored upstream clone.
    """
    try:
        import torch
        from ncore.impl.common import transformations as ncore_transformations
        from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
        from instant_nurec.datasets.tracks import CuboidTracks
        from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
        from instant_nurec.utils.types import HalfClosedInterval, TrackFlags
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "actor association building requires the InstantNuRec package and nvidia-ncore"
        ) from exc

    if scene.count == 0:
        raise ValueError("cannot associate an empty dynamic Gaussian asset")
    source_slots, slot_timestamps = infer_dynamic_source_slots(scene)
    source_middle = scene.keyframe_timestamps_us[:, 1]
    query_start = int(source_middle.min()) - options.query_margin_us
    query_end = int(source_middle.max()) + options.query_margin_us + 1
    loader = _create_ncore_loader(options.ncore_path)
    cuboid_df = compute_cuboid_df(loader, HalfClosedInterval(query_start, query_end))
    tracks = consolidate_cuboid_tracks(
        cuboid_df,
        loader,
        ["AUTOLABEL"],
        options.minimum_rig_distance_m,
        np.eye(4, dtype=np.float64),
    )

    track_ids: list[str] = []
    track_poses: list[np.ndarray] = []
    track_timestamps: list[np.ndarray] = []
    track_dims: list[np.ndarray] = []
    track_details: list[dict[str, object]] = []
    source_span_us = max(int(source_middle.max() - source_middle.min()), 1)
    for track_id, track in tracks.items():
        poses = np.stack(track["poses"], axis=0)
        timestamps = np.asarray(track["timestamps_us"], dtype=np.int64)
        if len(timestamps) <= 1:
            continue
        first = (poses[0] @ ncore_transformations.se3_inverse(poses[1])) @ poses[0]
        last = (poses[-1] @ ncore_transformations.se3_inverse(poses[-2])) @ poses[-1]
        poses = np.concatenate((first[None], poses, last[None]), axis=0).astype(np.float32)
        timestamps = np.concatenate(
            (
                [timestamps[0] - (timestamps[1] - timestamps[0])],
                timestamps,
                [timestamps[-1] + (timestamps[-1] - timestamps[-2])],
            )
        ).astype(np.int64)
        travel_distance_m = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
        track_span_us = max(int(timestamps[-1] - timestamps[0]), 1)
        scaled_travel_distance_m = travel_distance_m * source_span_us / track_span_us
        label_class = str(track["label_class"])
        is_dynamic = (
            label_class in NCoreInstantNuRecDataset.UNCONDITIONALLY_DYNAMIC_LABELS
            or scaled_travel_distance_m > options.minimum_travel_distance_m
        )
        if not is_dynamic:
            continue
        dimension = np.asarray(track["dimension"], dtype=np.float32)
        track_ids.append(str(track_id))
        track_poses.append(poses)
        track_timestamps.append(timestamps)
        track_dims.append(dimension)
        track_details.append(
            {
                "track_id": str(track_id),
                "class_id": label_class,
                "dimension_m": dimension.astype(float).tolist(),
                "observation_count": int(len(timestamps) - 2),
                "scaled_travel_distance_m": scaled_travel_distance_m,
            }
        )
    if not track_ids:
        raise ValueError("no dynamic NCore cuboid tracks overlap the dynamic Gaussian asset")

    cuboid_tracks = CuboidTracks.Factory.from_numpy(
        track_ids,
        track_poses,
        track_timestamps,
        [TrackFlags.DYNAMIC] * len(track_ids),
        track_dims,
        device=torch.device("cpu"),
    )
    points = torch.from_numpy(scene.keyframe_positions[:, 1].astype(np.float32))
    point_times = torch.from_numpy(source_middle.astype(np.int64))
    padding = torch.tensor(options.cuboid_padding_m, dtype=torch.float32)
    _, track_indices_torch = cuboid_tracks.point_intersection_interpolate_pose(
        points,
        point_times,
        padding,
    )
    track_indices = track_indices_torch.cpu().numpy().astype(np.int32)
    counts = np.bincount(track_indices[track_indices >= 0], minlength=len(track_ids))
    for track_index, (detail, count) in enumerate(zip(track_details, counts, strict=True)):
        detail["associated_gaussian_count"] = int(count)
        detail["source_slots"] = [
            int(value)
            for value in np.unique(source_slots[track_indices == track_index])
        ]
    return DynamicActorAssociation(
        track_indices=track_indices,
        source_slot_indices=source_slots.astype(np.int16),
        slot_timestamps_us=slot_timestamps,
        dynamic_sha256=file_sha256(options.dynamic_path),
        gaussian_count=scene.count,
        tracks=tuple(track_details),
        metadata={
            "ncore_path": str(options.ncore_path.resolve()),
            "query_range_us": [query_start, query_end],
            "query_margin_us": options.query_margin_us,
            "minimum_rig_distance_m": options.minimum_rig_distance_m,
            "minimum_travel_distance_m": options.minimum_travel_distance_m,
            "cuboid_padding_m": list(options.cuboid_padding_m),
            "association_method": "middle-keyframe point in interpolated padded dynamic cuboid",
            "associated_gaussian_count": int(np.count_nonzero(track_indices >= 0)),
        },
    )
