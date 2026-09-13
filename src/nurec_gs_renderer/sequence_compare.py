from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .render import _write_json_atomic


def _load_manifest(directory: Path) -> dict[str, object]:
    path = directory / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("status") != "complete":
        raise ValueError(f"sequence manifest is not complete: {path}")
    return value


def _output_path(directory: Path, record: dict[str, object], key: str) -> Path | None:
    outputs = record.get("outputs", {})
    value = outputs.get(key) if isinstance(outputs, dict) else None
    if value is None:
        return None
    path = directory / str(value)
    if not path.is_file():
        raise FileNotFoundError(f"manifest output is missing: {path}")
    return path


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def compare_sequence_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, object]:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    baseline = _load_manifest(baseline_dir)
    candidate = _load_manifest(candidate_dir)
    baseline_frames = baseline.get("frames", [])
    candidate_frames = candidate.get("frames", [])
    if len(baseline_frames) != len(candidate_frames):
        raise ValueError("sequence frame counts do not match")

    records: list[dict[str, object]] = []
    previous_delta: np.ndarray | None = None
    for index, (left, right) in enumerate(zip(baseline_frames, candidate_frames, strict=True)):
        identity = (left.get("frame_id"), left.get("timestamp_start_us"), left.get("timestamp_us"))
        if identity != (right.get("frame_id"), right.get("timestamp_start_us"), right.get("timestamp_us")):
            raise ValueError(f"sequence frame identity mismatch at index {index}")
        record: dict[str, object] = {
            "index": index,
            "frame_id": left.get("frame_id"),
            "timestamp_start_us": left.get("timestamp_start_us"),
            "timestamp_us": left.get("timestamp_us"),
        }
        left_rgb_path = _output_path(baseline_dir, left, "rgb")
        right_rgb_path = _output_path(candidate_dir, right, "rgb")
        if left_rgb_path is not None and right_rgb_path is not None:
            left_rgb = _read_rgb(left_rgb_path)
            right_rgb = _read_rgb(right_rgb_path)
            if left_rgb.shape != right_rgb.shape:
                raise ValueError(f"RGB shape mismatch at index {index}")
            delta = right_rgb - left_rgb
            absolute = np.abs(delta)
            record["rgb_mean_absolute_change"] = float(absolute.mean())
            record["rgb_p95_absolute_change"] = float(np.percentile(absolute, 95.0))
            record["rgb_changed_pixel_fraction"] = float(np.any(absolute > (0.5 / 255.0), axis=-1).mean())
            record["correction_delta_temporal_change"] = None if previous_delta is None else float(
                np.abs(delta - previous_delta).mean()
            )
            previous_delta = delta

        left_alpha = _output_path(baseline_dir, left, "alpha")
        right_alpha = _output_path(candidate_dir, right, "alpha")
        right_alpha_value: np.ndarray | None = None
        if left_alpha is not None and right_alpha is not None:
            record["alpha_byte_identical"] = left_alpha.read_bytes() == right_alpha.read_bytes()
            right_alpha_value = (
                np.asarray(Image.open(right_alpha).convert("L"), dtype=np.float32) / 255.0
            )

        left_depth = _output_path(baseline_dir, left, "depth_npy")
        right_depth = _output_path(candidate_dir, right, "depth_npy")
        if left_depth is not None and right_depth is not None:
            left_value = np.load(left_depth, allow_pickle=False)
            right_value = np.load(right_depth, allow_pickle=False)
            identical = np.array_equal(left_value, right_value, equal_nan=True)
            record["depth_identical"] = bool(identical)
            record["depth_max_absolute_change_m"] = 0.0 if identical else float(
                np.nanmax(np.abs(left_value - right_value))
            )

        sky_valid = _output_path(candidate_dir, right, "sky_valid")
        if sky_valid is not None:
            valid = np.asarray(Image.open(sky_valid).convert("L"), dtype=np.float32) / 255.0
            upper = valid[: len(valid) // 2]
            invalid = upper < 0.5
            record["upper_half_sky_invalid_fraction"] = float(invalid.mean())
            if right_alpha_value is not None:
                low_alpha = right_alpha_value[: len(valid) // 2] < 0.05
                record["upper_half_low_alpha_fallback_fraction"] = float(
                    (invalid & low_alpha).mean()
                )
                record["low_alpha_fallback_fraction_within_low_alpha"] = float(
                    (invalid & low_alpha).sum() / max(int(low_alpha.sum()), 1)
                )
        records.append(record)

    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records if record.get(key) is not None]

    rgb_changes = values("rgb_mean_absolute_change")
    temporal_changes = values("correction_delta_temporal_change")
    invalid_sky = values("upper_half_sky_invalid_fraction")
    low_alpha_fallback = values("upper_half_low_alpha_fallback_fraction")
    low_alpha_conditional = values("low_alpha_fallback_fraction_within_low_alpha")
    return {
        "schema_version": 1,
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "frame_count": len(records),
        "all_alpha_byte_identical": all(
            bool(record.get("alpha_byte_identical", True)) for record in records
        ),
        "all_depth_identical": all(bool(record.get("depth_identical", True)) for record in records),
        "rgb_mean_absolute_change": None if not rgb_changes else float(np.mean(rgb_changes)),
        "rgb_p95_frame_mean_absolute_change": None
        if not rgb_changes
        else float(np.percentile(rgb_changes, 95.0)),
        "correction_delta_temporal_change_mean": None
        if not temporal_changes
        else float(np.mean(temporal_changes)),
        "correction_delta_temporal_change_p95": None
        if not temporal_changes
        else float(np.percentile(temporal_changes, 95.0)),
        "upper_half_sky_invalid_fraction_mean": None
        if not invalid_sky
        else float(np.mean(invalid_sky)),
        "upper_half_sky_invalid_fraction_p95": None
        if not invalid_sky
        else float(np.percentile(invalid_sky, 95.0)),
        "upper_half_low_alpha_fallback_fraction_mean": None
        if not low_alpha_fallback
        else float(np.mean(low_alpha_fallback)),
        "upper_half_low_alpha_fallback_fraction_p95": None
        if not low_alpha_fallback
        else float(np.percentile(low_alpha_fallback, 95.0)),
        "low_alpha_fallback_fraction_within_low_alpha_mean": None
        if not low_alpha_conditional
        else float(np.mean(low_alpha_conditional)),
        "low_alpha_fallback_fraction_within_low_alpha_p95": None
        if not low_alpha_conditional
        else float(np.percentile(low_alpha_conditional, 95.0)),
        "frames": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nurec-gs-sequence-compare",
        description="Compare two completed renderer sequence directories frame by frame.",
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_sequence_runs(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report, args.output)
    print(f"Compared {report['frame_count']} frames; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
