from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pose_sources import _create_ncore_loader
from .render import _write_json_atomic


def _load_manifest(directory: Path) -> dict[str, object]:
    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete sequence: {directory}")
    return manifest


def _image(directory: Path, record: dict[str, object], key: str, mode: str) -> np.ndarray:
    relative = record["outputs"][key]
    return np.asarray(Image.open(directory / relative).convert(mode), dtype=np.float32) / 255.0


def _source_image(sensor, frame_index: int, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(sensor.get_frame_image_array(frame_index), dtype=np.uint8))
    if image.size != size:
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def evaluate(
    *,
    ncore_path: Path,
    camera_id: str,
    static_dir: Path,
    dynamic_dir: Path,
) -> dict[str, object]:
    static_manifest = _load_manifest(static_dir)
    dynamic_manifest = _load_manifest(dynamic_dir)
    static_frames = static_manifest["frames"]
    dynamic_frames = dynamic_manifest["frames"]
    if len(static_frames) != len(dynamic_frames):
        raise ValueError("static and dynamic sequence lengths differ")

    loader = _create_ncore_loader(ncore_path)
    sensor = loader.get_camera_sensor(camera_id)
    records: list[dict[str, object]] = []
    previous_source: np.ndarray | None = None
    previous_static: np.ndarray | None = None
    previous_dynamic: np.ndarray | None = None

    for index, (static_record, dynamic_record) in enumerate(
        zip(static_frames, dynamic_frames, strict=True)
    ):
        identity = (
            static_record["frame_id"],
            static_record["timestamp_start_us"],
            static_record["timestamp_us"],
        )
        if identity != (
            dynamic_record["frame_id"],
            dynamic_record["timestamp_start_us"],
            dynamic_record["timestamp_us"],
        ):
            raise ValueError(f"frame identity mismatch at {index}")
        frame_index = int(str(static_record["frame_id"]).split("_", 1)[0])
        static_rgb = _image(static_dir, static_record, "rgb", "RGB")
        dynamic_rgb = _image(dynamic_dir, dynamic_record, "rgb", "RGB")
        static_alpha = _image(static_dir, static_record, "alpha", "L")
        source = _source_image(
            sensor,
            frame_index,
            (static_rgb.shape[1], static_rgb.shape[0]),
        )

        static_error = np.mean(np.abs(static_rgb - source), axis=-1)
        dynamic_error = np.mean(np.abs(dynamic_rgb - source), axis=-1)
        change = np.mean(np.abs(dynamic_rgb - static_rgb), axis=-1)
        changed = change > (1.0 / 255.0)
        core = static_alpha >= 0.9
        semi = (static_alpha > 0.05) & (static_alpha < 0.9)
        background = static_alpha <= 0.05

        def masked_mean(value: np.ndarray, mask: np.ndarray) -> float | None:
            return None if not mask.any() else float(value[mask].mean())

        record: dict[str, object] = {
            "index": index,
            "frame_index": frame_index,
            "frame_id": static_record["frame_id"],
            "timestamp_us": static_record["timestamp_us"],
            "active_dynamic_gaussians": int(dynamic_record.get("active_dynamic_gaussians", 0)),
            "static_mae": float(static_error.mean()),
            "dynamic_mae": float(dynamic_error.mean()),
            "core_static_mae": masked_mean(static_error, core),
            "core_dynamic_mae": masked_mean(dynamic_error, core),
            "semi_static_mae": masked_mean(static_error, semi),
            "semi_dynamic_mae": masked_mean(dynamic_error, semi),
            "background_static_mae": masked_mean(static_error, background),
            "background_dynamic_mae": masked_mean(dynamic_error, background),
            "changed_pixel_fraction": float(changed.mean()),
            "changed_static_mae": masked_mean(static_error, changed),
            "changed_dynamic_mae": masked_mean(dynamic_error, changed),
            "changed_improved_fraction": masked_mean(
                (dynamic_error < static_error).astype(np.float32), changed
            ),
            "rgb_mean_absolute_change": float(np.abs(dynamic_rgb - static_rgb).mean()),
        }
        if previous_source is not None:
            source_delta = source - previous_source
            static_delta = static_rgb - previous_static
            dynamic_delta = dynamic_rgb - previous_dynamic
            record["static_temporal_delta_residual"] = float(
                np.abs(static_delta - source_delta).mean()
            )
            record["dynamic_temporal_delta_residual"] = float(
                np.abs(dynamic_delta - source_delta).mean()
            )
        else:
            record["static_temporal_delta_residual"] = None
            record["dynamic_temporal_delta_residual"] = None
        records.append(record)
        previous_source = source
        previous_static = static_rgb
        previous_dynamic = dynamic_rgb

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [record[key] for record in records if record.get(key) is not None],
            dtype=np.float64,
        )

    def comparison(static_key: str, dynamic_key: str) -> dict[str, float]:
        static = values(static_key)
        dynamic = values(dynamic_key)
        return {
            "static_mean": float(static.mean()),
            "dynamic_mean": float(dynamic.mean()),
            "relative_improvement_percent": float(
                (static.mean() - dynamic.mean()) / max(static.mean(), 1e-12) * 100.0
            ),
            "improved_frame_fraction": float((dynamic < static).mean()),
        }

    improvements = values("static_mae") - values("dynamic_mae")
    best = np.argsort(improvements)[-10:][::-1]
    worst = np.argsort(improvements)[:10]
    return {
        "schema_version": 1,
        "ncore_path": str(ncore_path),
        "camera_id": camera_id,
        "static_dir": str(static_dir),
        "dynamic_dir": str(dynamic_dir),
        "frame_count": len(records),
        "full_frame": comparison("static_mae", "dynamic_mae"),
        "core": comparison("core_static_mae", "core_dynamic_mae"),
        "semi": comparison("semi_static_mae", "semi_dynamic_mae"),
        "background": comparison("background_static_mae", "background_dynamic_mae"),
        "changed_region": comparison("changed_static_mae", "changed_dynamic_mae"),
        "temporal_delta": comparison(
            "static_temporal_delta_residual", "dynamic_temporal_delta_residual"
        ),
        "changed_pixel_fraction_mean": float(values("changed_pixel_fraction").mean()),
        "changed_improved_fraction_mean": float(values("changed_improved_fraction").mean()),
        "rgb_mean_absolute_change": float(values("rgb_mean_absolute_change").mean()),
        "active_dynamic_gaussians": {
            "mean": float(values("active_dynamic_gaussians").mean()),
            "p95": float(np.percentile(values("active_dynamic_gaussians"), 95)),
            "max": int(values("active_dynamic_gaussians").max()),
        },
        "best_frames": [records[int(i)] for i in best],
        "worst_frames": [records[int(i)] for i in worst],
        "frames": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate dynamic Gaussian sequence against NCore source frames")
    parser.add_argument("--ncore-path", required=True, type=Path)
    parser.add_argument("--camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--static-dir", required=True, type=Path)
    parser.add_argument("--dynamic-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate(
        ncore_path=args.ncore_path,
        camera_id=args.camera_id,
        static_dir=args.static_dir,
        dynamic_dir=args.dynamic_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
