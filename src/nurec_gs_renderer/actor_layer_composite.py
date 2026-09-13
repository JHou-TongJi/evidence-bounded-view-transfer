"""Compose separately rendered frozen-static and canonical-actor layers.

The direct FTheta actor optimizer trains an explicit actor-over-static
visibility model so frozen dynamic leakage cannot participate in its backward
graph.  This utility applies the identical premultiplied-alpha composition to
two exact-camera render directories, preserving reproducible manifests for
the existing actor ROI audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def layer_actor_over_static(
    static_rgb: np.ndarray,
    static_alpha: np.ndarray,
    actor_rgb_with_sky: np.ndarray,
    actor_alpha: np.ndarray,
    actor_sky_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``C_actor + (1-alpha_actor) C_static`` in premultiplied form."""
    static_rgb = np.asarray(static_rgb, dtype=np.float32)
    actor_rgb = np.asarray(actor_rgb_with_sky, dtype=np.float32)
    actor_sky = np.asarray(actor_sky_rgb, dtype=np.float32)
    static_alpha = np.asarray(static_alpha, dtype=np.float32)
    actor_alpha = np.asarray(actor_alpha, dtype=np.float32)
    if static_rgb.shape != actor_rgb.shape or static_rgb.shape != actor_sky.shape or static_rgb.ndim != 3 or static_rgb.shape[-1] != 3:
        raise ValueError("static RGB, actor RGB and actor sky must be matching HxWx3 arrays")
    if static_alpha.shape != static_rgb.shape[:2] or actor_alpha.shape != static_alpha.shape:
        raise ValueError("alpha dimensions must match RGB")
    actor_premultiplied = actor_rgb - (1.0 - actor_alpha[..., None]) * actor_sky
    rgb = np.clip(actor_premultiplied + (1.0 - actor_alpha[..., None]) * static_rgb, 0.0, 1.0)
    alpha = np.clip(actor_alpha + (1.0 - actor_alpha) * static_alpha, 0.0, 1.0)
    return rgb.astype(np.float32), alpha.astype(np.float32)


def layer_actor_over_static_depth_ordered(
    static_rgb: np.ndarray,
    static_alpha: np.ndarray,
    static_depth: np.ndarray,
    actor_rgb_with_sky: np.ndarray,
    actor_alpha: np.ndarray,
    actor_depth: np.ndarray,
    actor_sky_rgb: np.ndarray,
    *,
    alpha_threshold: float = .02,
    absolute_depth_margin_m: float = .0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Composite actor only where its expected depth is in front of static.

    The returned boolean is deliberately exposed to diagnostics: a static PLY
    leak may still be closer than a sparse actor surface, and treating an
    invalid/behind actor alpha as foreground would reintroduce ghosting.
    """
    static_depth = np.asarray(static_depth, np.float32)
    actor_depth = np.asarray(actor_depth, np.float32)
    if static_depth.shape != actor_depth.shape or static_depth.shape != np.asarray(static_alpha).shape:
        raise ValueError("static/actor depth and alpha must share HxW dimensions")
    if alpha_threshold < 0 or absolute_depth_margin_m < 0:
        raise ValueError("alpha threshold and depth margin must be non-negative")
    actor_present = np.asarray(actor_alpha, np.float32) >= alpha_threshold
    actor_valid = np.isfinite(actor_depth)
    static_valid = np.isfinite(static_depth)
    actor_front = actor_present & actor_valid & (~static_valid | (actor_depth <= static_depth + absolute_depth_margin_m))
    original_alpha = np.asarray(actor_alpha, np.float32)
    gated_alpha = np.where(actor_front, original_alpha, 0.)
    # ``actor_rgb_with_sky`` is already composed with *original_alpha*.
    # Gating only alpha would leave its premultiplied colour behind in a
    # rejected pixel.  Recover that colour first, then remove it together with
    # alpha where the static layer is in front.
    actor_premultiplied = np.asarray(actor_rgb_with_sky, np.float32) - (1. - original_alpha[..., None]) * np.asarray(actor_sky_rgb, np.float32)
    actor_premultiplied = np.where(actor_front[..., None], actor_premultiplied, 0.)
    rgb = np.clip(actor_premultiplied + (1. - gated_alpha[..., None]) * np.asarray(static_rgb, np.float32), 0., 1.)
    alpha = np.clip(gated_alpha + (1. - gated_alpha) * np.asarray(static_alpha, np.float32), 0., 1.)
    depth = np.where(actor_front, actor_depth, static_depth).astype(np.float32)
    return rgb, alpha, depth, actor_front


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _manifest(path: Path) -> dict[str, object]:
    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("status") != "complete":
        raise ValueError(f"input manifest is not complete: {path}")
    return value


def compose_render_directories(
    static_root: Path, actor_root: Path, output: Path, *, depth_order: bool = False,
    depth_alpha_threshold: float = .02, absolute_depth_margin_m: float = .0,
) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty layer-composite directory: {output}")
    static_manifest, actor_manifest = _manifest(static_root), _manifest(actor_root)
    static_frames = {str(frame["frame_id"]): frame for frame in static_manifest["frames"] if frame.get("complete")}
    actor_frames = {str(frame["frame_id"]): frame for frame in actor_manifest["frames"] if frame.get("complete")}
    if set(static_frames) != set(actor_frames):
        raise ValueError("static and actor render directories must contain the same complete frame IDs")
    output.mkdir(parents=True)
    for name in ("rgb", "alpha", "depth", "sky_rgb", "sky_valid", "actor_front"):
        (output / name).mkdir()
    records: list[dict[str, object]] = []
    for frame in static_manifest["frames"]:
        frame_id = str(frame["frame_id"])
        if frame_id not in actor_frames:
            continue
        static_frame, actor_frame = static_frames[frame_id], actor_frames[frame_id]
        static_out, actor_out = static_frame["outputs"], actor_frame["outputs"]
        required_static = {"rgb", "alpha"}
        required_actor = {"rgb", "alpha"}
        if not required_static.issubset(static_out) or not required_actor.issubset(actor_out):
            raise ValueError("static and actor-only renders both need RGB and alpha")
        actor_rgb = _read_rgb(actor_root / actor_out["rgb"])
        # A canonical actor-only render may legitimately have no sky asset.
        # In that case its RGB already has a black background, so black is the
        # exact background term to subtract from its premultiplied color.
        actor_sky = (
            _read_rgb(actor_root / actor_out["sky_rgb"])
            if "sky_rgb" in actor_out
            else np.zeros_like(actor_rgb)
        )
        static_rgb, static_alpha = _read_rgb(static_root / static_out["rgb"]), _read_alpha(static_root / static_out["alpha"])
        actor_alpha = _read_alpha(actor_root / actor_out["alpha"])
        static_depth = static_out.get("depth_npy")
        actor_depth = actor_out.get("depth_npy")
        if depth_order:
            if static_depth is None or actor_depth is None:
                raise ValueError("depth-ordered composition requires depth_npy from static and actor renders")
            rgb, alpha, depth, actor_front = layer_actor_over_static_depth_ordered(
                static_rgb, static_alpha, np.load(static_root / static_depth), actor_rgb, actor_alpha,
                np.load(actor_root / actor_depth), actor_sky, alpha_threshold=depth_alpha_threshold,
                absolute_depth_margin_m=absolute_depth_margin_m,
            )
        else:
            rgb, alpha = layer_actor_over_static(static_rgb, static_alpha, actor_rgb, actor_alpha, actor_sky)
            depth, actor_front = None, None
        stem = Path(str(static_out["rgb"])).stem
        Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB").save(output / "rgb" / f"{stem}.png")
        Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8), mode="L").save(output / "alpha" / f"{stem}.png")
        output_paths: dict[str, str] = {"rgb": f"rgb/{stem}.png", "alpha": f"alpha/{stem}.png"}
        if "sky_rgb" in actor_out and "sky_valid" in actor_out:
            Image.open(actor_root / actor_out["sky_rgb"]).save(output / "sky_rgb" / f"{stem}.png")
            Image.open(actor_root / actor_out["sky_valid"]).save(output / "sky_valid" / f"{stem}.png")
            output_paths.update(sky_rgb=f"sky_rgb/{stem}.png", sky_valid=f"sky_valid/{stem}.png")
        if depth is not None:
            np.save(output / "depth" / f"{stem}.npy", depth)
            output_paths["depth_npy"] = f"depth/{stem}.npy"
            Image.fromarray(actor_front.astype(np.uint8) * 255, mode="L").save(output / "actor_front" / f"{stem}.png")
            output_paths["actor_front"] = f"actor_front/{stem}.png"
        elif static_depth is not None and actor_depth is not None:
            s_depth, a_depth = np.load(static_root / static_depth), np.load(actor_root / actor_depth)
            depth = np.where(actor_alpha >= .02, a_depth, s_depth).astype(np.float32)
            np.save(output / "depth" / f"{stem}.npy", depth)
            output_paths["depth_npy"] = f"depth/{stem}.npy"
        record = dict(static_frame)
        record["outputs"] = output_paths
        record["layer_composite"] = {"mode": "actor_over_static_depth_ordered" if depth_order else "actor_over_static",
                                     "actor_alpha_threshold_for_depth": depth_alpha_threshold if depth_order else .02,
                                     "absolute_depth_margin_m": absolute_depth_margin_m if depth_order else None}
        records.append(record)
    manifest = dict(static_manifest)
    manifest.update({
        "status": "complete", "frames": records, "frame_count": len(records), "completed_frame_count": len(records),
        "run_metadata": {"layer_composite": {"mode": "actor_over_static_depth_ordered" if depth_order else "actor_over_static", "static_render_dir": str(static_root.resolve()), "actor_render_dir": str(actor_root.resolve()), "absolute_depth_margin_m": absolute_depth_margin_m if depth_order else None}},
        "rendered_this_run": len(records), "resumed_frames": 0,
    })
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compose exact static and actor-only render directories using actor-over-static alpha")
    parser.add_argument("--static-render-dir", type=Path, required=True)
    parser.add_argument("--actor-render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-order", action="store_true", help="Apply actor only where expected actor depth is in front of static depth")
    parser.add_argument("--depth-alpha-threshold", type=float, default=.02)
    parser.add_argument("--absolute-depth-margin-m", type=float, default=.0)
    args = parser.parse_args(argv)
    result = compose_render_directories(args.static_render_dir, args.actor_render_dir, args.output_dir, depth_order=args.depth_order, depth_alpha_threshold=args.depth_alpha_threshold, absolute_depth_margin_m=args.absolute_depth_margin_m)
    print(f"Wrote actor-over-static render manifest: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
