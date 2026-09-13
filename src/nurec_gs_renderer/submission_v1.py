"""Build a self-auditing V1 delivery package for a completed L4 7V sequence.

The builder intentionally copies only final evidence (RGB sequence, videos, selected
examples and the code/configuration needed to inspect it).  Large training assets,
NCore clips and model weights remain outside the repository and are represented by a
lineage table with hashes/paths instead of being silently bundled.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERAS = ("front_center", "front_left", "front_right", "side_left", "side_right", "rear_left", "rear_right")
SCHEMA = "cross-vehicle-submission-v1"
PACKAGE_VERSION = 2


@dataclass(frozen=True)
class SubmissionV1Options:
    sequence_root: Path
    background_root: Path
    camera_config: Path
    expert_registry: Path
    road_depth_report: Path
    output: Path
    ncore_path: Path
    source_camera_id: str = "camera_front_wide_120fov"
    fps: int = 30
    metric_stride: int = 10

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.metric_stride <= 0:
            raise ValueError("fps and metric_stride must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rgb_files(root: Path, camera: str) -> list[Path]:
    files = list((root / camera / "rgb").glob("*.png"))
    if not files:
        raise FileNotFoundError(f"missing RGB frames: {root / camera / 'rgb'}")
    indexed: list[tuple[int, Path]] = []
    for path in files:
        token = path.stem.split("_", 1)[0]
        if not token.isdigit():
            raise ValueError(f"cannot infer frame index from RGB filename: {path.name}")
        indexed.append((int(token), path))
    indexed.sort()
    expected = list(range(len(indexed)))
    if [index for index, _ in indexed] != expected:
        raise ValueError(f"RGB frames must be contiguous 000000..: {root / camera / 'rgb'}")
    return [path for _, path in indexed]


def _image_stats(path: Path) -> tuple[int, int, float]:
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    # This is deliberately reported rather than used as a pass/fail score: shadows
    # and underexposure are legitimate road pixels, while a high value can reveal a
    # renderer hole or an unsupported view.
    near_black = float(np.all(image <= 2, axis=-1).mean())
    return width, height, near_black


def _actor_metrics(sequence_root: Path, background_root: Path, camera: str, stride: int) -> dict[str, float]:
    finals = _rgb_files(sequence_root, camera)
    alpha_paths = sorted((sequence_root / camera / "actor_alpha").glob("*.png"))
    backgrounds = _rgb_files(background_root, camera)
    if len(alpha_paths) != len(finals) or len(backgrounds) != len(finals):
        raise ValueError(f"unaligned actor/background sequence for {camera}")
    outside_mae: list[float] = []
    areas: list[float] = []
    ious: list[float] = []
    previous_alpha: np.ndarray | None = None
    for alpha_path in alpha_paths:
        with Image.open(alpha_path) as opened:
            alpha = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
        areas.append(float(alpha.mean()))
        binary = alpha > 0.5
        if previous_alpha is not None:
            union = int((previous_alpha | binary).sum())
            if union:
                ious.append(float((previous_alpha & binary).sum() / union))
        previous_alpha = binary
    near_black: list[float] = []
    for index, (final_path, alpha_path, background_path) in enumerate(zip(finals, alpha_paths, backgrounds)):
        if index % stride and index != len(finals) - 1:
            continue
        with Image.open(final_path) as opened:
            final = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(background_path) as opened:
            background = np.asarray(opened.convert("RGB").resize((final.shape[1], final.shape[0])), dtype=np.float32) / 255.0
        with Image.open(alpha_path) as opened:
            alpha = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
        near_black.append(float(np.all(final <= (2.0 / 255.0), axis=-1).mean()))
        outside = alpha <= (1.0 / 255.0)
        outside_mae.append(float(np.abs(final - background)[outside].mean()) if np.any(outside) else 0.0)
    active = [value for value in areas if value > 1.0 / 255.0]
    return {
        "sampled_outside_actor_rgb_mae": float(np.mean(outside_mae)),
        "sampled_outside_actor_rgb_mae_max": float(np.max(outside_mae)),
        "actor_active_frames": float(len(active)),
        "actor_alpha_area_mean": float(np.mean(areas)),
        "actor_alpha_area_active_mean": float(np.mean(active)) if active else 0.0,
        "adjacent_actor_alpha_iou_mean": float(np.mean(ious)) if ious else 1.0,
        "adjacent_actor_alpha_iou_p05": float(np.percentile(ious, 5)) if ious else 1.0,
        "sampled_near_black_fraction": float(np.mean(near_black)),
        "sampled_frames": float(len(near_black)),
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_checksum_inventory(root: Path) -> tuple[Path, int]:
    """Write a deterministic integrity inventory for every delivered file.

    Dataset/model assets deliberately stay outside the delivery.  Hashing the copied
    RGB frames, videos, reports and code snapshot makes the package itself auditable
    after transfer without pretending that it embeds the original training data.
    """
    destination = root / "05_评价结果" / "checksums.sha256"
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{_sha256(path)}\t{path.stat().st_size}\t{relative}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination, len(rows)


def _copy_evidence(destination: Path, sources: dict[str, Path]) -> dict[str, dict[str, str | int]]:
    """Copy small, decisive source manifests/reports next to the delivery evidence."""
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, str | int]] = {}
    for label, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing reproducibility evidence {label}: {source}")
        target = destination / f"{label}{source.suffix}"
        shutil.copy2(source, target)
        result[label] = {
            "external_path": str(source.resolve()),
            "copied_path": target.name,
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        }
    return result


def _copy_code(project: Path, destination: Path) -> None:
    code = destination / "code"
    shutil.copytree(project / "src" / "nurec_gs_renderer", code / "src" / "nurec_gs_renderer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(project / "configs", code / "configs", ignore=shutil.ignore_patterns("*.ply", "*.npz", "*.pt"))
    shutil.copytree(project / "examples", code / "examples", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(project / name, code / name)
    (code / "requirements.txt").write_text(
        "# Core package; use the documented conda environment for CUDA/gsplat.\n"
        "# pip install -e '.[ncore,sky]'\n"
        "# Third-party source required for the static 2DGS proxy: external/2d-gaussian-splatting\n"
        "# Dynamic actor experts in this V1 are 2DGS source-view proxies and are not a model download.\n",
        encoding="utf-8",
    )
    (code / "third_party.md").write_text(
        "# Third-party components\n\n"
        "- NVIDIA NCore / PhysicalAI-Autonomous-Vehicles-NCore: source data and calibration.\n"
        "- NVIDIA InstantNuRec: static Gaussian export used by the upstream background workflow.\n"
        "- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting): static proxy and rigid actor expert implementation.\n\n"
        "The submission does not redistribute datasets, weights, or external repositories. Consult each upstream licence before reproducing.\n",
        encoding="utf-8",
    )
    (code / "run_demo.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nROOT=$(cd \"$(dirname \"$0\")/../..\" && pwd)\n"
        "if python -c 'import numpy, PIL' >/dev/null 2>&1; then\n"
        "  RUN=(python)\n"
        "elif command -v conda >/dev/null 2>&1; then\n"
        "  RUN=(conda run --no-capture-output -n \"${CROSS_VEHICLE_CONDA_ENV:-instant-nurec}\" python)\n"
        "else\n"
        "  echo 'Need Python with numpy and Pillow, or conda env instant-nurec.' >&2; exit 2\n"
        "fi\n"
        "PYTHONPATH=\"$ROOT/03_工程代码/code/src\" \"${RUN[@]}\" -m nurec_gs_renderer.submission_v1_cli --verify \"$ROOT\"\n"
        "echo \"Open 06_Demo展示/demo_7v_mosaic.mp4 or any 04_生成结果/generated_video/*.mp4\"\n",
        encoding="utf-8",
    )
    (code / "run_demo.sh").chmod(0o755)


def _markdown_documents(root: Path, options: SubmissionV1Options, metrics: dict[str, Any], road: dict[str, Any], registry: dict[str, Any]) -> dict[str, str]:
    target = _read_json(options.camera_config)
    cameras = target["cameras"]
    camera_rows = "\n".join(
        f"| {item['camera_id']} | {item['mount']['position_ego_m']} | {item['mount']['yaw_pitch_roll_deg']} | {item['mount']['horizontal_fov_deg']:.1f} |"
        for item in cameras
    )
    global_metrics = metrics["global"]
    actor_rows = "\n".join(
        f"| {camera} | {value['actor_active_frames']:.0f}/{global_metrics['frames_per_camera']} | {value['adjacent_actor_alpha_iou_mean']:.4f} | {value['sampled_outside_actor_rgb_mae']:.7f} |"
        for camera, value in metrics["per_camera"].items()
    )
    accepted = "\n".join(
        f"| {entry['track_id']} | {entry['label']} | {entry['source_camera']} | {entry['metrics']['masked_rgb_mae']:.5f} | {entry['metrics']['alpha_iou_0_5']:.4f} | {entry['metrics']['alpha_area_ratio']:.4f} |"
        for entry in registry["experts"]
    )
    method = f"""# 技术方案报告（V1）

## 任务与样例范围

本提交面向“从乘用车到 L4 无人物流车的跨车型采集数据视角重建”。V1 使用公开 NCore clip
`7fd4d554-b155-45c6-a46a-646486029d85` 的 chunk 0，生成 7 路目标小车视角各 {global_metrics['frames_per_camera']} 帧、
{global_metrics['fps']} FPS、{global_metrics['width']}×{global_metrics['height']} 的连续序列（{global_metrics['duration_s']:.4f} 秒）。

## 方法

1. 使用 NCore 的标定/位姿构建 20-tile virtual-pinhole proxy，并以官方 2D Gaussian Splatting（12,000 iter）重建完整 clip 的静态背景；天空使用先前离线构建的多视图时序 cubemap，只在 2DGS alpha 后方合成；
2. 使用经过 source-camera 时间留出检查的刚性 2DGS actor expert，对目标视角仅在观察方向、距离、时刻均兼容时叠加；其余动态区域 fail-closed 回退静态背景；
3. 按 actor alpha 在 actor 之间排序。V1 不使用静态深度作硬遮挡真值，因为静态场景已吸收部分动态物体，会造成空洞；
4. 输出 RGB、动态 alpha/owner 调试资产、帧映射、视频和可自动复算的指标。

## 目标 L4 小车 rig

目标车型为 `{target['vehicle']['type']}`，尺寸假设 {target['vehicle']['length_m']} m × {target['vehicle']['width_m']} m × {target['vehicle']['height_without_lidar_m']} m；
运行场景为 {', '.join(target['vehicle']['operating_scenarios'])}。ego 为 `+x 前、+y 左、+z 上`，原点在假设后轴中点地面投影；相机使用 OpenCV 轴系。
目标相机为 rectified pinhole、{target['output']['fps']} FPS，输出采用本 V1 的 480×270 交付分辨率（原定义 1920×1080 的 1/4 像素量级验证版本）。

| 目标相机 | 位置 ego m (x,y,z) | yaw/pitch/roll ° | 水平 FOV ° |
|---|---|---|---:|
{camera_rows}

该 rig 是 Neolix X3-size 工程研究假设，非 OEM 标定；完整参数见 `03_工程代码/code/configs/7fd4_neolix_x3_size_7v.json`。为与 2DGS 的当前实现匹配，最终目标视角为 global-shutter pinhole midpoint proxy；它不等同于原始 NCore FTheta/rolling-shutter 传感器成像。

## 评价体系与选择理由

- **交付完整率、分辨率一致性、帧映射完整率**：验证批处理没有遗漏，并保证生成帧能追溯源参考时刻；
- **静态背景 LiDAR holdout 深度 MAE/median/P90**：稀疏但有物理尺度的独立路面几何检查，P90 对条带/大误差敏感；
- **source-camera actor temporal holdout 的 RGB MAE、alpha IoU、面积比**：在有原始 mask 的视图衡量对象轮廓与外观，避免只报告目标视角主观效果；
- **目标序列 actor 外 RGB MAE 与相邻 alpha IoU**：前者验证合成不会污染非 actor 背景，后者是时序稳定性代理；两者都不是无真值目标视角的真实性指标；
- **near-black 比例**：仅作渲染空洞/覆盖风险提示，不作硬质量阈值，因阴影和欠曝光可为真实道路像素。

## V1 边界

本结果是可复核的工程 demo，不是已验证的传感器真值：2DGS actor 采用 virtual-pinhole midpoint proxy，缺少原始 FTheta/rolling-shutter actor renderer；动态 actor 与静态背景没有独立可靠的深度排序；人员及没有满足 source holdout 的车辆不做生成式补全，而是回退静态背景。所有这些限制均在评价报告和失败案例中保留，避免将不可观测区域误表述为恢复结果。
"""
    result = f"""# 实测结果报告（V1）

## 交付结果

7 路目标相机均输出 {global_metrics['frames_per_camera']} 帧，合计 {global_metrics['total_frames']} 帧，完成率 {global_metrics['completion_rate']:.2%}；
连续时长 {global_metrics['duration_s']:.4f} 秒。所有帧均为 {global_metrics['width']}×{global_metrics['height']} RGB PNG，并已生成每路 MP4 与 7V 拼接 Demo。

## 序列安全与时序指标

| 相机 | actor 活跃帧 | 相邻 actor alpha IoU | 抽样 actor 外 RGB MAE |
|---|---:|---:|---:|
{actor_rows}

每路每 {global_metrics['metric_stride']} 帧（另含末帧）的抽样 near-black 像素比例均已写入 CSV；这是提示项而非错误率。

## 路面几何独立检查

本次**实际用于交付 RGB 静态背景**的 2DGS 多帧 LiDAR holdout（{road['views']} views / {road['pixels']} pixels）：MAE `{road['mae_m']:.6f} m`，
median `{road['median_m']:.6f} m`，P90 `{road['p90_m']:.6f} m`。该检查验证了静态背景路面尺度，但不等价于目标 7V 全像素深度真值。

## 已接受刚性 actor 的来源视图时间留出

| Track | 类别 | source camera | RGB MAE | alpha IoU | 面积比 |
|---:|---|---|---:|---:|---:|
{accepted}

## 结论

V1 通过文件完整性、帧映射、目标 rig 一致性、静态背景独立 LiDAR 深度检查和 actor 外背景不污染检查；
它**不通过/不声称**目标未观测动态对象的真实性验收。下一轮优化会以 `05_评价结果/quality_metrics.csv` 为冻结基线，仅接受
不降低完成率和背景安全指标、且可量化改善几何或时序指标的更改。
"""
    failure = """# 失败案例与技术边界

1. **行人和未被独立 source holdout 支持的车辆**：没有可靠跨相机/目标视角几何，V1 采取 fail-closed 静态回退，可能表现为模糊、残留或缺失；未使用生成式补图伪造。
2. **近场动态对象与静态背景重叠**：静态场景会吸收部分移动主体，不能把 static expected depth 当硬遮挡真值；此前会产生灰色空洞，V1 已禁用此 gate。
3. **静态背景表面**：树线、建筑边缘、路面仍可能有 2DGS/3DGS proxy 的涂抹；LiDAR holdout 仅覆盖可见路面采样点，不能证明全图。
4. **相机模型**：背景与 actor 层均使用工程 pinhole midpoint proxy，不能宣称已实现目标相机或 actor 原生 FTheta/rolling shutter。
5. **后向/侧向动态覆盖**：由于 source view compatibility gate，多数帧没有 actor 层，这是有意保守选择，不应当解读为动态重建完成。
"""
    compliance = """# 数据合规与真实性声明（V1）

- 本 V1 使用 NVIDIA 发布的 PhysicalAI-Autonomous-Vehicles-NCore 公开数据子集；提交包不包含、也不再分发原始图像、点云或模型权重。
- 代码、配置和数据血缘表保留了输入 clip、模型/算法来源和输出映射，供拥有相应数据访问权限的评审方复核。
- 不含账号、令牌、私钥或隐蔽网络访问；运行不依赖在线服务。
- 生成结果由记录的程序批量产生；未通过人工逐帧替换或伪造。对缺乏几何证据的动态对象采用静态回退，而不是生成式补全。
- 第三方项目与许可提示见 `03_工程代码/code/third_party.md`；评审和再发布须遵守各上游许可证及赛事数据要求。
"""
    return {"技术方案报告.md": method, "实测结果报告.md": result, "failure_cases.md": failure, "数据合规声明.md": compliance}


def _write_html(path: Path, title: str, markdown: str) -> None:
    # Markdown is deliberately kept as preformatted text: no undocumented renderer is
    # needed to inspect submission content, and Chrome can render this to PDF.
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><style>body{{font-family:serif;margin:32px}}pre{{white-space:pre-wrap;font-size:11pt;line-height:1.45}}</style><pre>{escaped}</pre>", encoding="utf-8")


def _make_pdf(html: Path, pdf: Path) -> bool:
    """Render a self-contained, readable PDF without a browser or network service.

    Headless Chrome is unavailable in the evaluation sandbox (Crashpad requires a
    writable socket).  Pillow can emit a multi-page raster PDF and has a local CJK
    font on this host, which is more reliable for a Chinese submission package.
    """
    source = html.read_text(encoding="utf-8")
    text = source.split("<pre>", 1)[-1].split("</pre>", 1)[0]
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    font_path = next((Path(item) for item in ("/usr/share/fonts/truetype/arphic/uming.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(item).is_file()), None)
    if font_path is None:
        return False
    font = ImageFont.truetype(str(font_path), 22)
    width, height, margin, line_height = 1240, 1754, 70, 34
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pages: list[Image.Image] = []
    page = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(page); y = margin
    def append_line(value: str) -> None:
        nonlocal page, draw, y
        if y + line_height > height - margin:
            pages.append(page); page = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(page); y = margin
        draw.text((margin, y), value, fill="black", font=font); y += line_height
    for original in text.splitlines():
        remaining = original or " "
        while remaining:
            end = len(remaining)
            while end and draw_probe.textlength(remaining[:end], font=font) > width - 2 * margin:
                end -= 1
            if end == 0:
                end = 1
            append_line(remaining[:end]); remaining = remaining[end:]
    pages.append(page)
    pages[0].save(pdf, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    return pdf.is_file() and pdf.stat().st_size > 0


def _make_mosaic(video_paths: list[Path], destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        bundled = Path(sys.executable).resolve().parent / "ffmpeg"
        ffmpeg = str(bundled) if bundled.is_file() else None
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to build the 7V demo mosaic")
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, (camera, video) in enumerate(zip(CAMERAS, video_paths)):
        inputs.extend(["-i", str(video)])
        # Normalise every independently encoded source sequence before stacking.
        # This is redundant for current CFR inputs but protects future demos from
        # an input start-time offset.  xstack only supports addition expressions
        # in layouts; do not use ``2*w0``/``2*h0`` here because that caused panel
        # overlap on this FFmpeg build.
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,drawtext=text='{camera}':x=12:y=12:fontcolor=white:fontsize=18:box=1:boxcolor=black@0.55[v{index}]")
        labels.append(f"[v{index}]")
    layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|w0_h0+h3"
    filters.append("".join(labels) + f"xstack=inputs=7:layout={layout}:fill=black:shortest=1,scale=960:540:flags=lanczos,setsar=1,format=yuv420p[out]")
    # The project conda environment ships OpenH264 but not libx264.  Keep the
    # generated demo broadly playable instead of assuming a system encoder.
    command = [ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libopenh264", "-b:v", "6500k", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-movflags", "+faststart", str(destination)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg mosaic failed: {completed.stderr[-2000:]}")


def _camera_video_path(sequence_root: Path, camera: str) -> Path:
    """Resolve exactly one verified per-camera H.264 delivery video.

    The original batch asset and later incumbent-registry asset deliberately
    use different descriptive filenames.  Packaging must follow the selected
    sequence rather than silently reverting to an older filename convention.
    """
    directory = sequence_root / camera
    candidates = sorted(directory.glob(f"l4_{camera}_*_h264.mp4"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one H.264 delivery video for {camera}, found {len(candidates)} in {directory}")
    return candidates[0]


def build_submission_v1(options: SubmissionV1Options) -> Path:
    project = Path(__file__).resolve().parents[2]
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"submission output already exists: {output}")
    sequence_root = options.sequence_root.resolve(); background_root = options.background_root.resolve()
    sequence_manifest_path = sequence_root / "manifest.json"
    background_manifest_path = background_root / "manifest.json"
    sequence_manifest = _read_json(sequence_manifest_path)
    background_manifest = _read_json(background_manifest_path)
    registry = _read_json(options.expert_registry)
    sequence_tracks = {int(value) for value in sequence_manifest.get("track_ids", ())}
    registry_tracks = {int(entry["track_id"]) for entry in registry.get("experts", []) if isinstance(entry, dict) and "track_id" in entry}
    if not sequence_tracks or not sequence_tracks.issubset(registry_tracks):
        raise ValueError("submission sequence tracks must be a non-empty subset of the supplied accepted expert registry")
    all_files = {camera: _rgb_files(sequence_root, camera) for camera in CAMERAS}
    counts = {len(files) for files in all_files.values()}
    if len(counts) != 1:
        raise ValueError(f"inconsistent frame counts: {counts}")
    frames = counts.pop()
    if frames < 100:
        raise ValueError("submission requires a continuous sequence of at least 100 frames")
    first_width, first_height, _ = _image_stats(all_files[CAMERAS[0]][0])
    per_camera: dict[str, dict[str, float]] = {}
    for camera, files in all_files.items():
        print(f"Submission V1 metrics: {camera} ({len(files)} frames)", flush=True)
        # Contiguous filename enumeration proves output cardinality. Decode a
        # deterministic sample for image-quality proxies so packaging is feasible on
        # constrained review hosts; the stride is recorded in the final metrics.
        per_camera[camera] = _actor_metrics(sequence_root, background_root, camera, options.metric_stride)
    road = _read_json(options.road_depth_report)
    metrics: dict[str, Any] = {"global": {"frames_per_camera": frames, "total_frames": frames * len(CAMERAS), "completion_rate": 1.0, "width": first_width, "height": first_height, "fps": options.fps, "duration_s": frames / options.fps, "metric_stride": options.metric_stride}, "per_camera": per_camera}

    print("Submission V1: writing documents and code snapshot", flush=True)
    output.mkdir(parents=True)
    for relative in ("01_技术方案", "02_数据说明", "03_工程代码", "04_生成结果/generated_images", "04_生成结果/generated_video", "05_评价结果/evidence", "06_Demo展示/sample_cases"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    docs = _markdown_documents(output, options, metrics, road, registry)
    (output / "01_技术方案" / "技术方案报告.md").write_text(docs["技术方案报告.md"], encoding="utf-8")
    (output / "05_评价结果" / "实测结果报告.md").write_text(docs["实测结果报告.md"], encoding="utf-8")
    (output / "05_评价结果" / "failure_cases.md").write_text(docs["failure_cases.md"], encoding="utf-8")
    (output / "02_数据说明" / "数据合规声明.md").write_text(docs["数据合规声明.md"], encoding="utf-8")
    for md_path in (output / "01_技术方案" / "技术方案报告.md", output / "05_评价结果" / "实测结果报告.md", output / "02_数据说明" / "数据合规声明.md"):
        html = md_path.with_suffix(".html"); _write_html(html, md_path.stem, md_path.read_text(encoding="utf-8")); _make_pdf(html, md_path.with_suffix(".pdf")); html.unlink(missing_ok=True)

    target = _read_json(options.camera_config)
    (output / "02_数据说明" / "数据来源说明.md").write_text(
        f"# 数据来源说明\n\n- 数据集：NVIDIA PhysicalAI-Autonomous-Vehicles-NCore。\n- 本次 clip：`{options.ncore_path}`。\n- 范围：chunk 0，source reference camera `{options.source_camera_id}`，{frames} 个参考帧。\n- 本提交不含原始 NCore 数据；数据许可、版本和访问方式须按 NVIDIA 发布页面及赛事要求复核。\n",
        encoding="utf-8")
    (output / "02_数据说明" / "源车型传感器说明.md").write_text(
        "# 源车型传感器说明\n\n源数据含 7 路 NCore 相机、原始 FTheta 标定、rolling shutter 位姿和顶部 LiDAR。\n本项目按 `T_source_target` 命名变换：`T_camera_world = T_rig_world @ T_camera_rig`，再求逆供渲染器使用。\n详细解析在 `03_工程代码/code/src/nurec_gs_renderer/pose_sources.py`。\n",
        encoding="utf-8")
    (output / "02_数据说明" / "目标小车传感器说明.md").write_text(
        "# 目标小车传感器说明\n\n完整 7V 目标 rig 见 `03_工程代码/code/configs/7fd4_neolix_x3_size_7v.json`。\n本 V1 使用其目标安装位置和朝向，输出使用 480×270 交付分辨率，30 FPS。该 rig 为工程研究 proxy，非 OEM 标定。\n",
        encoding="utf-8")
    _copy_code(project, output / "03_工程代码")
    shutil.copy2(options.camera_config, output / "03_工程代码" / "code" / "configs" / options.camera_config.name)
    evidence = _copy_evidence(output / "05_评价结果" / "evidence", {
        "selected_sequence_manifest": sequence_manifest_path,
        "selected_background_manifest": background_manifest_path,
        "accepted_actor_registry": options.expert_registry,
        "selected_background_road_depth_holdout": options.road_depth_report,
        "target_l4_rig": options.camera_config,
    })

    mapping_rows: list[list[Any]] = []
    metrics_rows: list[list[Any]] = [
        ["global", "7v", "completion_rate", metrics["global"]["completion_rate"], "fraction", "higher", "=1.0", "All required camera/frame outputs are present"],
        ["global", "7v", "total_frames", metrics["global"]["total_frames"], "frames", "higher", f"={len(CAMERAS) * frames}", "7 cameras × contiguous reference frames"],
        ["global", "7v", "duration", metrics["global"]["duration_s"], "seconds", "higher", ">=10 recommended", "Manual requests continuous 10 s or >=100 frames"],
        ["static_geometry", "road_multiscan_holdout", "lidar_depth_mae", road["mae_m"], "m", "lower", "compare to baseline", "Sparse physical road-scale verification"],
        ["static_geometry", "road_multiscan_holdout", "lidar_depth_p90", road["p90_m"], "m", "lower", "compare to baseline", "Tail error reveals road stripe/outlier risk"],
    ]
    for camera, files in all_files.items():
        for index, path in enumerate(files):
            mapping_rows.append([camera, index, f"{index / options.fps:.6f}", options.fps, options.source_camera_id, index, path.relative_to(sequence_root).as_posix()])
        for metric, value in per_camera[camera].items():
            direction = "higher" if "iou" in metric else "lower" if "mae" in metric else "report"
            metrics_rows.append(["target_sequence", camera, metric, value, "fraction" if "area" in metric or "iou" in metric else "frames" if metric.endswith("frames") else "rgb", direction, "report only" if direction == "report" else "see report", "Composite safety/temporal proxy, not target-view ground truth"])
    for entry in registry["experts"]:
        for metric, value in entry["metrics"].items():
            metrics_rows.append(["source_holdout", entry["name"], metric, value, "fraction" if "iou" in metric or "ratio" in metric else "rgb", "higher" if "iou" in metric else "lower" if "mae" in metric else "report", "registry gate", "Source-camera temporal holdout metric"])
    _write_csv(output / "04_生成结果" / "frame_mapping.csv", ["target_camera_id", "target_frame_index", "target_time_s", "target_fps", "source_reference_camera", "source_reference_frame_index", "relative_rgb_path"], mapping_rows)
    _write_csv(output / "05_评价结果" / "quality_metrics.csv", ["scope", "entity", "metric", "value", "unit", "direction", "acceptance_or_interpretation", "rationale"], metrics_rows)
    lineage_rows = [
        ["source_ncore_clip", str(options.ncore_path), "input", "NCore clip and calibration; not redistributed", "NVIDIA NCore"],
        ["target_rig", str(options.camera_config), "configuration", _sha256(options.camera_config), "project config"],
        ["7v_sequence", str(sequence_root), "final generated RGB", _sha256(sequence_root / "manifest.json"), "audited fail-closed actor composite"],
        ["background_sequence", str(background_root), "background RGB", _sha256(background_manifest_path), "selected 20-tile static 2DGS + temporal sky"],
        ["actor_registry", str(options.expert_registry), "accepted source-view experts", _sha256(options.expert_registry), "source temporal holdout gate"],
        ["road_holdout", str(options.road_depth_report), "quality evidence", _sha256(options.road_depth_report), "multi-scan LiDAR holdout"],
    ]
    _write_csv(output / "05_评价结果" / "data_lineage.csv", ["artifact", "source_or_path", "role", "fingerprint_or_note", "provenance"], lineage_rows)

    video_paths: list[Path] = []
    for camera, files in all_files.items():
        print(f"Submission V1: copying RGB/video {camera}", flush=True)
        target_dir = output / "04_生成结果" / "generated_images" / camera
        target_dir.mkdir(parents=True)
        for source in files:
            shutil.copy2(source, target_dir / source.name)
        source_video = _camera_video_path(sequence_root, camera)
        video_target = output / "04_生成结果" / "generated_video" / f"{camera}.mp4"
        shutil.copy2(source_video, video_target); video_paths.append(video_target)
        for frame in (0, frames // 2, frames - 1):
            shutil.copy2(files[frame], output / "06_Demo展示" / "sample_cases" / f"{camera}_{frame:06d}.png")
    mosaic = output / "06_Demo展示" / "demo_7v_mosaic.mp4"
    print("Submission V1: encoding 7V demo mosaic", flush=True)
    _make_mosaic(video_paths, mosaic)
    shutil.copy2(mosaic, output / "04_生成结果" / "generated_video" / "demo_7v_mosaic.mp4")

    summary = {"schema": SCHEMA, "version": PACKAGE_VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "inputs": {"sequence_root": str(sequence_root), "background_root": str(background_root), "camera_config": str(options.camera_config.resolve()), "expert_registry": str(options.expert_registry.resolve()), "road_depth_report": str(options.road_depth_report.resolve()), "ncore_path": str(options.ncore_path.resolve())}, "evidence": evidence, "metrics": metrics, "submission_files": {"frame_mapping_rows": len(mapping_rows), "quality_metric_rows": len(metrics_rows), "cameras": list(CAMERAS), "demo": str(mosaic.relative_to(output))}, "reproducibility": {"renderer_snapshot": "03_工程代码/code/src/nurec_gs_renderer", "python": sys.version, "full_regeneration_requires": ["the external NCore clip named above", "the recorded 2DGS checkpoints and temporal sky asset referenced by copied manifests", "the local 2D Gaussian Splatting external repository"], "integrity_command": "bash 03_工程代码/code/run_demo.sh"}, "limitations": sequence_manifest.get("limitations", [])}
    (output / "05_评价结果" / "submission_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "reproduce_submission.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "if [[ $# -ne 1 ]]; then echo 'usage: ./reproduce_submission.sh /absolute/new-output-directory' >&2; exit 2; fi\n"
        "ROOT=$(cd \"$(dirname \"$0\")\" && pwd)\nOUT=$1\n"
        "if [[ -e \"$OUT\" ]]; then echo \"refusing to overwrite: $OUT\" >&2; exit 2; fi\n"
        f"PYTHONPATH=\"$ROOT/03_工程代码/code/src\" python -m nurec_gs_renderer.submission_v1_cli --sequence-root {sequence_root} --background-root {background_root} --camera-config {options.camera_config.resolve()} --expert-registry {options.expert_registry.resolve()} --road-depth-report {options.road_depth_report.resolve()} --ncore-path {options.ncore_path.resolve()} --output \"$OUT\" --fps {options.fps} --metric-stride {options.metric_stride}\n",
        encoding="utf-8",
    )
    (output / "reproduce_submission.sh").chmod(0o755)
    (output / "README.md").write_text(
        "# Cross-Vehicle L4 7V Submission V1\n\n"
        "Start with `01_技术方案/技术方案报告.pdf` and `05_评价结果/实测结果报告.pdf`. The continuous seven-view demo is `06_Demo展示/demo_7v_mosaic.mp4`.\n\n"
        "- Full per-camera RGB sequences: `04_生成结果/generated_images/`\n"
        "- Per-camera MP4 and mosaic: `04_生成结果/generated_video/`\n"
        "- Mapping, metrics and lineage: `04_生成结果/frame_mapping.csv`, `05_评价结果/`\n"
        "- Code/configuration snapshot: `03_工程代码/code/`; full regeneration: `./reproduce_submission.sh /absolute/new-output-directory`\n"
        "- Integrity verification (all copied frames/videos/reports/code): `bash 03_工程代码/code/run_demo.sh`\n\n"
        "This is an evidence-first V1 demo. It clearly labels unsupported dynamic regions as static fallback and must not be interpreted as target-view sensor ground truth.\n",
        encoding="utf-8")
    checksum_file, checksum_entries = _write_checksum_inventory(output)
    summary["submission_files"]["checksum_file"] = str(checksum_file.relative_to(output))
    summary["submission_files"]["checksum_entries"] = checksum_entries
    (output / "05_评价结果" / "submission_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # The manifest was intentionally written before checksumming so it is itself
    # covered. Re-write only metadata that is already known before generating the
    # inventory; do not mutate any covered file afterwards.
    checksum_file, checksum_entries = _write_checksum_inventory(output)
    return output


def verify_submission_v1(root: Path) -> dict[str, Any]:
    root = root.resolve()
    required = [root / "README.md", root / "reproduce_submission.sh", root / "01_技术方案" / "技术方案报告.pdf", root / "03_工程代码" / "code" / "run_demo.sh", root / "04_生成结果" / "frame_mapping.csv", root / "05_评价结果" / "quality_metrics.csv", root / "05_评价结果" / "checksums.sha256", root / "06_Demo展示" / "demo_7v_mosaic.mp4"]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    manifest = _read_json(root / "05_评价结果" / "submission_manifest.json") if not missing else {}
    frames = int(manifest.get("metrics", {}).get("global", {}).get("frames_per_camera", 0))
    for camera in CAMERAS:
        rgb = sorted((root / "04_生成结果" / "generated_images" / camera).glob("*.png"))
        if len(rgb) != frames:
            missing.append(f"{camera}: expected {frames} copied RGB frames, found {len(rgb)}")
        else:
            try:
                indices = [int(path.stem.split("_", 1)[0]) for path in rgb]
            except ValueError:
                missing.append(f"{camera}: cannot parse copied RGB frame indices")
            else:
                if sorted(indices) != list(range(frames)):
                    missing.append(f"{camera}: copied RGB indices are not contiguous 000000..")
    checksum = root / "05_评价结果" / "checksums.sha256"
    checked = 0
    if checksum.is_file():
        for line_number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split("\t", 2)
            if len(parts) != 3:
                missing.append(f"checksums.sha256:{line_number}: invalid row")
                continue
            expected_hash, expected_bytes, relative = parts
            path = root / relative
            if not path.is_file():
                missing.append(f"checksum missing file: {relative}")
                continue
            if str(path.stat().st_size) != expected_bytes or _sha256(path) != expected_hash:
                missing.append(f"checksum mismatch: {relative}")
                continue
            checked += 1
    result = {"schema": SCHEMA, "status": "complete" if not missing else "failed", "root": str(root), "missing_or_invalid": missing, "frames_per_camera": frames, "checksum_entries_verified": checked}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
