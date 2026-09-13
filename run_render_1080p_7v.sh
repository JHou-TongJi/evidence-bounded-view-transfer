#!/usr/bin/env bash
# Portable final-result replay: static 2DGS -> camera-gated actor layer -> MP4.
# Usage:
#   cd 03_工程代码 && source env.local && bash run_render_1080p_7v.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$SCRIPT_DIR/source_snapshot}"
RIG_TEMPLATE="${RIG_TEMPLATE:-$SCRIPT_DIR/configs/7fd4_neolix_x3_size_7v.json}"
CONDA_ENV="${CONDA_ENV:-instant-nurec}"
VIDEO_ENCODER="${VIDEO_ENCODER:-libopenh264}"
FPS="${FPS:-30}"
FRAME_COUNT="${FRAME_COUNT:-299}"

require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }; }
require_new_dir() { [[ ! -e "$1" ]] || { echo "refusing to overwrite existing path: $1" >&2; exit 2; }; }

: "${NCORE_SEQUENCE_JSON:?set NCORE_SEQUENCE_JSON to an authorized NCore sequence JSON}"
: "${TWO_DGS_ROOT:?set TWO_DGS_ROOT to an official 2DGS checkout}"
: "${STATIC_DATASET:?set STATIC_DATASET to the generated 20-tile 2DGS dataset}"
: "${STATIC_MODEL:?set STATIC_MODEL to the 12k static 2DGS model directory}"
: "${SKY_ASSET:?set SKY_ASSET to the temporal sky JSON asset}"
: "${ACTOR_REGISTRY:?set ACTOR_REGISTRY to the accepted rigid actor registry JSON}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new writable output root}"

STATIC_OUT="${STATIC_OUT:-$OUTPUT_ROOT/static}"
ACTOR_OUT="${ACTOR_OUT:-$OUTPUT_ROOT/actors}"
MEDIA_OUT="${MEDIA_OUT:-$OUTPUT_ROOT/media}"
RESOLVED_RIG="$OUTPUT_ROOT/resolved_target_rig.json"

require_file "$NCORE_SEQUENCE_JSON"
require_dir "$TWO_DGS_ROOT"
require_file "$TWO_DGS_ROOT/scene/gaussian_model.py"
require_dir "$STATIC_DATASET"
require_file "$STATIC_DATASET/ncore_2dgs_manifest.json"
require_dir "$STATIC_MODEL"
require_file "$STATIC_MODEL/point_cloud/iteration_12000/point_cloud.ply"
require_file "$SKY_ASSET"
require_file "$ACTOR_REGISTRY"
require_dir "$CODE_ROOT"
require_file "$CODE_ROOT/src/nurec_gs_renderer/two_dgs_target_render_cli.py"
require_file "$RIG_TEMPLATE"
require_new_dir "$STATIC_OUT"
require_new_dir "$ACTOR_OUT"
[[ ! -e "$RESOLVED_RIG" ]] || { echo "refusing to overwrite: $RESOLVED_RIG" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT" "$MEDIA_OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

CONDA_RUN=(conda run --no-capture-output -n "$CONDA_ENV")
"${CONDA_RUN[@]}" env PYTHONPATH="$CODE_ROOT/src" python - "$RIG_TEMPLATE" "$RESOLVED_RIG" "$NCORE_SEQUENCE_JSON" <<'PY'
import json
import sys
from pathlib import Path

template, output, ncore = map(Path, sys.argv[1:])
config = json.loads(template.read_text(encoding="utf-8"))
config["ncore_path"] = str(ncore.resolve())
Path(output).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
PY

"${CONDA_RUN[@]}" env PYTHONPATH="$CODE_ROOT/src:$TWO_DGS_ROOT" python -m nurec_gs_renderer.two_dgs_target_render_cli \
  --two-dgs-root "$TWO_DGS_ROOT" --source-path "$STATIC_DATASET" --model-path "$STATIC_MODEL" \
  --camera-config "$RESOLVED_RIG" --output "$STATIC_OUT" --iteration 12000 \
  --chunk-index 0 --frame-start 0 --frame-end "$FRAME_COUNT" --width 1920 --height 1080 \
  --device cuda --sky-cubemap "$SKY_ASSET" --sky-confidence-threshold 0.5 --sky-min-sample-count 1

"${CONDA_RUN[@]}" env PYTHONPATH="$CODE_ROOT/src:$TWO_DGS_ROOT" python -m nurec_gs_renderer.two_dgs_actor_target_render_cli \
  --two-dgs-root "$TWO_DGS_ROOT" --camera-config "$RESOLVED_RIG" --expert-registry "$ACTOR_REGISTRY" \
  --background-rgb-dir "$STATIC_OUT" --output "$ACTOR_OUT" --chunk-index 0 \
  --frame-start 0 --frame-end "$FRAME_COUNT" --width 1920 --height 1080 --device cuda \
  --max-view-angle-deg 25 --min-distance-ratio 0.6 --max-distance-ratio 1.6 \
  --temporal-fade-us 100000

CAMERAS=(front_left front_center front_right side_left side_right rear_left rear_right)
for camera in "${CAMERAS[@]}"; do
  require_file "$ACTOR_OUT/$camera/rgb/$(printf '%06d' $((FRAME_COUNT - 1))).png"
  "${CONDA_RUN[@]}" ffmpeg -y -framerate "$FPS" -start_number 0 \
    -i "$ACTOR_OUT/$camera/rgb/%06d.png" -frames:v "$FRAME_COUNT" \
    -c:v "$VIDEO_ENCODER" -pix_fmt yuv420p -movflags +faststart \
    "$MEDIA_OUT/l4_${camera}_1920x1080_h264.mp4"
done

"${CONDA_RUN[@]}" ffmpeg -y \
  -i "$MEDIA_OUT/l4_front_left_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_front_center_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_front_right_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_side_left_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_side_right_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_rear_left_1920x1080_h264.mp4" \
  -i "$MEDIA_OUT/l4_rear_right_1920x1080_h264.mp4" \
  -filter_complex "[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[2:v]scale=640:360[v2];[3:v]scale=640:360[v3];[4:v]scale=640:360[v4];[5:v]scale=640:360[v5];[6:v]scale=640:360[v6];[v0][v1][v2][v3][v4][v5][v6]xstack=inputs=7:layout=0_0|640_0|1280_0|0_360|1280_360|0_720|1280_720:fill=black,format=yuv420p[mosaic]" \
  -map "[mosaic]" -an -r "$FPS" -c:v "$VIDEO_ENCODER" -pix_fmt yuv420p -movflags +faststart \
  "$MEDIA_OUT/l4_7v_mosaic_1920x1080_h264.mp4"

printf 'render replay completed\nstatic=%s\nactors=%s\nmedia=%s\nrig=%s\n' "$STATIC_OUT" "$ACTOR_OUT" "$MEDIA_OUT" "$RESOLVED_RIG"
