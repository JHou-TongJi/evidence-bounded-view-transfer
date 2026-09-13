#!/usr/bin/env bash
# Full-chunk static 2DGS experiment for the 100-degree L4 front-centre view.
#
# Usage:
#   bash examples/2dgs_fullclip_100fov.sh build
#   bash examples/2dgs_fullclip_100fov.sh train
#   bash examples/2dgs_fullclip_100fov.sh render
#
# This is a static-background diagnostic.  Dynamic source masks exclude
# vehicles, people and work equipment from RGB supervision; they do not make
# 2DGS a dynamic reconstruction method.
set -euo pipefail

PHASE=${1:?usage: $0 build|train|render}
PROJECT_ROOT=/home/zrz/Cross-Vehicle-Data-Rec
NCORE_PATH=/data/zrz/Cross-Vehicle/ncore/clips/7fd4d554-b155-45c6-a46a-646486029d85/pai_7fd4d554-b155-45c6-a46a-646486029d85.json
STATIC_PLY=/data/zrz/Cross-Vehicle/outputs/merged/7fd4d554/pa-multiview-5cam-2m-v1/pa-multiview-5cam-merged-7fd4-2m-v1/ply/pai_7fd4d554-b155-45c6-a46a-646486029d85/pai_7fd4d554-b155-45c6-a46a-646486029d85.ply
DYNAMIC_MASKS=/data/zrz/Cross-Vehicle/outputs/dynamic-mask-refinement/7fd4d554/chunk0-v2-all7-sam2small-source-priority-lidar/manifest.json
TWO_DGS_ROOT=${PROJECT_ROOT}/external/2d-gaussian-splatting
# v1 was audited complete (3,289 images + COLMAP sparse/0), so reuse it.
# Keep model/render artifacts versioned separately to avoid overwriting runs.
DATASET=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/chunk0-clean-static-proxy-5cam-11tile-299f-480-100k-v1
MODEL=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/chunk0-clean-static-fullclip-11tile-6000-v2
RENDER=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/l4-front-center-clean-static-fullclip-11tile-diagnostic-v2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=${MAX_JOBS:-4}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-2dgs-fullclip}

case "$PHASE" in
  build)
    PYTHONPATH=${PROJECT_ROOT}/src conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_dataset_cli \
      --ncore-path "$NCORE_PATH" --static-ply "$STATIC_PLY" --output "$DATASET" \
      --chunk-index 0 --width 480 --height 270 --horizontal-fov-deg 55 \
      --tile-yaws-deg=-45,0,45 --rear-tile-yaws-deg=0 \
      --frame-start 0 --frame-end 299 --max-initial-points 100000 \
      --rectify-device cuda --dynamic-mask-manifest "$DYNAMIC_MASKS"
    ;;
  train)
    cd "$TWO_DGS_ROOT"
    PYTHONPATH=$PWD conda run --no-capture-output -n instant-nurec \
      python train.py --source_path "$DATASET" --model_path "$MODEL" \
      --iterations 6000 --eval --lambda_dssim 0 --lambda_dist 10 --lambda_normal 0.05 \
      --densify_until_iter 3000 \
      --test_iterations 500 1000 2000 3000 4000 5000 6000 \
      --save_iterations 1000 2000 3000 4000 5000 6000 \
      --checkpoint_iterations 1000 2000 3000 4000 5000 6000
    ;;
  render)
    PYTHONPATH=${PROJECT_ROOT}/src:${TWO_DGS_ROOT} conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_target_render_cli \
      --two-dgs-root "$TWO_DGS_ROOT" --source-path "$DATASET" --model-path "$MODEL" \
      --camera-config ${PROJECT_ROOT}/configs/7fd4_neolix_x3_size_7v.json \
      --output "$RENDER" --iteration 6000 --camera-id front_center --chunk-index 0 \
      --frame-start 0 --frame-end 299 --width 480 --height 270 --device cuda
    conda run --no-capture-output -n instant-nurec ffmpeg -y -framerate 30 -start_number 0 \
      -i "$RENDER/front_center/rgb/%06d.png" -frames:v 299 \
      -c:v libopenh264 -b:v 2500k -pix_fmt yuv420p -movflags +faststart \
      "$RENDER/front_center/l4_front_center_static_2dgs_fullclip_11tile_h264.mp4"
    ;;
  *)
    echo "unknown phase: $PHASE (expected build, train or render)" >&2
    exit 2
    ;;
esac
