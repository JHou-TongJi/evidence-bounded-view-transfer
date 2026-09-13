#!/usr/bin/env bash
# Full static 2DGS road initialisation + multi-scan sparse metric-depth experiment.
#
# Run each phase independently so long training is restartable:
#   bash examples/2dgs_fullclip_road_lidar_sky.sh road-seed
#   bash examples/2dgs_fullclip_road_lidar_sky.sh build
#   bash examples/2dgs_fullclip_road_lidar_sky.sh train
#   bash examples/2dgs_fullclip_road_lidar_sky.sh evaluate
#   bash examples/2dgs_fullclip_road_lidar_sky.sh render
set -euo pipefail

PHASE=${1:?usage: $0 road-seed|build|train|evaluate|render}
PROJECT_ROOT=/home/zrz/Cross-Vehicle-Data-Rec
NCORE_PATH=/data/zrz/Cross-Vehicle/ncore/clips/7fd4d554-b155-45c6-a46a-646486029d85/pai_7fd4d554-b155-45c6-a46a-646486029d85.json
STATIC_PLY=/data/zrz/Cross-Vehicle/outputs/merged/7fd4d554/pa-multiview-5cam-2m-v1/pa-multiview-5cam-merged-7fd4-2m-v1/ply/pai_7fd4d554-b155-45c6-a46a-646486029d85/pai_7fd4d554-b155-45c6-a46a-646486029d85.ply
DYNAMIC_MASKS=/data/zrz/Cross-Vehicle/outputs/dynamic-mask-refinement/7fd4d554/chunk0-v2-all7-sam2small-source-priority-lidar/manifest.json
SKY=/data/zrz/Cross-Vehicle/outputs/sky/7fd4d554/chunk0-quality-v4-temporal.json
TWO_DGS_ROOT=${PROJECT_ROOT}/external/2d-gaussian-splatting
ROAD_SEED=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/chunk0-road-lidar-frontwide-299f-s5-v3-multiscan-depth.npz
DATASET=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/chunk0-clean-static-road-multiscan-depth-proxy-5cam-11tile-299f-480-200k-v1
MODEL=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/chunk0-clean-static-road-multiscan-depth-fullclip-11tile-6000-v1
EVALUATION=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/road-multiscan-depth-evaluation-fullclip-11tile-6000-v1-holdout.json
RENDER=/data/zrz/Cross-Vehicle/outputs/2dgs/7fd4d554/l4-front-center-clean-static-road-multiscan-depth-sky-fullclip-11tile-diagnostic-v1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=${MAX_JOBS:-4}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-2dgs-road-lidar}

case "$PHASE" in
  road-seed)
    PYTHONPATH=${PROJECT_ROOT}/src conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_road_lidar_cli \
      --ncore-path "$NCORE_PATH" --output "$ROAD_SEED" \
      --dynamic-mask-manifest "$DYNAMIC_MASKS" --chunk-index 0 \
      --source-camera-id camera_front_wide_120fov --frame-start 0 --frame-end 299 --stride 5 \
      --voxel-size-m .15 --min-voxel-observations 2 --max-points 100000 \
      --min-range-m 3 --max-range-m 60 --ground-local-z-min-m -3.2 --ground-local-z-max-m -1.4 \
      --device cuda
    ;;
  build)
    PYTHONPATH=${PROJECT_ROOT}/src conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_dataset_cli \
      --ncore-path "$NCORE_PATH" --static-ply "$STATIC_PLY" --output "$DATASET" \
      --chunk-index 0 --width 480 --height 270 --horizontal-fov-deg 55 \
      --tile-yaws-deg=-45,0,45 --rear-tile-yaws-deg=0 \
      --frame-start 0 --frame-end 299 --max-initial-points 100000 \
      --rectify-device cuda --dynamic-mask-manifest "$DYNAMIC_MASKS" \
      --road-lidar-seed "$ROAD_SEED" --replace-road-initialization \
      --road-depth-supervision --road-depth-stride 5 \
      --road-depth-holdout-every 5 --road-depth-holdout-offset 4 \
      --road-depth-min-range-m 2.0 --road-depth-ego-exclusion-radius-m 2.5 \
      --road-depth-lidar-window 2 --road-depth-max-lidar-delta-us 250000 \
      --road-depth-lidar-decay-us 100000
    ;;
  train)
    cd "$TWO_DGS_ROOT"
    PYTHONPATH=$PWD conda run --no-capture-output -n instant-nurec \
      python train.py --source_path "$DATASET" --model_path "$MODEL" \
      --iterations 6000 --eval --lambda_dssim 0 --lambda_dist 10 --lambda_normal 0.05 \
      --lambda_road_depth 0.20 --lambda_road_normal 0.02 --road_depth_start_iter 500 \
      --densify_until_iter 3000 \
      --test_iterations 500 1000 2000 3000 4000 5000 6000 \
      --save_iterations 1000 2000 3000 4000 5000 6000 \
      --checkpoint_iterations 1000 2000 3000 4000 5000 6000
    ;;
  evaluate)
    PYTHONPATH=${PROJECT_ROOT}/src:${TWO_DGS_ROOT} conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_road_depth_evaluate_cli \
      --two-dgs-root "$TWO_DGS_ROOT" --source-path "$DATASET" --model-path "$MODEL" \
      --iteration 6000 --split holdout --output "$EVALUATION" --device cuda
    ;;
  render)
    PYTHONPATH=${PROJECT_ROOT}/src:${TWO_DGS_ROOT} conda run --no-capture-output -n instant-nurec \
      python -m nurec_gs_renderer.two_dgs_target_render_cli \
      --two-dgs-root "$TWO_DGS_ROOT" --source-path "$DATASET" --model-path "$MODEL" \
      --camera-config ${PROJECT_ROOT}/configs/7fd4_neolix_x3_size_7v.json \
      --output "$RENDER" --iteration 6000 --camera-id front_center --chunk-index 0 \
      --frame-start 0 --frame-end 299 --width 480 --height 270 --device cuda \
      --sky-cubemap "$SKY" --sky-confidence-threshold .5 --sky-min-sample-count 1
    conda run --no-capture-output -n instant-nurec ffmpeg -y -framerate 30 -start_number 0 \
      -i "$RENDER/front_center/rgb/%06d.png" -frames:v 299 \
      -c:v libopenh264 -b:v 2500k -pix_fmt yuv420p -movflags +faststart \
      "$RENDER/front_center/l4_front_center_static_2dgs_road_multiscan_depth_sky_fullclip_h264.mp4"
    ;;
  *)
    echo "unknown phase: $PHASE (expected road-seed, build, train, evaluate or render)" >&2
    exit 2
    ;;
esac
