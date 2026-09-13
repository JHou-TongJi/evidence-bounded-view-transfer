#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if python -c 'import numpy, PIL' >/dev/null 2>&1; then
  RUN=(python)
elif command -v conda >/dev/null 2>&1; then
  RUN=(conda run --no-capture-output -n "${CROSS_VEHICLE_CONDA_ENV:-instant-nurec}" python)
else
  echo 'Need Python with numpy and Pillow, or conda env instant-nurec.' >&2; exit 2
fi
PYTHONPATH="$ROOT/03_工程代码/code/src" "${RUN[@]}" -m nurec_gs_renderer.submission_v1_cli --verify "$ROOT"
echo "Open 06_Demo展示/demo_7v_mosaic.mp4 or any 04_生成结果/generated_video/*.mp4"
