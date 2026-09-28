# Engineering Code and Reproduction Guide

This repository provides the frozen code for the final seven-view L4
demonstration, the target camera rig, a portable re-render script and an
environment-variable template. It supports two scopes of reproduction:

1. **Re-rendering the final result** — given this scene's static 2DGS
   checkpoint, temporal sky and actor registry, regenerate 7×299 RGB frames at
   1920×1080, the seven single-camera MP4s and the seven-view mosaic;
2. **Rebuilding the assets from licensed NCore data** — the full engineering
   pipeline from the NCore clip through InstantNuRec export, sky construction,
   2DGS training and the actor registry. That pipeline requires licensed data,
   external repositories and lengthy GPU training, and is not distributed with
   this package.

[`configs/7fd4_neolix_x3_size_7v.json`](configs/7fd4_neolix_x3_size_7v.json) is
the target seven-view L4 engineering rig. Its `ncore_path` is a placeholder; the
run script writes a resolved copy using `NCORE_SEQUENCE_JSON`. `src/` holds the
frozen code and configuration files. Training data, model weights, the 2DGS
checkpoint, sky assets, the actor registry and the complete PNG sequence are all
excluded from this package.

## External projects, data and licences

| Component | Purpose | How to obtain |
| --- | --- | --- |
| [NVIDIA InstantNuRec](https://github.com/NVIDIA/instant-nurec) | Exporting the initial Gaussian scene and upstream assets from NCore V4 driving logs | Follow the official installation instructions and its code/model licences |
| [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) | Training and rendering the surface-constrained static 2DGS background | `git clone --recursive` the official repository; observe its Gaussian-Splatting License |
| [NVIDIA PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | Multi-camera, LiDAR, calibration, timestamp and scene-description input | Apply for and observe the dataset access agreement; must not be redistributed with a submission |
| [NVIDIA SegFormer B5 ADE20K](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | Sky semantic segmentation when building the world-direction temporal sky | Download the weights locally; observe the model card and upstream licence |

The results and code in this submission carry none of those external
repositories, data or weights. Confirm data access rights, model licences and
competition rules before starting.

## Reference environment

The combination below is the reference environment used for GPU rendering in this
submission. Other GPUs may use compatible versions, but the tests and a small
smoke run should be repeated.

| Category | Validated version / requirement |
| --- | --- |
| OS and Python | Linux; Python 3.11 |
| PyTorch / CUDA | PyTorch 2.7.0 + CUDA 12.8; the NVIDIA GPU compute capability must be compatible with the PyTorch CUDA extensions |
| Core rendering libraries | `gsplat 1.5.3`, `numpy 1.26.4`, `Pillow 12.2.0`, `plyfile >= 1.0` |
| NCore reading | `nvidia-ncore 18.7.0`, `universal-pathlib >= 0.2` |
| Optional sky/depth dependencies | `transformers 4.57.1`, `huggingface-hub 0.36.2`, `safetensors >= 0.4.3`, `scipy >= 1.10` |
| Video encoding | FFmpeg 8.1.2; an H.264 encoder (this submission used `libopenh264`) and the `yuv420p` pixel format |

Install the CUDA/PyTorch environment following the official InstantNuRec and 2DGS
repositories first, then install the renderer from this repository:

```bash
python -m pip install -e '.[ncore,sky,depth]'
```

The first use of `gsplat`, the 2DGS submodules or other CUDA extensions may
trigger JIT compilation. Set `TORCH_CUDA_ARCH_LIST` to the GPU's compute
capability; the reference value for the RTX 40 series is `8.9`.

## Asset interfaces

Copy `env.example` to a shell configuration file of your choice and fill in the
paths. The paths are yours to decide; the script assumes no server directory
structure.

| Variable | Required content | Minimum check |
| --- | --- | --- |
| `NCORE_SEQUENCE_JSON` | Licensed NCore sequence JSON | The file exists and the relative component-store paths are reachable |
| `TWO_DGS_ROOT` | Root of the official 2DGS checkout | Contains `scene/gaussian_model.py` |
| `STATIC_DATASET` | This scene's 20-tile training proxy dataset | Contains `ncore_2dgs_manifest.json` |
| `STATIC_MODEL` | Directory of the trained static 2DGS model | Contains `point_cloud/iteration_12000/point_cloud.ply` |
| `SKY_ASSET` | Temporal sky JSON index | The file exists and the slot assets it references are reachable |
| `ACTOR_REGISTRY` | Rigid actor registry accepted by the source-time holdout | `chunk0-rigid-incumbent-v1.json` or an equivalent schema file |
| `OUTPUT_ROOT` | A newly created output root | Writable; must not overwrite existing formal results |

The final asset identifiers for this submission are: static checkpoint
`chunk0-clean-static-road-nearfield-fullclip-20tile-12000-v1`, actor registry
`chunk0-rigid-incumbent-v1.json`, sequence length 299 frames, frame rate 30 FPS.

## Re-rendering the final 1080p result

```bash
cp env.example env.local
# Edit env.local and fill in the variables from the table above.
source env.local
bash run_render_1080p_7v.sh
```

The script runs in this order:

1. Replace the NCore input placeholder in the target rig with
   `NCORE_SEQUENCE_JSON`, producing a resolved rig for this run;
2. call `two_dgs_target_render_cli` to rasterise 7×299 static RGB frames at
   1920×1080 directly from the static 2DGS, the target rig and the temporal sky;
3. call `two_dgs_actor_target_render_cli`, which reads the same static RGB and
   the frozen registry and selects source-holdout-accepted actor experts under a
   maximum viewing angle of 25°, a distance ratio of 0.60–1.60 and a 100 ms fade;
   frames without reliable evidence keep the static background;
4. encode the seven RGB streams into 30 FPS H.264 `yuv420p` MP4s with FFmpeg and
   produce the 3×3 seven-view mosaic.

The output directory contains `static/`, `actors/`, `media/` and the resolved rig
for that run. The script refuses to overwrite a non-empty `static/` or `actors/`
directory, to avoid clobbering existing experimental results.

The final 1080p images are rasterised directly through the target pinhole camera
parameters; they are not a simple upscaling of the older 480×270 PNGs. The static
2DGS training proxy is still 480×270, so the higher output resolution mainly
improves sampling density and edge presentation; dynamic-object completeness and
the range of recoverable geometry remain bounded by the original observations and
training assets.

## Rebuilding the assets from NCore

A full rebuild should proceed in the stages below, saving the manifest, commands,
software versions and input hashes at each stage:

1. **Upstream static scene initialisation** — run the InstantNuRec export on the
   licensed NCore clip, retaining the PLY, the input clip identifier and the
   model version;
2. **Dynamic instances and sky assets** — build the temporal sky from NCore
   calibration, FTheta/rolling-shutter poses, dynamic instance masks and
   SegFormer;
3. **Static 2DGS training proxy** — run `two_dgs_dataset_cli` to generate the
   20-tile 480×270 pinhole proxy, the dynamic exclusion masks and multi-frame
   LiDAR road supervision;
4. **Static background training and selection** — train 12,000 iterations with
   the official 2DGS, freezing densification after the first 3,000; select the
   checkpoint on an independent road LiDAR holdout;
5. **Actor expert training and selection** — write to the registry only those
   rigid actors meeting the RGB, alpha IoU and area thresholds on the source-time
   holdout; dynamic targets failing the thresholds fall back to the static
   background in the target view;
6. **Target seven-view re-render and acceptance** — run this repository's script
   and check per-camera 299-frame completeness, video metadata, outside-actor RGB
   preservation and `evidence/checksums.sha256`.

`examples/` retains some historical experiment scripts, useful for understanding
dataset construction and training hyperparameters; the authoritative final
delivery is this file together with `env.example` and `run_render_1080p_7v.sh`.
Because the checkpoint, sky, actor registry and training data are not distributed
with this package, a full rebuild requires re-obtaining licensed data and running
the training stages above; it cannot be completed from this package alone.

## Verification

From the repository root:

```bash
sha256sum -c evidence/checksums.sha256
```

This verifies **the evidence files shipped in this repository**. A newly rendered
output directory should be checked independently for file count, video frame
count, resolution, frame rate and manifest consistency, and must not overwrite
the reference media in this package.
