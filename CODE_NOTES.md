# Frozen Engineering Code Snapshot

This is the frozen snapshot of the rendering, camera-reading, sky-compositing,
2DGS data-interface and dynamic-actor selection code used for the final
submission. It is used together with [`DEPLOY.md`](DEPLOY.md), `env.example` and
`run_render_1080p_7v.sh` in this repository's root. `DEPLOY.md` is the
**authoritative reproduction entry point** for the final 1080p seven-view L4
demonstration; this file documents code boundaries, installation and interface
semantics.

The snapshot contains no NCore data, model weights, InstantNuRec/2DGS external
code, training proxies, 2DGS checkpoints, sky assets, actor registry or complete
PNG results. It can therefore reproduce the engineering and re-rendering process,
but a full training reproduction still requires obtaining data and model
authorisation and re-running the corresponding GPU training.

## Code scope

### The static background and target-view chain used in production

The final demonstration uses this chain:

```text
NCore calibration, timestamps and vehicle poses
      + static 2DGS checkpoint + temporal sky
      + target L4 seven-view rig
      -> direct 1920x1080 static RGB rasterisation into the target pinhole camera
      + actor registry accepted by the source-time holdout
      -> conservative dynamic actor composited RGB
      -> single-camera MP4s and the seven-view mosaic
```

`nurec_gs_renderer.two_dgs_target_render_cli` renders the static 2DGS into the
target camera and supports the temporal sky.
`nurec_gs_renderer.two_dgs_actor_target_render_cli` selects dynamic actors only
from the already-accepted registry; where the view, distance or visibility
conditions are not met, that region keeps the static background. The final script
uses a maximum viewing angle of 25°, a target/source distance ratio of 0.60–1.60,
and a 100 ms fade in and out, to avoid forcing an actor that lacks reliable source
evidence into a novel view.

### General Gaussian rendering chain

`nurec_gs_renderer.cli` renders 3D Gaussian PLY files in InstantNuRec or
Graphdeco format, emitting RGB, geometric alpha and expected depth. It supports:

- undistorted pinhole target cameras;
- source cameras read directly from NCore `FThetaCameraModelParameters`;
- NCore rolling-shutter start/end exposure poses and scan direction;
- multi-camera rigs and NCore vehicle trajectories;
- an optional world-direction cubemap / temporal sky;
- an optional, reversible Gaussian DC colour sidecar.

Transform names follow NCore's `T_source_target` semantics. For target camera
mounting extrinsics the code computes:

```text
T_camera_world(t) = T_rig_world(t) @ T_camera_rig
```

where `T_camera_rig` carries a point in camera coordinates into rig coordinates;
this is then converted to the OpenCV world-to-camera form that `gsplat` requires.
The camera OpenCV axes are `+x` right, `+y` down, `+z` forward. The target L4 rig
uses global-shutter midpoint poses, while the source NCore FTheta cameras retain
per-pixel rolling-shutter rays.

### Sky and dynamic experiment code

`sky_cli` builds a direction-domain cubemap or temporal sky index from the 7 NCore
observations. The sky fills only `1 − geometry_alpha`; the emitted alpha and
expected depth always retain their geometric meaning. The `dynamic_*`,
`raw_dynamic_*`, `two_dgs_actor_*` and `emernerf_*` modules are retained for
dynamic-layer attribution, training-data generation and ablation studies; they
are not prerequisites for running the final static 2DGS demonstration.

## Directory structure

| Location | Contents |
| --- | --- |
| `src/nurec_gs_renderer/` | The Python package and its CLI implementations |
| `configs/` | Camera, evaluation and historical experiment configuration; the final target rig is `configs/7fd4_neolix_x3_size_7v.json` |
| `examples/` | Frozen historical 2DGS training/diagnostic scripts, with experiment-specific hyperparameters |
| `pyproject.toml` | Package dependencies, optional dependency groups and console scripts |
| `third_party.md` | Third-party data, code, models and licence boundaries |

The historical configurations and `examples/` exist to trace experiments; they
are not final commands to copy directly, and the asset identifiers inside them
must be replaced with locally held equivalents. For the final re-render use
`env.example` and `run_render_1080p_7v.sh` in the repository root, which depend
only on environment variables and are not bound to any particular machine's
directory layout.

## Installing the environment

Install a PyTorch build compatible with the local GPU/CUDA first, then prepare
the external projects following the official instructions for
[NVIDIA InstantNuRec](https://github.com/NVIDIA/instant-nurec) and
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting).
Then install this package:

```bash
python -m pip install -e '.[ncore,sky,depth]'
```

The dependency groups mean:

| Group | Purpose |
| --- | --- |
| no optional group | PLY, `torch`, `gsplat`, pinhole rendering |
| `ncore` | Reading NCore V4 sequences, FTheta calibration and the pose graph |
| `sky` | Local SegFormer weight inference and cubemap / temporal sky construction |
| `depth` | Optional data-construction features for depth priors such as Depth Anything |
| `dev` | pytest test tooling |

The reference combination is Linux, Python 3.11, PyTorch 2.7.0 + CUDA 12.8,
`gsplat 1.5.3`, `nvidia-ncore 18.7.0`, `transformers 4.57.1` and
`huggingface-hub 0.36.2`. The first call into `gsplat` or the 2DGS CUDA
extensions may trigger JIT compilation; set `TORCH_CUDA_ARCH_LIST` to match the
local GPU — for example `8.9` for an Ada-architecture GPU.

## Minimal health check

After installation, check that the CLIs import:

```bash
PYTHONPATH=src python -m nurec_gs_renderer.cli --help
PYTHONPATH=src python -m nurec_gs_renderer.two_dgs_target_render_cli --help
PYTHONPATH=src python -m nurec_gs_renderer.two_dgs_actor_target_render_cli --help
```

If the snapshot also contains a test directory:

```bash
PYTHONPATH=src pytest -q
```

A real GPU smoke test should at minimum verify the dimensions of one target
camera's RGB, alpha and expected depth, and that the output contains no NaNs and
is not anomalously all-black. A complete seven-view re-render should produce
7×299 frames, with each single-camera video at 1920×1080, 30 FPS, 299 frames.

## Input and output semantics

The static 2DGS training proxy uses pinhole tiles obtained by exact
un-projection from the source NCore FTheta/rolling-shutter images. The final L4
output is *not* an upscaling of the 480×270 training PNGs:
`two_dgs_target_render_cli` rasterises the checkpoint directly at the target
1920×1080 pinhole intrinsics and extrinsics. The observation density of the
training proxy still bounds the recoverable high-frequency geometry and dynamic
target detail.

The source masks, instance tracks, LiDAR and cross-view observations for dynamic
actors are used only for conservative selection and acceptance. A raw dynamic
object with no verifiable surface, visibility or source-time holdout support is
not force-composited. This reduces wrong ghosting in novel views, but it also
means some vehicles and pedestrians appear as static background or as a weak
dynamic presence.

## Licence and data boundaries

Before using this snapshot, read [`third_party.md`](third_party.md) and confirm
that the NVIDIA NCore data access agreement, the InstantNuRec and 2DGS code
licences, and the licence of each local model weight all permit the intended use.
This submission must not carry or publicly redistribute source images, LiDAR,
calibration, tracks, pre-trained weights or copies of external projects.
