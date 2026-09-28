# Measured Results: 1080p L4 Seven-View Re-Render

## Conclusion

This final result completes a full re-render from a single passenger-car NCore
road collection into an engineering-defined L4 seven-view rig. All seven video
streams are directly rasterised 1920×1080 RGB at 30 FPS, 299 frames, 9.966667 s,
corresponding to 2,093 final PNGs at 100% completion. The result passes the
file-and-video consistency check, an independent road-LiDAR check on the same
static 2DGS model, the rigid-actor source-camera temporal holdout, and the final
1080p outside-actor background-preservation check.

The result is a cross-vehicle multi-view reconstruction demonstration and a
baseline for subsequent model research. The target rig is an X3-size-class
engineering assumption, not an OEM calibration, and the target novel views have
no per-pixel ground truth. This report therefore does not substitute subjective
impression, target-view RGB MAE, or actor alpha for proof of novel-view veracity.

## 1. Overview of the measured data

### 1.1 Data source, scene and licence boundary

| Item | Measured configuration |
|---|---|
| Dataset | NVIDIA PhysicalAI Autonomous Vehicles NCore V4 reconstruction-ready subset |
| Final sample | clip `7fd4d554-b155-45c6-a46a-646486029d85`, chunk 0 |
| Scene | Daytime, overcast, low-speed suburban/residential road; road, kerbs, greenery, houses, parked vehicles, work vehicles and a few people |
| Source sensors | 7 FTheta RGB cameras, exposure START/END poses, rig/world pose, `lidar_top_360fov`, cuboid tracks, semantic/instance masks |
| Source timing | 299 reference timestamps, 30 FPS, approximately 9.97 s |
| Static training scale | 5 physical camera contexts, 20 virtual-pinhole tiles per timestamp, `299 × 20 = 5,980` proxy images at 480×270 |
| Target output | 7 rectified pinhole RGB streams; each 1920×1080, 299 frames, 30 FPS |
| Data licensing | Data access and use follow the original publisher's terms. The submission contains only derived results, a few representative frames, configuration and evaluation evidence; it does not redistribute source images, LiDAR, weights or the full PNG sequence |

The FTheta and rolling-shutter models of the source images are used for data
reading, sky construction, instance supervision and geometric audit; the final
target output uses an engineering-defined pinhole camera with a global-shutter
exposure-midpoint model. See [`data-sources.md`](data-sources.md) and
[`data-compliance.md`](data-compliance.md).

### 1.2 Final delivery scale

The final output consists of 7 synchronised single-camera videos and 1 seven-view
mosaic. Each single-camera MP4 is encoded directly from the final
actor-composite 1080p PNGs.

| Check | Measured result | Verdict |
|---|---:|---|
| Single-camera video count | 7 | Complete |
| Single-camera resolution | 1920×1080 | Meets protocol |
| Encoding / pixel format | H.264 / yuv420p | Decodable |
| Frame rate | 30 FPS | Meets protocol |
| Frames per camera | 299 | Meets protocol |
| Duration per camera | 9.966667 s | Consistent with 299/30 |
| Corresponding PNGs across seven cameras | 2,093 / 2,093 | 100% |
| Seven-view mosaic | 1920×1080, 299 frames, 30 FPS | For synchronised viewing only |

The mosaic downscales each of the 7 native views to 640×360 and places them in a
3×3 grid; the centre and bottom-centre cells are black layout placeholders. It
does not replace the native single-camera videos and takes no part in image
quality scoring.

## 2. Engineering environment

### 2.1 Validated environment

| Category | Validated configuration |
|---|---|
| Hardware | NVIDIA RTX 4090 (Ada, sm_89); long jobs on server GPU 1 / GPU 3 |
| OS / Python | Linux / Python 3.11 |
| PyTorch / CUDA | PyTorch 2.7.0 / CUDA 12.8 |
| Core rendering | 2D Gaussian Splatting, `gsplat 1.5.3`, `numpy 1.26.4`, `Pillow 12.2.0`, `plyfile >=1.0` |
| NCore | `nvidia-ncore 18.7.0` |
| Auxiliary models | SegFormer B5 ADE20K; Depth Anything V2 Metric Outdoor Small, used only as a LiDAR-anchored weak prior |
| Video tooling | FFmpeg 8.1.2, `libopenh264`, `yuv420p` |

External projects include NVIDIA InstantNuRec and the official 2D Gaussian
Splatting. CUDA extensions are compiled locally on the target machine; RTX 40
series uses `TORCH_CUDA_ARCH_LIST=8.9`. For detailed dependencies and asset
interfaces see [`../CODE_NOTES.md`](../CODE_NOTES.md).

### 2.2 Run entry point

With authorised NCore data, the static 2DGS checkpoint, the temporal sky and the
actor registry in place, the final re-render entry point is:

```bash
cp env.example env.local
# Fill in NCORE_SEQUENCE_JSON, TWO_DGS_ROOT, STATIC_DATASET,
# STATIC_MODEL, SKY_ASSET, ACTOR_REGISTRY, OUTPUT_ROOT
source env.local
bash run_render_1080p_7v.sh
```

The script resolves the target rig, renders the static seven views, performs
fail-closed actor compositing, encodes 7 single-camera MP4s and produces the
synchronised mosaic. Retraining completely from NCore still requires the external
repositories, licensed data and model weights; it cannot be done from the
slimmed-down submission package alone.

### 2.3 Compute volume and processing time

| Stage | Auditable scale | Recorded this run |
|---|---:|---|
| Static training proxy construction | 5,980 proxy images at 480×270 | Auditable via manifest |
| Static 2DGS training | 12,000 iterations; densification frozen after the first 3,000 | Checkpoint/evaluation records auditable; total training wall-clock not written to the manifest |
| Static L4 seven-view 1080p rasterisation | 7×299 frames | Asset file time window approximately 100 min 40 s |
| Actor compositing and seven-stream encoding | 7×299 frames + 7 MP4s | Asset file time window approximately 52 min 06 s |
| Seven-view mosaic | 299 frames | Written at 17:28:20 |

The two time windows are computed from the first and last file modification times
in the 2026-08-26 formal output directory. They illustrate the actual workload of
this run only. They are not a performance benchmark isolating I/O, caching and
concurrency, and they exclude training and asset-construction time for the 2DGS,
actors and sky. Because the earlier experiments did not record end-to-end
wall-clock uniformly, this report does not fill in estimated values; a unified
runner should in future write stage start/end times, peak VRAM and a hardware
identifier into the manifest.

## 3. Method and models

### 3.1 Target vehicle views and coordinate conversion

The target is a research seven-view rig at the Neolix X3 size class, not an OEM
calibration. All seven cameras are 1920×1080 rectified pinholes with a 100°
horizontal field of view at heights of approximately 1.62–1.72 m, covering front
centre, front left, front right, side left, side right, rear left and rear right.

The target ego frame is right-handed: `+x` forward, `+y` left, `+z` up; cameras
use the OpenCV frame: `+x` right, `+y` down, `+z` forward. NCore's
`T_source_target` carries a source point into the target frame, so target camera
poses compose as:

$$
\Large T_{camera\_world}(t) = T_{rig\_world}(t) \cdot T_{camera\_rig}
$$
$$
\Large view\_matrix(t) = \mathrm{inverse}(T_{camera\_world}(t))
$$

The virtual L4 vehicle follows the source collection vehicle's world trajectory
at each instant, replacing only the passenger car's sensor mounting layout with
the target seven-view rig. Complete camera intrinsics and extrinsics are in
[`configs/7fd4_neolix_x3_size_7v.json`](../configs/7fd4_neolix_x3_size_7v.json).

### 3.2 Final generation pipeline

```text
NCore 7V RGB / calibration / rolling pose / LiDAR / mask
  ├─ Native FTheta -> 20 exposure-midpoint virtual-pinhole tiles
  ├─ Source-priority dynamic mask excludes dynamic RGB supervision
  ├─ Multi-scan LiDAR road depth supervision -> 12k static 2DGS
  ├─ 7 cameras x 18 time slots -> world-direction temporal sky
  ├─ Source-camera temporal holdout -> registry of 6 rigid actor experts
  └─ Target L4 seven-view direct 1920x1080 rasterisation
       ├─ Static 2DGS RGB/alpha
       ├─ Temporal sky composited only behind low alpha
       └─ Actor admitted at view angle <=25 deg, distance ratio 0.60-1.60,
          inside the time window, with 100 ms fade
```

The static layer uses surface-aligned 2DGS; the road's metric scale is
constrained by real LiDAR, and Depth Anything V2 supplies only a low-weight prior
within the LiDAR neighbourhood. The sky cubemap fills only transparent
background; it does not cover wrongly high-alpha geometry. The dynamic layer
admits only rigid experts that meet the source-camera temporal holdout; at most
one expert is selected per track per timestamp, and the renderer falls back to
the static background when the evidence conditions are not met. Because the
static expected depth may itself contain dynamic leakage, this final does not use
it as hard occlusion ground truth for actors.

### 3.3 Routes tested and selection conclusions

The project validated not only the final route but also systematically ruled out
several approaches that "run end to end but do not hold up on quality". The table
lists representative conclusions only; no rejected route entered the final.

| Route | Representative measured result | Decision |
|---|---|---|
| InstantNuRec static PLY + dynamic NPZ | Recovers some near vehicles; 43.9% MAE improvement in the four-frame changed dynamic region, but only 1.68% full-image and 0.30% temporal, with multiple outlines remaining | Diagnostic only; not the final dynamic layer |
| clean-static v3/v4 + canonical actor | v3: 31.79% ROI improvement over 67 frames, 2.07% full-image, but temporal delta worsened 19.98%; v4 background structure cleaner but image quality regressed | Not serialised |
| Street Gaussians v1–v7 | v7 better than v6, but exact four-frame MAE 0.06893, still worse than static 0.06094; high actor/static alpha overlap | Rejected for production |
| Native FTheta/rolling direct actor + pose correction | Short windows improve ROI; 67-frame temporal residual worsened by 30.65% and 35.77% respectively | Stopped adding iterations / tuning pose |
| SAM2 mask, raw LiDAR actor, RoMa correspondence | SAM2 as supervision sidecar only; strict RGB-LiDAR densification added just 1 point with 2.81% surface observed; both RoMa geometric audits accepted 0 | Insufficient evidence; no actor written |
| EmerNeRF native FTheta/rolling | v2: 18.30% ROI improvement over a 22-frame short window, but the longer v3 window was negative; v5 visible-alpha IoU only 0.1015 | Not carried into persons/sequence/L4 |
| 11-tile -> 20-tile static 2DGS | Multi-scan LiDAR, near-field/lateral tiles and the 12k holdout selection improve road outline and depth; still does not solve dynamic leakage | Chosen as the final static background |
| Rigid actor expert registry | 6 source-time holdout experts pass the RGB/silhouette/area thresholds; the target side then gates fail-closed on direction, distance and time | Adopted as a limited dynamic augmentation |

Together these experiments show that the current deliverable capability comes
mainly from static surfaces, the world-direction sky, continuous target poses and
a conservative local rigid actor layer. Complete novel-view reconstruction of
dynamic objects remains the principal bottleneck.

## 4. Metric attainment

### 4.1 Evaluation principles and metric definitions

The target L4 novel views have no synchronised real imagery, so metrics are
organised in four layers — delivery completeness, static geometry, source-side
actor holdout, and final compositing-safety / temporal proxies:

- **Completion rate** — output frames present and decodable / frames required.
- **Road depth MAE** — `mean(|d_render − d_lidar|)` over valid road pixels of raw
  LiDAR in independent proxy views; median and P90 describe typical and tail
  error respectively.
- **Source actor masked RGB MAE** — computed only inside the instance mask, on
  held-out time frames of the source camera.
- **Alpha IoU / area ratio** — silhouette intersection-over-union and area ratio
  between predicted `alpha > 0.5` and the instance mask.
- **Outside-actor RGB MAE** — compares composited RGB against the static
  background at the same resolution, over final 1080p pixels where
  `actor alpha ≤ 1/255`. It checks only whether the compositor polluted the
  region outside the actor.
- **Adjacent alpha IoU** — IoU of `alpha > 0.5` between adjacent frames; a proxy
  for silhouette continuity. Objects entering or leaving the frustum, occlusion,
  and gate switching all reduce it.

Apart from completion rate and protocol consistency, each metric covers only its
own explicitly stated valid domain. They must not be combined and read as
"per-pixel ground-truth accuracy over the target L4 views".

### 4.2 Static road LiDAR holdout

The static 2DGS uses exactly the same iteration-12000 model as the final, and is
evaluated on 240 independent proxy views over 6,067,229 valid projected road
LiDAR pixels:

| Valid road pixels | MAE | Median | P90 |
|---:|---:|---:|---:|
| 6,067,229 | 1.067868 m | 0.253191 m | 1.997339 m |

The result shows the model retains the metric scale constraint on LiDAR-supported
static road surface; the P90 simultaneously shows a pronounced depth tail error
in near-field, edge and sparsely covered regions. It does not evaluate vehicles,
persons, or depth across all seven-view pixels. Raw evidence:
[`evidence/road_lidar_holdout_same_static_model.json`](../evidence).

### 4.3 Rigid actor source-camera temporal holdout

| Track | Class | Source camera | Masked RGB MAE | Alpha IoU | Area ratio |
|---:|---|---|---:|---:|---:|
| 17 | heavy_truck | camera_cross_right_120fov | 0.051415 | 0.9663 | 1.0064 |
| 17 | heavy_truck | camera_front_wide_120fov | 0.038291 | 0.8691 | 1.0007 |
| 17 | heavy_truck | camera_rear_right_70fov | 0.062443 | 0.8479 | 1.0576 |
| 18 | automobile | camera_cross_right_120fov | 0.046973 | 0.9493 | 1.0147 |
| 42 | automobile | camera_cross_left_120fov | 0.024260 | 0.9753 | 1.0051 |
| 67 | automobile | camera_cross_left_120fov | 0.034380 | 0.9119 | 1.0239 |

These six entries prove only that each expert passed selection on its own source
camera's held-out time interval. They do not prove correctness in the target L4
novel view. The final renderer continues to restrict them by direction, distance
and time gates. Raw evidence:
[`evidence/accepted_actor_registry.json`](../evidence).

### 4.4 Final 1080p compositing safety and temporal proxies

The following values are recomputed frame by frame over the final 2,093
actor-composite 1080p PNGs; the 480p results are not carried over:

| Target camera | Actor-active frames | Outside-actor RGB MAE | Max outside-actor MAE | Adjacent alpha IoU mean | P05 |
|---|---:|---:|---:|---:|---:|
| front_center | 147 / 299 | 0.000000774 | 0.000005411 | 0.883610 | 0.817488 |
| front_left | 62 / 299 | 0.000000229 | 0.000003746 | 0.796156 | 0.104079 |
| front_right | 102 / 299 | 0.000001794 | 0.000017870 | 0.811464 | 0.209195 |
| side_left | 37 / 299 | 0.000000201 | 0.000004737 | 0.631412 | 0.000000 |
| side_right | 115 / 299 | 0.000002880 | 0.000062388 | 0.706673 | 0.215680 |
| rear_left | 18 / 299 | 0.000000030 | 0.000000895 | 0.720364 | 0.000000 |
| rear_right | 46 / 299 | 0.000001123 | 0.000023918 | 0.479380 | 0.000000 |

All outside-actor MAEs are near zero, showing the compositor essentially
preserved the static background outside actor regions. This does not prove the
actor's own geometry or texture is correct. The lower P05 on the side and rear
cameras reflects objects entering and leaving the frustum, insufficient source
coverage, and strict gate rejection. The project retains this fail-closed
behaviour rather than forcing completion with an unvalidated model. Per-camera
evidence is in the seven JSON files such as
[`evidence/front_center.json`](../evidence).

### 4.5 What was and was not attained

| Item | Conclusion |
|---|---|
| Seven-view file and video protocol completeness | Attained: 7×299, 100% |
| Native 1080p direct output | Attained: not upsampled from 480p |
| Independent LiDAR quantification of the static road | Completed and reported as an error distribution; not equivalent to passing a safety accuracy threshold |
| Source-side actor temporal holdout | 6 rigid experts pass the registry thresholds |
| Outside-actor background preservation | Attained: per-camera mean MAE `2.97e-8` to `2.88e-6` |
| Per-pixel ground-truth validation in the target L4 view | **Not attained**: no synchronised real target-rig ground truth exists |
| Complete dynamic object and person reconstruction | **Not attained** |
| Real target lens / rolling shutter / exposure / ISP simulation | **Not attained** |
| Reliable static-depth actor occlusion ordering | **Not attained**; the hard gate is not enabled in the final |

## 5. Result samples

> The three sample stills referenced in this section
> (`source_vehicle_vs_target_l4_frame270.png`,
> `target_l4_front_center_frame020.png`,
> `target_l4_7v_mosaic_frame000.png`) are withheld from this repository under §4
> of [`data-compliance.md`](data-compliance.md). The moving equivalents are in
> [`../demo/`](../demo).

### 5.1 Source passenger car and the corresponding target directions

The left column holds the source passenger car's forward wide, left cross wide
and right cross wide FTheta views; the right column holds the target L4 front
centre, side left and side right pinhole views at the same reference time on the
same world trajectory. Mounting position, orientation, intrinsics and imaging
model all differ, so the two columns should not agree per pixel. The comparison
confirms the cross-vehicle field-of-view change and simultaneity, and also shows
objectively that near-vehicle appearance still blurs under a large view change.

### 5.2 Target L4 front centre, native 1080p

This frame comes directly from the final `front_center` actor-composite RGB PNG.
Road direction, the parked work vehicle and background layout are legible, but
the sky boundary, tree line, near vehicles and local surfaces still show 2DGS
smearing. 1920×1080 only raises sampling density; it adds no unobserved geometry.

### 5.3 Target L4 seven-view synchronised preview

The layout runs front left / front centre / front right, side left / placeholder /
side right, rear left / placeholder / rear right. The image is for checking
simultaneity and coverage direction across the seven streams; each tile is
downscaled to 640×360, so image-quality judgements should return to the seven
native single-camera videos.

This final deliberately produces no pseudo point-cloud or BEV visualisation that
would make the sparse LiDAR look denser than it is. Road geometry is instead
re-checked through the numerical holdout of independently projected LiDAR, whose
statistics and raw evidence are given in §4.2.

## 6. Analysis of practical benefit

### 6.1 Data reuse and collection cost

Once the static scene assets are built from a single passenger-car multi-sensor
collection, the target rig JSON can be swapped and seven synchronised views at
different mounting poses generated in batch. This run produced 2,093 target
camera-timestamps from 299 source reference timestamps, showing that the pipeline
can reuse existing world geometry, trajectory, sky and some rigid object assets.
That reduces dependence on real collection from the target vehicle for early
camera-layout argumentation, demonstration video production and algorithm
interface integration.

It does not, however, license reading "one fewer collection with the target
vehicle" as "no target-vehicle validation required". Real target lens
calibration, exposure/ISP, vehicle-envelope occlusion and target-view dynamic
ground truth still require a real vehicle or high-confidence simulation. What the
current pipeline reduces is the cost of design validation and presentation, not
the cost of safety validation.

### 6.2 Uses this can support

- Visualisation of the seven-camera mounting scheme and field-of-view coverage
  for an L4 delivery vehicle;
- integration testing of cross-vehicle data interfaces, time synchronisation,
  video protocol and batch-processing chain;
- method comparison for static scene reconstruction, road LiDAR supervision and
  conservative actor compositing;
- a common static background and reject-option baseline for subsequent
  view-dependent, visibility-aware dynamic models.

### 6.3 Uses this cannot directly support

- Perception training ground truth or dynamic-object ground-truth annotation for
  autonomous driving;
- collision-distance measurement, safety certification, closed-loop
  driving-policy training;
- judgements about the veracity of pedestrian behaviour, occlusion relationships
  or complete vehicle surfaces;
- any claim to be a real calibration or real camera stream from Neolix or any
  other OEM model.

## 7. Failure cases and directions for improvement

The principal failure today is not video packaging or the coordinate chain, but
that the source observations under-constrain the geometry, appearance and
visibility of the target novel view: the static layer absorbs moving objects,
rigid actors cover only already-observed surfaces, and the side, rear and near
field amplify both problems. Specific visible symptoms, evidence, rejected fixes
and prohibited uses are in
[`failure-cases-and-limits.md`](failure-cases-and-limits.md).

Priorities from here:

1. Obtain at least two valid native FTheta/rolling observations of the object, or
   a small real collection from the target rig, for a strict
   cross-camera/target-view holdout;
2. move dynamic objects to an object layer with explicit instance occupancy,
   visibility, unknown surface, view-dependent appearance and depth ordering,
   rather than relying on a single rigid body, monocular texture or static
   expected depth;
3. rebuild a static background that genuinely does not absorb vehicles or persons,
   using multi-camera dynamic masks, and keep the dual acceptance of independent
   LiDAR plus temporal holdout;
4. emit explicit uncertainty and invalid regions for sky, tree line and near-field
   edges — prefer marking something unknown over hiding an error with generative
   inpainting;
5. replace the engineering rig once real intrinsics, extrinsics, distortion,
   rolling shutter, exposure and ISP for the target vehicle are available, and
   re-run the full acceptance suite;
6. fix training/rendering wall-clock, peak VRAM, hardware and input hashes into a
   unified runner, so that resource figures are comparable across later
   reproductions.

**Overall judgement:** the pipeline is connected end to end and has delivered the
target L4 seven-view re-render. The static world and the protocol layer are usable
as an engineering demonstration; dynamic novel-view veracity, sensor veracity and
safety-critical usability do not yet meet a deliverable ground-truth standard.
