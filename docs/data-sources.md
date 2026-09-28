# Data Source Notes

## 1. Dataset origin, version and access

This project uses NVIDIA's
[PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)
dataset — a reconstruction-ready subset selected from the large-scale
PhysicalAI-Autonomous-Vehicles raw autonomous-driving data and converted to the
standard **NCore V4** format.

| Item | Description |
|---|---|
| Dataset name | PhysicalAI-Autonomous-Vehicles-NCore |
| Publisher | NVIDIA Corporation |
| Data format version | NCore V4 |
| Access channel | Hugging Face dataset page |
| Openness | The dataset page is publicly viewable; downloading files requires signing in and accepting the NVIDIA dataset licence agreement |
| Dataset scale | The dataset card describes roughly 1.1k NCore clips; the source dataset contains 306,152 multi-sensor driving clips and approximately 1,700 hours of collected data |
| Scope actually used here | This submission uses only chunk 0 of `7fd4d554-b155-45c6-a46a-646486029d85` |
| Scene in this sample | Daytime, low speed, overcast, suburban / residential road |

The raw sequence file used for the final result is
`pai_7fd4d554-b155-45c6-a46a-646486029d85.json` together with its per-sensor
component stores, listed in §3 below.

## 2. Volume of data used

This project distinguishes raw observations, training/construction inputs,
validation inputs and final presentation outputs, so that they are not conflated
into a single data category.

| Data category | Scale | Use |
|---|---|---|
| Raw NCore scene | 1 clip, 1 chunk | The only source scene for this final result |
| Source RGB cameras | 7 cameras, 299 reference timestamps each | Static scene training, sky construction, dynamic-instance supervision, source-view evaluation |
| Roof LiDAR | 199 `lidar_top_360fov` sweeps | Road scale constraint, depth holdout, dynamic-object observability audit |
| Static scene context | 5 physical source cameras, 20 virtual-pinhole tile directions, 299 time references | Training the final static 2DGS background |
| Sky observations | 7 cameras × 18 chunk time samples | Building the world-direction temporal sky cubemap |
| Rigid actor experts | 6 experts passing the source-camera temporal holdout threshold, covering tracks 17/18/42/67 | Conservative dynamic augmentation of the target views |
| Final target output | 7 target cameras × 299 frames = 2,093 frames | 1920×1080, 30 FPS, 9.9667 s seven-view L4 demonstration |
| Final submission media | 7 single-camera MP4s, 1 synchronised seven-view mosaic, 10 representative frames | Presentation and review; the full RGB PNG sequence is not copied |

The final videos are encoded from the 2,093 rendered target-view RGB frames, but
the submission package retains only the MP4s and representative frames, to keep
the size down and to limit repeated distribution of derived imagery.

## 3. NCore V4 data structure

Each NCore clip consists of a sequence JSON and several sensor storage packages.
The dataset card gives the typical structure as:

```text
clips/<clip_uuid>/
├── pai_<clip_uuid>.json
├── pai_<clip_uuid>.ncore4.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_cross_left_120fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_cross_right_120fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_front_tele_30fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_front_wide_120fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_rear_left_70fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_rear_right_70fov.zarr.itar
├── pai_<clip_uuid>.ncore4-camera_rear_tele_30fov.zarr.itar
└── pai_<clip_uuid>.ncore4-lidar_top_360fov.zarr.itar
```

### 3.1 Sequence JSON

`pai_<clip_uuid>.json` is the data entry point. It declares the sensor component
stores, the time range, the coordinate-transform graph and the data schema. The
project uses it to locate camera images, LiDAR, offline calibration, ego pose and
cuboid tracks.

### 3.2 Camera data and calibration

Each camera store contains:

- RGB image frames;
- frame exposure start and end timestamps;
- camera intrinsics, including the FTheta distortion model, polynomial
  parameters, principal point, maximum field of view and linear correction terms;
- the camera's mounting extrinsics relative to the rig, `T_sensor_rig`;
- auxiliary information such as valid-region and ego masks;
- rolling-shutter scan-direction information.

The source cameras use the FTheta model, and some carry a rolling shutter. A raw
image therefore cannot be treated as a pinhole projection; the project retains
the native NCore camera model when reading raw observations, building the sky and
performing geometric audits.

### 3.3 LiDAR, trajectory and dynamic labels

The roof LiDAR data provides sparse three-dimensional measurements of the road
and the neighbourhood of dynamic targets. The fields used are the LiDAR
timestamp, sensor origin, ray direction and range, together with their transforms
to the rig and world frames.

NCore additionally provides:

- offline ego motion, i.e. `T_rig_world(t)`;
- offline intrinsics and extrinsics for cameras and LiDAR;
- cuboid labels — three-dimensional bounding boxes, classes, track IDs and
  temporal association for dynamic objects;
- scene metadata such as camera and sensor masks.

Together these allow the multiple source observations to be aligned into a single
NCore world frame, and the reconstructed scene to be re-observed from the target
L4 camera poses.

## 4. How the data is used

### 4.1 Static background training

The final static background is trained with the official 2D Gaussian Splatting
implementation. Native NCore FTheta images are first resampled into several
virtual-pinhole tiles, and the source camera poses form the training views.

Training uses:

- RGB from 5 physical source cameras;
- NCore camera calibration, vehicle pose and timestamps;
- conservative dynamic-instance masks, to exclude dynamic-target regions from
  static RGB supervision;
- raw LiDAR road depth, for road scale and surface-depth constraints;
- a weak Depth Anything V2 road prior after LiDAR scale calibration.

Depth Anything V2 is a pre-trained monocular depth model, not newly collected
data added by this project. It supplies only a low-weight prior on valid static
road regions; real LiDAR remains the principal physical depth constraint.

### 4.2 Sky asset construction

The sky module uses 7 source cameras, 18 time samples and a local SegFormer sky
segmentation to build a world-direction temporal cubemap. That asset supplies sky
colour only at pixels the static 2DGS alpha does not cover; it takes no part in
the geometric training of the road or of dynamic targets.

### 4.3 Dynamic actor training and selection

Actor experts for rigid vehicles and work equipment are trained and filtered
using raw instance masks, cuboid tracks, source camera RGB and a temporal-holdout
strategy. Each expert must pass thresholds on the source camera's held-out time
interval for:

- in-mask RGB MAE;
- alpha IoU;
- alpha area ratio.

Dynamic objects failing these thresholds do not enter the final registry.

Pedestrians, and dynamic objects that could not pass cross-time or cross-camera
validation, are not written into the final output as trusted actors.

### 4.4 Validation and evaluation

The roles of training, validation and presentation data are as follows:

| Category | Content used | Purpose |
|---|---|---|
| Static training | 5 source cameras, training time references, static-region masks | Optimising the 2DGS static surface |
| Static validation | Independent proxy views, road LiDAR holdout | Computing road-depth MAE, median and P90 |
| Actor validation | Source-camera held-out frames and instance masks | Selecting reliable actor experts |
| Compositing safety audit | Final 1080p actor-composite RGB, actor alpha, static RGB | Verifying that the background outside an actor is not polluted by the compositor |
| Presentation data | Seven-view MP4s, mosaic, representative frames | Human inspection and submission review |
| Target-view ground truth | None — no real collected target-L4 imagery exists | Not used for training; not used for pixel-level veracity scoring |

## 5. Final submission samples and traceability

Every target camera frame can be traced through
[`evidence/frame_mapping.csv`](../evidence/frame_mapping.csv) — 2,093 rows, one
per delivered frame — to:

- the target camera ID;
- the output frame index and its time in the sequence;
- the corresponding NCore source reference camera and reference frame index;
- the relative path of the delivered RGB.

The rig configuration, the static 2DGS checkpoint, the temporal sky asset and
the actor expert registry are identified once per run rather than per frame, and
are recorded in
[`evidence/static_2dgs_1080p_manifest.json`](../evidence/static_2dgs_1080p_manifest.json),
[`evidence/actor_composite_1080p_manifest.json`](../evidence/actor_composite_1080p_manifest.json)
and
[`evidence/accepted_actor_registry.json`](../evidence/accepted_actor_registry.json).
Together with the road LiDAR holdout report and
[`evidence/checksums.sha256`](../evidence/checksums.sha256), these constitute the
traceability evidence for the results.

## 6. Data limitations

1. Only one chunk of one NCore clip is used. Scene coverage is limited and cannot
   represent all roads, weather, urban forms and traffic densities.
2. The static 2DGS mainly recovers road, buildings, tree lines, kerbs and static
   vehicle background; near-field dynamic vehicles, pedestrians and occluded
   surfaces may still appear blurred, leak static content, or be missing.
3. The target L4 cameras form an engineering-defined rectified pinhole rig. They
   carry no OEM calibration, lens distortion, exposure response, ISP or rolling
   shutter from a real target vehicle.
4. The final seven-view output has no corresponding real collected RGB ground
   truth from the target vehicle, so dynamic-object quality and cross-camera
   novel-view quality can only be assessed through source-view holdouts,
   geometric constraints and compositing-safety proxies.
5. Observations of the same dynamic object at different times and from different
   source cameras may be incomplete. This project uses fail-closed gates: it
   prefers to fall back to the static background rather than force the generation
   of an unvalidated dynamic surface.
