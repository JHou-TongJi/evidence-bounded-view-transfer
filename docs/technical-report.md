# Technical Report: Passenger-Car Road Data to L4 Delivery-Vehicle Seven-View Reconstruction

**Team:** Perception Data Reconstruction Team
**Members:** Zhong Ruizhi, Chen Silei

## 1. Task definition and delivery scope

Given the multi-camera images, calibration, vehicle trajectory and auxiliary
LiDAR / object tracks of an NCore V4 passenger-car road clip, this project
generates a seven-view visual sequence for a virtual L4 low-speed autonomous
delivery vehicle, in the same NCore world frame and at the same reference times.

The fixed input sample is chunk 0 of NCore clip
`7fd4d554-b155-45c6-a46a-646486029d85`. The output is seven RGB streams, each
1920×1080, 299 frames at 30 FPS, for a sequence duration of 9.9667 s. Delivery
consists of seven MP4s, one synchronised seven-view mosaic, and a set of
representative frames.

## 2. Input data and coordinate relationships

The input is a raw NCore V4 road collection: RGB from seven FTheta cameras,
per-frame exposure start and end times, camera mounting extrinsics, the rig's
world pose, the sparse `lidar_top_360fov` LiDAR, and cuboid tracks with
conservative dynamic-instance masks. The source cameras are distorted FTheta and
may carry a rolling shutter; the target vehicle output is an engineering-defined
rectified pinhole with a global-shutter midpoint proxy.

NCore uses the `T_source_target` naming convention: the transform carries a point
in the source frame into the target frame. For a target camera:

```text
p_rig   = T_camera_rig · p_camera
p_world = T_rig_world(t) · p_rig
T_camera_world(t) = T_rig_world(t) · T_camera_rig
```

The renderer then takes the inverse to obtain the OpenCV world-to-camera view
matrix. The virtual L4 vehicle follows the source collection vehicle's trajectory
and world localisation at each instant; along that same trajectory, the passenger
car's sensor layout is replaced by the target vehicle's seven-view camera layout.
The road scene therefore stays in the same world frame; what changes is the
mounting position, orientation and imaging parameters of the cameras relative to
the vehicle body.

## 3. Main reconstruction and generation pipeline

```text
NCore 7V RGB / pose / LiDAR / mask
  ├─ FTheta -> 20 tile virtual-pinhole proxy
  ├─ Static 2D Gaussian Splatting (5 cameras, all 299 frames)
  │    └─ Road LiDAR depth supervision + conservative dynamic mask excluding RGB supervision
  ├─ Seven-camera multi-timestamp sky semantic fusion -> world-direction temporal cubemap
  ├─ Registry of rigid 2DGS actor experts passing the source-camera temporal holdout
  └─ Target L4 seven-view direct 1920x1080 surfel rasterisation
       ├─ Static 2DGS RGB
       ├─ Temporal sky sampled behind alpha
       └─ At most one actor expert overlaid, only when the view / distance / time gates pass
```

### 3.1 Static background: surface-aligned 2DGS

The final static background model uses 5 context cameras, 20 virtual-pinhole
tiles, 480×270 proxy training input and 12,000 iterations of the official
[**2D Gaussian Splatting**](https://github.com/hbb1/2d-gaussian-splatting)
implementation. 2DGS constrains Gaussians to local surface discs rather than free
volumetric splats, which mitigates the thick slabs, streaking and distant
floaters that plain 3DGS produces at roads, kerbs, tree lines and building edges.

During training, a source-priority dynamic-instance mask excludes **dynamic
target** RGB supervision, and the road region uses **real LiDAR depth** as a
physical scale constraint. The monocular depth model
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), after
LiDAR scale calibration, supplies a low-weight depth prior on the road region,
constraining the static surface alongside the real LiDAR supervision. The final
1080p images are rasterised directly from the trained 2DGS model through the
target pinhole camera parameters, which raises sampling density and softens edge
aliasing; dynamic targets and insufficiently observed high-frequency geometry
remain limited by the original training observations.

### 3.2 Sky: world-direction temporal cubemap

A world-frame cubemap is built from 7 source cameras, 18 time samples within the
chunk, and a local SegFormer sky mask. At render time the sky at the same
timestamp is sampled according to the target pixel's world direction, and
composited as:

$$
\Large C_{final} = C_{2DGS} + (1 - \alpha_{2DGS}) \cdot C_{sky}
$$

The sky fills only background that 2DGS does not cover; it does not modify the
2DGS alpha, the road surface or actor geometry. Invalid cubemap directions fall
back to black, so cubemap coverage and seams remain recorded as a known risk.

### 3.3 Dynamic rigid objects: camera-gated actor experts

No attempt is made to force complete novel-view generation for vehicles and work
equipment. Each candidate actor is first evaluated on held-out frames of its
source camera for masked RGB MAE, alpha IoU and area ratio; only the 6 experts
meeting the thresholds enter the registry. In a target view, at most one expert
is selected for a given actor, and only when: the viewing-direction angle is no
more than 25°, the distance ratio to the camera lies in 0.60–1.60, and the time
falls inside its source observation window, with a 100 ms fade in and out at the
boundaries. At every other instant the renderer falls back strictly to the static
background.

The purpose of this fail-closed policy is to avoid compositing a monocular or
cross-view-unreliable actor into a ghost image. Actors are ordered among
themselves by local surface depth. The static background's expected depth is
*not* used as hard occlusion ground truth, because the static background may
itself have baked in dynamic targets; experiment showed that such a hard gate
produces grey holes.

## 4. The target L4 vehicle and its seven-camera definition

The target vehicle is specified as an **L4 low-speed autonomous delivery
vehicle**, with body dimensions, payload capacity and application scenarios
referencing the [public Neolix X3 product material](https://www.neolix.cn/productTechnology).
From this the project builds an engineering target vehicle model whose size class
and sensor layout suit industrial parks, campuses, residential developments and
last-mile delivery roads; the camera intrinsics and extrinsics are virtual
calibration parameters defined for the reconstruction experiment. The vehicle is
assumed to be 2.69 m long, 1.08 m wide and 1.82 m high, with a 1.73 m wheelbase,
520 kg payload and 3.0 m³ cargo volume.

The target ego frame is right-handed: `+x` forward, `+y` left, `+z` up, with the
origin at the ground projection of the rear-axle midpoint. Cameras use the OpenCV
frame: `+x` right, `+y` down, `+z` forward. Every camera is a rectified pinhole
producing 1920×1080 at 30 FPS, with a 100° horizontal and 67.673° vertical field
of view, and intrinsics `fx = fy = 805.5356 px, cx = 959.5 px, cy = 539.5 px`.

| Camera | Position, ego (m) `(x,y,z)` | yaw / pitch / roll (°) | Purpose |
|---|---|---:|---|
| front_center | (2.20, 0.00, 1.72) | (0, −4, 0) | Forward road main view |
| front_left | (2.06, 0.46, 1.68) | (50, −6, 0) | Front-left overlap region |
| front_right | (2.06, −0.46, 1.68) | (−50, −6, 0) | Front-right overlap region |
| side_left | (1.10, 0.53, 1.66) | (100, −8, 0) | Left near field |
| side_right | (1.10, −0.53, 1.66) | (−100, −8, 0) | Right near field |
| rear_left | (−0.30, 0.44, 1.62) | (150, −7, 0) | Rear-left coverage |
| rear_right | (−0.30, −0.44, 1.62) | (−150, −7, 0) | Rear-right coverage |

The complete 4×4 mounting matrices, the on-board LiDAR metadata and the scope of
the calibration are in [`configs/7fd4_neolix_x3_size_7v.json`](../configs/7fd4_neolix_x3_size_7v.json).

## 5. Interpreting the results, and their boundaries

The static background, the world-direction sky and the continuous target poses
are the principal trustworthy components of this result. A rigid actor expert
represents only "a local dynamic augmentation that was validated in the source
view and is compatible with the target-view gates". Pedestrians, vehicles that
failed holdout acceptance, back faces and occluded surfaces do not receive
trustworthy geometry; they may appear as residue, blur or absence within the
static background.

This project serves as an engineering validation sample for cross-vehicle
seven-camera layout design, static road-environment reconstruction, continuous
novel-view generation and conservative dynamic-target augmentation. For the
completeness of dynamic targets, their occlusion relationships and sensor-level
veracity, further evaluation against real collected data from the target vehicle,
or stricter cross-view validation, is still recommended.
