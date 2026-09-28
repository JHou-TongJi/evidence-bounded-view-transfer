# Source Vehicle Sensor Notes

## 1. Overview of the source data

The source data for this project comes from NVIDIA's
[PhysicalAI Autonomous Vehicles NCore dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).
The current experiment uses clip:

```text
7fd4d554-b155-45c6-a46a-646486029d85
```

The source is a passenger-car / autonomous-driving collection rig. The data
contains multi-camera RGB of road scenes, camera calibration, exposure times,
vehicle world poses, a roof-mounted LiDAR, and dynamic cuboid tracks — enough to
construct a cross-vehicle novel-view rendering task.

## 2. Source sensor suite

| Sensor type | Typical device / data content | Role in this project |
|---|---|---|
| 7 road cameras | FTheta RGB images, intrinsics, mounting extrinsics, exposure start/end times | Static scene training, sky construction, dynamic-target supervision, source-view validation |
| Roof LiDAR | `lidar_top_360fov` sparse point cloud | Road-surface scale constraint, depth audit, dynamic-target observability analysis |
| Vehicle pose graph | `T_rig_world(t)` | Aligning source cameras, LiDAR and the target vehicle rig into one world frame |
| Cuboid tracks | Spatio-temporal bounding boxes for vehicles, work equipment, pedestrians, etc. | Dynamic-instance selection, actor supervision, time-window definition |
| Dynamic-instance / semantic masks | Raw masks and conservatively propagated masks | Excluding dynamic RGB during static training; building dynamic-target supervision and held-out evaluation |

## 3. Source camera views and imaging model

The current NCore clip uses several road cameras covering forward, forward
telephoto, left and right cross, rear and rear telephoto directions. The camera
IDs directly relevant to this experiment are:

| Source camera ID | Field of view in the name | Primary use |
|---|---|---|
| `camera_front_wide_120fov` | 120° | Forward road main view; temporal reference camera |
| `camera_front_tele_30fov` | 30° | Forward long-range detail |
| `camera_cross_left_120fov` | 120° | Left crossing view |
| `camera_cross_right_120fov` | 120° | Right crossing view |
| `camera_rear_left_70fov` | 70° | Rear-left coverage |
| `camera_rear_right_70fov` | 70° | Rear-right coverage |
| `camera_rear_tele_30fov` | 30° | Rear long-range detail |

The source cameras use the FTheta imaging model defined by the NCore calibration.
That model stores the polynomial relationship between pixel radius and incidence
angle, the principal point, the maximum visible angle and linear correction
parameters, so it cannot simply be equated with an ordinary pinhole camera.

For `camera_front_wide_120fov`, this project reads the NCore raw calibration
directly and uses it for exact source-view validation:

| Item | Parameter |
|---|---|
| Native image resolution | 1920 × 1080 |
| Camera model | NCore FTheta |
| Nominal field of view | 120° |
| Shutter type | `ROLLING_TOP_TO_BOTTOM` |
| Example mounting position | `(2.010, −0.061, 1.610) m`, relative to the source vehicle rig |
| Exposure poses | Rig pose solved separately at each frame's START and END times |
| Source-view validation | FTheta rays computed per scan-line pose and aligned against the exported static Gaussian scene |

## 4. Source coordinate frames and transforms

NCore transforms follow the `T_source_target` naming convention, meaning a
transform that carries a point in the source frame into the target frame. For any
source camera:

$$
\Large p_{rig} = T_{camera\_rig} \cdot p_{camera}
$$
$$
\Large p_{world} = T_{rig\_world}(t) \cdot p_{rig}
$$

and therefore:

$$
\Large T_{camera\_world}(t) = T_{rig\_world}(t) \cdot T_{camera\_rig}
$$

where:

- `camera` — the source camera's OpenCV frame: `+x` right, `+y` down, `+z` forward;
- `rig` — the source collection vehicle's frame;
- `world` — the NCore global road-scene frame;
- `T_camera_rig` — the source camera's mounting extrinsics;
- `T_rig_world(t)` — the source vehicle's world pose at time `t`.

The source LiDAR uses the vehicle-frame convention:

```text
+x: vehicle forward
+y: vehicle left
+z: vehicle up
```

LiDAR points are first carried into `world` by the vehicle pose at the
corresponding time, then projected into the target view by the target camera's
world-to-camera transform.

## 5. Differences between source and target cameras

| Comparison | Source passenger-car rig | Target L4 delivery rig |
|---|---|---|
| Vehicle class | Passenger car / autonomous collection vehicle | Low-speed L4 autonomous delivery vehicle |
| Camera model | Native FTheta; some cameras rolling-shutter | Rectified pinhole; global-shutter midpoint proxy |
| Forward main camera FOV | 120° FTheta wide angle | 100° pinhole wide angle |
| Main camera mounting height | Front wide, approx. 1.61 m | Front centre, 1.72 m |
| Camera layout | Front, cross and rear combination oriented to source-vehicle road collection | Seven-view surround layout: front centre, front left/right, side left/right, rear left/right |
| Output resolution | Defined by each source camera's native NCore calibration | Uniform 1920 × 1080 |
| Trajectory | The source vehicle's real collection trajectory | The same world trajectory, re-observing the scene through the target camera extrinsics |

In the actual pipeline, the source-side FTheta and rolling-shutter calibrations
are used to read the raw observations, build the sky, and supervise static and
dynamic reconstruction; the final seven-view L4 images are obtained by
rasterising the reconstructed scene directly through the target vehicle's pinhole
camera parameters.

## 6. Boundaries of use

The cameras, LiDAR and trajectories in the source data supply real road
observations and geometric constraints for cross-vehicle reconstruction, but the
source contains no real collected imagery from the target delivery vehicle. The
target seven-view sequence therefore represents a virtual-view result generated
from a shared world frame, the source-side multi-modal observations, and the
target rig definition. The target vehicle's real lens distortion, exposure, ISP,
sensor noise and dynamic-target completeness still require further evaluation
against a real-vehicle calibration or additional validation data.
