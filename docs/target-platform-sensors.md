# Target Platform Sensor Notes

## 1. Target vehicle and the basis for the configuration

The target vehicle for this project is defined as an **L4 low-speed autonomous
delivery vehicle**, addressed at low-speed operating scenarios such as industrial
parks, campuses, residential developments, logistics parks and urban last-mile
delivery roads. Vehicle dimensions and payload capacity follow the
[public Neolix X3 product material](https://www.neolix.cn/productTechnology);
combined with the cross-vehicle view-reconstruction requirements of this
experiment, they define the engineering target rig `neolix_x3_size_7v_v1`.

That rig defines the spatial layout of the target sensors and the views they
output. The camera intrinsics and extrinsics are engineering parameters for a
virtual reconstruction experiment; they do not represent any manufacturer's OEM
calibration.

| Item | Value |
|---|---|
| Vehicle class | L4 low-speed autonomous delivery vehicle |
| Body length | 2.69 m |
| Body width | 1.08 m |
| Body height | 1.82 m (excluding roof LiDAR) |
| Wheelbase | 1.73 m |
| Cargo volume | 3.0 m³ |
| Reference payload | 520 kg |
| Design top speed | 50 km/h |
| Typical operating scenarios | industrial park, campus, residential development, logistics park, urban last-mile delivery road |
| Sensor topology | 7 surround RGB cameras + 1 roof LiDAR (metadata only) |

## 2. Coordinate frames

The target vehicle uses the body frame `ego`:

```text
+x: vehicle forward
+y: vehicle left
+z: vehicle up
```

The `ego` origin is defined as the ground projection of the assumed rear-axle
midpoint. The target cameras use the OpenCV camera frame:

```text
+x: image right
+y: image down
+z: lens forward
```

The rigid transform from target camera to vehicle body is written:

$$
\Large p_{ego} = T_{camera\_rig} \cdot p_{camera}
$$

where $T_{camera\_rig}$ carries a point in camera coordinates into the target
vehicle `ego` frame. The target vehicle's world trajectory is aligned with the
rig pose of the source NCore collection vehicle at the same instant:

$$
\Large T_{camera\_world}(t) = T_{rig\_world}(t) \cdot T_{camera\_rig}
$$

The renderer uses the inverse as the OpenCV world-to-camera view matrix. The
virtual delivery vehicle therefore follows the source vehicle's road trajectory
while generating new views from the target vehicle's own camera mounting
positions, orientations and imaging parameters.

## 3. Target camera imaging model

All seven target cameras use a single rectified pinhole model, producing
undistorted RGB output.

| Item | Parameter |
|---|---|
| Camera count | 7 |
| Output resolution | 1920 × 1080 |
| Output frame rate | 30 FPS |
| Horizontal field of view | 100° |
| Vertical field of view | 67.673° |
| Focal length | `fx = fy = 805.5356 px` |
| Principal point | `cx = 959.5 px`, `cy = 539.5 px` |
| Shutter model | global-shutter midpoint proxy |
| Raw-lens reference | The automotive wide-angle capability range of the [Leopard Imaging LI-IMX390-GMSL2 datasheet](https://leopardimaging.com/wp-content/uploads/2023/12/LI-IMX390-GMSL2-xxxH_Datasheet_V1.4.pdf) |

The experiment outputs rectified pinhole images, so that projection and rendering
are uniform across target views. It does not simulate real lens distortion,
exposure, ISP, motion blur or rolling shutter.

## 4. Mounting parameters of the seven target cameras

Camera positions are given as `(x, y, z)` in metres in the `ego` frame;
`yaw / pitch / roll` are in degrees.

| Camera ID | Position `(x,y,z)` m | Mounting height | yaw / pitch / roll | H-FOV | Primary coverage |
|---|---|---|---|---|---|
| `front_center` | (2.20, 0.00, 1.72) | 1.72 m | (0°, −4°, 0°) | 100° | forward road, lane markings, distant targets |
| `front_left` | (2.06, 0.46, 1.68) | 1.68 m | (50°, −6°, 0°) | 100° | front-left road, left crossing region |
| `front_right` | (2.06, −0.46, 1.68) | 1.68 m | (−50°, −6°, 0°) | 100° | front-right road, right crossing region |
| `side_left` | (1.10, 0.53, 1.66) | 1.66 m | (100°, −8°, 0°) | 100° | left near field, kerb and parallel targets |
| `side_right` | (1.10, −0.53, 1.66) | 1.66 m | (−100°, −8°, 0°) | 100° | right near field, kerb and parallel targets |
| `rear_left` | (−0.30, 0.44, 1.62) | 1.62 m | (150°, −7°, 0°) | 100° | rear-left road and blind-spot coverage |
| `rear_right` | (−0.30, −0.44, 1.62) | 1.62 m | (−150°, −7°, 0°) | 100° | rear-right road and blind-spot coverage |

The transform for the front-centre camera, as an example:

$$
\Large T_{front\_center\_rig} =
\begin{bmatrix}
0.000000 & -0.069756 & 0.997564 & 2.200000 \\
-1.000000 & 0.000000 & 0.000000 & 0.000000 \\
0.000000 & -0.997564 & -0.069756 & 1.720000 \\
0.000000 & 0.000000 & 0.000000 & 1.000000
\end{bmatrix}
$$

The complete 4×4 `T_camera_rig` matrices for the other six cameras are in:

```text
configs/7fd4_neolix_x3_size_7v.json
```

## 5. Relationship to the source passenger-car sensors

Target camera mounting heights are 1.62–1.72 m. Taking this experiment's source
front wide-angle camera as the comparison, its NCore calibration puts it at
approximately 1.61 m; the target front-centre camera sits at 1.72 m, and the
front-side cameras are likewise distributed closer to the delivery vehicle's own
body boundary.

This project therefore does not merely change image orientation. It changes, at
the same time:

- camera position relative to the vehicle's front/rear and left/right boundaries;
- camera mounting height;
- the coverage relationship among front, side and rear surround views;
- the target camera's pinhole intrinsics and 100° field of view;
- output image resolution and the organisation of the camera array.

Together these constitute the cross-vehicle view transfer from a passenger-car
collection rig to the seven-view rig of a low-speed autonomous delivery vehicle.
