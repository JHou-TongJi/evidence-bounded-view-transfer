# Failure Cases and Limits of Use

## 1. Purpose and the criteria applied

This final result keeps its failures visible. It does not use generative
inpainting, frame interpolation or post-processing to disguise unknown regions as
real observations. The analysis below rests on the final single-camera 1080p
output, the seven-view mosaic, the road LiDAR holdout, the actor source-time
holdout, and the historical A/B experiments.

Three different senses of "passing" must be kept apart:

- **Pipeline pass** — the program completes training/rendering, the output format
  is correct and contains no NaNs. This says nothing about image quality.
- **Proxy-metric pass** — a threshold is met within an explicitly stated domain:
  a named source camera, the valid LiDAR pixels, or the compositing-safety
  domain. It does not extrapolate automatically to a novel target L4 view.
- **Target-view veracity pass** — requires independent real collection from the
  target rig, or equivalent high-confidence ground truth. This project does not
  have that evidence.

The final result can therefore only be described as "completed an engineering
re-render of the L4 seven-view rig and passed layered proxy checks". It must not
be described as "recovered the real dynamic seven-view world".

## 2. Smearing, stretching and static leakage on near-field vehicles

### Visible symptom

Nearby work vehicles, parked vehicles and occlusion boundaries show thick slabs,
duplicated outlines, stretched texture, bright edges, or vehicles that appear
"fixed onto the background". The problem is more pronounced in the front-right,
side-right and rear views. It is directly observable in the right-hand column of
the source/target comparison.

> The source-versus-target comparison still
> (`source_vehicle_vs_target_l4_frame270.png`) is withheld from this repository
> under §4 of [`data-compliance.md`](data-compliance.md). The same comparison is
> visible in [`demo/camera_comparison_640x360_demo.mp4`](../demo).

### Cause

Although the static 2DGS excludes the corresponding RGB loss via dynamic masks,
the initial geometry and the multi-timestamp reconstruction can still absorb
dynamic objects or the background around them. There is substantial translation,
rotation and near-field parallax between the target and source cameras, and a
single-view rigid actor carries no information about unobserved sides, backs or
real occlusions. Raising the output to 1920×1080 only increases sampling density;
it adds no surface evidence.

### Fixes tried and not adopted

| Attempt | Result and conclusion |
|---|---|
| clean-static v3 + exact-FTheta actor | 31.79% ROI improvement over 67 frames, 2.07% full-image improvement, but temporal delta worsened by 19.98% — failed sequence acceptance |
| multi-camera clean-static v4 | Dynamic-mask coverage and static/dynamic separation improved, but static ROI MAE worsened; old actors also degraded when composited onto the new background |
| Hard occlusion by static expected depth | The static layer itself contains dynamic leakage, so it wrongly rejects actors into grey holes; this hard gate is not enabled in the final |
| side-right single-camera near-vehicle expert | Passed the source-time holdout, but covered only 7 frames on the target side and produced bright patches, thick edges and wrong parallax — not admitted to the registry |
| Direct 1080p re-render | Finer edge sampling, but the missing near-vehicle surface is unchanged; not evidence of any geometric improvement |

### Direction for improvement

First obtain at least two valid observations of the object, or a small real
collection from the target vehicle, then train an object layer with explicit
occupancy/visibility, unknown surface, view-dependent appearance and depth
ordering. Continuing to rely on monocular texture, loosening the mask/LiDAR
gates, simply adding iterations, or masking missing surfaces with generative
colour fill are all ruled out.

## 3. Cross-view and temporal incompleteness of rigid dynamic actors

### Visible symptom

Only 6 rigid experts, over 4 tracks, entered the registry. When a target does not
satisfy the source direction, distance or time window, the renderer falls back to
the static background. This can produce vehicles that flicker in and out, edges
that vary in thickness, a drop in alpha IoU at the moment of entry or exit, or
uncovered vehicles left as static ghosts.

In the final statistics, the adjacent alpha IoU P05 is 0 for side-left, rear-left
and rear-right; the rear-right mean is only 0.479380. These figures pool normal
frustum entry and exit with insufficient source coverage and gate switching. A
camera with a higher mean must therefore not be read as having correct dynamic
geometry, and a P05 of 0 must not be read on its own as a rendering collapse.

### Cause

The source-time holdout validates an expert only at held-out instants on its own
source camera. A direction gate of ≤25°, a distance ratio of 0.60–1.60 and a
source time window can reject obvious extrapolation, but they cannot synthesise
an unseen body surface, and they cannot prove that occlusion ordering and
appearance are correct in the target camera.

### Historical evidence

- InstantNuRec dynamic NPZ improved MAE by 43.9% in the changed dynamic region of
  a representative frame, but only 1.68% full-image and 0.30% temporally;
  nearest-timestamp and actor-nearest selection were unstable across frames.
- The exact-FTheta canonical actor worsened the 67-frame temporal residual by
  30.65%; learning only a pose correction worsened it by 35.77% instead.
- Street Gaussians v7 gave a four-frame exact MAE of 0.06893, still worse than
  the same frames' static 0.06094, with large-area actor/static alpha overlap.
- The source anchor gave raw/dense coverage of only 0.272%/0.247% on the front
  centre at target L4 frame 15, and at most 4.191% on the right — evidence that
  source masks and LiDAR cannot directly complete a novel-view vehicle.

### Direction for improvement

Make "does the actor exist, which surface is visible, what is in front, which
regions are unknown" an explicit learnable and auditable state. Target-side
acceptance should include at minimum a cross-physical-camera temporal holdout,
silhouette IoU, area ratio, visible depth ordering, and a continuous-window
temporal residual. The existing registry may continue as a fail-closed baseline,
but the angle and distance thresholds must not be widened in pursuit of a higher
active-frame count.

## 4. Pedestrians and deformable targets are not recovered

### Visible symptom

Pedestrians may be diluted into the static background, blurred, discontinuous or
missing entirely. No reliable human trajectory, pose, size, orientation or
occlusion relationship may be inferred from the output RGB.

### Cause and work done

Pedestrians are not rigid canonical actors. The project completed a native
FTheta/rolling multi-person data bundle, per-track sampling and smoke tests, but
the rigid actor registry explicitly excludes deformable persons. The EmerNeRF
vehicle v2 short window gave a positive signal, but the longer v3 window was
negative; v5's visible-alpha IoU was only 0.1015, below the gate used for the
leading vehicle. No formal person pilot was therefore started, and no person was
written into the final output.

### Direction for improvement

Use a time-varying or deformable representation, instance-level visibility, and
cross-camera silhouette/depth ordering supervision. Run person experiments only
after the vehicle object layer passes the same strict holdout. Treating a
SAM2-only mask as real geometry, or optimising a person as a rigid body, are
ruled out.

## 5. Side, rear, sky and low-coverage regions

### Visible symptom

Side and rear views show blur, opaque low-quality surfaces, black patches, bright
fill or low-texture regions where the source view is occluded, at near-field
edges, and at tree-line and sky boundaries. The two solid black cells at the
centre and bottom-centre of the mosaic are only 3×3 layout placeholders; black,
grey or blurred regions *inside* a camera tile are actual output and should be
treated as low-confidence or unknown.

### Cause

The 20-tile, 5-camera context improved surround coverage but did not eliminate
cross-timestamp dynamic contamination or unobserved surfaces. The temporal sky
has only 18 time slots and is composited only behind `1 − alpha`; it cannot cover
a tree line, vehicle or building that has wrongly high alpha. Road LiDAR is
sparse supervision, and the final holdout P90 is still 1.997339 m, so tail
geometry error is not negligible.

### Verified boundaries

- The 11-tile multi-scan 12k holdout MAE/P90 was 1.10098/2.00667 m; the 20-tile
  final is 1.06787/1.99734 m. The improvement is mainly near-field coverage, not
  omnidirectional accuracy.
- The 20-tile version reduced the black radial banding of earlier versions, but
  the low-alpha and sky-valid backfill proportion rose, so bright blurred holes
  appear locally.
- The LiDAR-anchored weak prior from Depth Anything V2 offered only about a 3.1%
  candidate improvement in road MAE, with near-side dynamic smearing unchanged,
  so it did not replace the final 12k model.
- A road-surface curvature constraint improved MAE by only 0.23%, below the
  threshold that would justify re-rendering all seven views.

### Direction for improvement

Add real side and rear static observations and near-field LiDAR coverage, and
emit an explicit uncertainty/invalid mask at reconstruction time. Add a
reliable-coverage audit and seam optimisation to the cubemap, while still
permitting only transparent sky to be filled. Covering wrongly high-alpha
geometry with sky or propagated RGB is ruled out.

## 6. Observability ceiling of correspondence, masks and sparse LiDAR

### Measured limits

- Conservative raw RGB-LiDAR densification added only 1 Gaussian, with observed
  surface at 2.81% and unknown at 97.19%; loosening the threshold only pulls
  background points into the vehicle.
- RoMa accepted 0 candidates in both native FTheta/rolling triangulation audits —
  the forward short baseline and the 1.345 m lateral baseline. The candidate rays
  have angle, but the closest-distance and cuboid constraints do not hold.
- SAM2 can increase temporal mask coverage, but the cross-view audit shows some
  new masks are occluded by static depth or lack independent camera support. It
  can serve only as a supervision sidecar; it cannot write actors or generate
  target imagery directly.
- The raw LiDAR object layer shows that an "already observed face" can be
  validated, but cuboid surface-probe coverage is only 8.57% — not enough to
  reconstruct a complete vehicle.

### Direction for improvement

A dedicated multi-view dynamic correspondence/reconstruction model, or additional
real observations, are required — not further lowering of the triangulation,
LiDAR 3D gate or multi-view support thresholds. Unknown surfaces should stay
transparent and carry a confidence value with the output.

## 7. Imaging model and target rig assumptions

The source NCore cameras are FTheta with rolling-shutter start/end exposure
poses; the final L4 output is a rectified pinhole with a global-shutter midpoint
proxy. The project does not simulate real target lens distortion, motion blur,
exposure control, colour response, noise, the compression chain or the ISP.

The target seven-view extrinsics and the vehicle dimensions are an X3-size-class
engineering assumption, not an OEM calibration from Neolix or any other
manufacturer. Even where scene geometry is correct, the output cannot substitute
for a real vehicle's raw camera stream. Once real intrinsics, extrinsics and an
imaging model for the target vehicle become available, the rig JSON should be
replaced and coverage, road LiDAR, time synchronisation, target-view and
sensor-domain acceptance all re-run.

## 8. Limits on interpreting the metrics

| Metric | What it establishes | What it does not establish |
|---|---|---|
| Video frame count / resolution / FPS | Output protocol and batch completeness | That the imagery is correct, or that synchronisation error is zero |
| Road LiDAR MAE / median / P90 | Metric-scale error on valid LiDAR road pixels | Depth over all seven-view pixels; vehicle or person geometry |
| Source masked RGB MAE | Texture error within held-out frames of the source camera | Appearance in the novel target L4 view |
| Source alpha IoU / area ratio | Silhouette dilation or contraction in the source view | Occluded back faces; silhouette in the target view |
| Outside-actor RGB MAE | That the compositor did not visibly pollute static RGB outside the actor | That the actor itself, or the background itself, is correct |
| Adjacent alpha IoU | A proxy for silhouette continuity | Dynamic ground truth; correct object identity and 3-D motion |

A subjective impression that the output "looks clearer" cannot substitute for
these domain-specific metrics, and 1080p must not be presented as new geometric
evidence. The absence of per-pixel real ground truth in the target view is the
fundamental boundary of the current evaluation.

## 9. Prohibited and permitted uses

### Prohibited

- Autonomous-driving safety certification, road compliance testing and
  collision-distance measurement;
- ground-truth annotation of dynamic objects or pedestrians, trajectory ground
  truth, or occlusion ground truth;
- direct use as perception training ground truth or as closed-loop driving-policy
  training data;
- claiming the output as a real calibration, real sensor stream or real vehicle
  collection from any OEM model;
- redefining generatively inpainted, sharpened or frame-interpolated derivative
  video as training ground truth.

### Permitted

- Engineering demonstration of cross-vehicle camera layout and synchronised
  seven-view video;
- visualisation of the static scene, road scale and target rig coverage;
- integration testing of the rendering, encoding and batch-processing interfaces;
- a safe baseline for subsequent dynamic architectures with explicit visibility,
  view-dependent appearance, deformation and uncertainty.

**Final judgement:** the current pipeline *can* be integrated and *can* complete
a seven-view re-render for the target L4 vehicle, but only at the level of the
static world, protocol completeness and a conservative local rigid augmentation
layer. Full dynamic veracity, target sensor veracity and safety-critical
usability have all not been established.
