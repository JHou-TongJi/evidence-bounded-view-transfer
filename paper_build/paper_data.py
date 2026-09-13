#!/usr/bin/env python
"""Single source of truth for every number that appears in the manuscript.

Nothing in the content modules may hard-code a measured value; it must come from
here, and everything here is read from a file under H:/ARIS/papers/paper07/.
`provenance()` emits the audit trail used by paper-claim-audit.
"""
import collections
import csv
import json
import os
import re
import statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EV = os.path.join(ROOT, "05_评价结果", "evidence")
CFG = os.path.join(ROOT, "03_工程代码", "configs", "7fd4_neolix_x3_size_7v.json")
QCSV = os.path.join(ROOT, "05_评价结果", "quality_metrics.csv")

CAMS = ["front_center", "front_left", "front_right", "side_left",
        "side_right", "rear_left", "rear_right"]
NICE = {c: c.replace("_", "\u2005") for c in CAMS}
NFRAMES = 299
NCAM = 7
NTOTAL = NCAM * NFRAMES          # 2093


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


RIG = _load(CFG)
HOLD = _load(os.path.join(EV, "road_lidar_holdout_same_static_model.json"))
REG = _load(os.path.join(EV, "accepted_actor_registry.json"))
COMP = _load(os.path.join(EV, "actor_composite_1080p_manifest.json"))
STATIC = _load(os.path.join(EV, "static_2dgs_1080p_manifest.json"))
CAMJ = {c: _load(os.path.join(EV, c + ".json")) for c in CAMS}

with open(QCSV, encoding="utf-8") as f:
    QUALITY = list(csv.DictReader(f))


# --------------------------------------------------------------- rig
def rig_rows():
    """Target extrinsics table, straight from the config. Fixes prior-draft D1/D2."""
    out = []
    for c in RIG["cameras"]:
        m = c["mount"]
        x, y, z = m["position_ego_m"]
        yaw, pitch, roll = m["yaw_pitch_roll_deg"]
        out.append((c["camera_id"], x, y, z, yaw, pitch, roll,
                    m["horizontal_fov_deg"], m["vertical_fov_deg"]))
    return out


VEH = RIG["vehicle"]
INTR = RIG["intrinsics"]
HW = RIG["camera_hardware_reference"]

#: The only source-camera mounting NCore publishes for this clip, from
#: 02_数据说明/源车型传感器说明.md ("安装位置示例 (2.010, -0.061, 1.610) m,
#: 相对源车 rig"). The rig config sets the target ego frame equal to the source
#: rig frame at every timestamp, so source and target mountings are directly
#: comparable in one frame.
SRC_CAM = {"id": "camera_front_wide_120fov", "position_rig_m": (2.010, -0.061, 1.610)}


def platform_spec():
    """Target platform and sensor specification, straight from the rig config.

    Returns (group, item, value) rows. Every value is read from
    03_工程代码/configs/7fd4_neolix_x3_size_7v.json; nothing is transcribed.
    """
    c0 = RIG["cameras"][0]["mount"]
    out = RIG["output"]
    li = RIG["lidars"][0]
    cal = RIG["calibration_status"]
    return [
        ("Vehicle", "Class", VEH["type"]),
        ("Vehicle", "Envelope L x W x H", "%.2f x %.2f x %.2f m (excl. LiDAR)"
         % (VEH["length_m"], VEH["width_m"], VEH["height_without_lidar_m"])),
        ("Vehicle", "Wheelbase", "%.2f m" % VEH["wheelbase_m"]),
        ("Vehicle", "Rated payload", "%d kg" % VEH["payload_kg"]),
        ("Vehicle", "Cargo volume", "%.1f m3" % VEH["cargo_volume_m3"]),
        ("Vehicle", "Design top speed", "%d km/h" % VEH["maximum_design_speed_kph"]),
        ("Vehicle", "Operating scenarios", ", ".join(VEH["operating_scenarios"])),
        ("Camera rig", "Count / topology", "7 surround RGB, rectified pinhole"),
        ("Camera rig", "Output", "%d x %d at %d FPS" % (INTR["width"], INTR["height"], out["fps"])),
        ("Camera rig", "Field of view", "%.0f deg H / %.3f deg V"
         % (c0["horizontal_fov_deg"], c0["vertical_fov_deg"])),
        ("Camera rig", "Intrinsics", "fx = fy = %.4f px, (cx, cy) = (%.1f, %.1f)"
         % (INTR["fx"], INTR["cx"], INTR["cy"])),
        ("Camera rig", "Mounting height", "%.2f-%.2f m above ground"
         % (min(c["mount"]["position_ego_m"][2] for c in RIG["cameras"]),
            max(c["mount"]["position_ego_m"][2] for c in RIG["cameras"]))),
        ("Camera rig", "Shutter model", "global-shutter exposure-midpoint proxy"),
        ("Raw-lens ref.", "Sensor class", HW["model"]),
        ("Raw-lens ref.", "Native resolution", "%d x %d"
         % (HW["raw_sensor_resolution"][0], HW["raw_sensor_resolution"][1])),
        ("Raw-lens ref.", "Native lens FOV", "%.0f deg H / %.0f deg V"
         % (HW["raw_lens_horizontal_fov_deg"], HW["raw_lens_vertical_fov_deg"])),
        ("Raw-lens ref.", "Max native rate", "%d FPS" % HW["maximum_raw_frame_rate_fps"]),
        ("LiDAR", "Placement", "%s, (%.2f, %.2f, %.2f) m in ego frame"
         % (li["lidar_id"], li["position_ego_m"][0], li["position_ego_m"][1],
            li["position_ego_m"][2])),
        ("LiDAR", "Role", li["role"]),
        ("Provenance", "Vehicle dimensions", cal["vehicle_dimensions"]),
        ("Provenance", "Camera extrinsics", cal["camera_extrinsics"]),
        ("Provenance", "Intrinsics", cal["intrinsics"]),
        ("Provenance", "Calibration scope", VEH["calibration_scope"]),
    ]


def source_to_target_offsets():
    """Distance from the published source camera to each target camera.

    Returns (camera_id, distance_m, height_change_m), nearest first. This is the
    only camera-displacement quantity the evidence supports: the source vehicle
    envelope and its remaining camera mountings are not published.
    """
    import math
    s = SRC_CAM["position_rig_m"]
    out = []
    for c in RIG["cameras"]:
        p = c["mount"]["position_ego_m"]
        out.append((c["camera_id"], math.dist(s, p), p[2] - s[2]))
    return sorted(out, key=lambda r: r[1])


# --------------------------------------------------------------- holdout
def _parse_view(name):
    m = re.match(r"(\d+)_(\d+)_(camera_[a-z0-9_]+?)(?:_([mp]\d+)_pitch)?_([mp]\d+)$", name)
    if not m:
        raise ValueError("unparsed holdout view: " + name)
    frame, _, cam, pitch, yaw = m.groups()
    return frame, cam, pitch or "p0", yaw


HOLD_VIEWS = []
for _v in HOLD["per_view"]:
    _f, _c, _p, _y = _parse_view(_v["image"])
    HOLD_VIEWS.append(dict(frame=_f, cam=_c, pitch=_p, yaw=_y, **_v))

HOLD_FRAMES = sorted({v["frame"] for v in HOLD_VIEWS})
HOLD_CAMS = sorted({v["cam"] for v in HOLD_VIEWS})
HOLD_MAE = [v["mae_m"] for v in HOLD_VIEWS]


def _tiles_per_frame():
    """Virtual-pinhole tiles per timestamp, read from the holdout itself.

    The holdout view count is the product of timestamps and tiles per timestamp.
    It is *not* that product times the camera count: the tiles are distributed
    across the source cameras, not repeated per camera. Stating it as a
    three-way product overcounts by the number of cameras, which is the error
    this helper exists to make impossible.
    """
    per_frame = set(collections.Counter(v["frame"] for v in HOLD_VIEWS).values())
    if len(per_frame) != 1:
        raise ValueError("holdout timestamps carry unequal tile counts: %s"
                         % sorted(per_frame))
    n = per_frame.pop()
    if n * len(HOLD_FRAMES) != HOLD["views"]:
        raise ValueError("holdout arithmetic does not close: %d x %d != %d"
                         % (n, len(HOLD_FRAMES), HOLD["views"]))
    return n


def _tile_split():
    """How the tiles per timestamp divide across the physical source cameras."""
    per_cam = collections.Counter(v["cam"] for v in HOLD_VIEWS)
    nf = len(HOLD_FRAMES)
    groups = collections.defaultdict(list)
    for cam, n in per_cam.items():
        groups[n // nf].append(cam)
    return "; ".join(
        "%d from %d camera%s" % (k, len(groups[k]), "s" if len(groups[k]) > 1 else "")
        for k in sorted(groups, reverse=True))


def wmae(rows):
    px = sum(r["pixels"] for r in rows)
    return sum(r["mae_m"] * r["pixels"] for r in rows) / px, px


def mounting_sensitivity(scales=(1.0, 1.25, 1.5, 1.75, 2.0)):
    """How far the target cameras sit from the source camera as the rig grows.

    Reviewers reasonably ask how the coverage figures would move on the larger
    deployment platform. That cannot be measured: the deployment vehicle's
    extrinsics are not available, which is the whole gap. What *can* be computed
    is the quantity the angular and distance-ratio gates are driven by — the
    displacement between the one source-camera mounting NCore publishes and each
    target camera — under a first-order model in which a larger body spreads the
    mountings proportionally about the ego origin. Camera height is held fixed,
    because a taller body does not necessarily raise the cameras.

    This is a geometric model, not a measurement of any real platform. It bounds
    a direction, not a value.

    Returns rows of (scale, vehicle_length_m, min_m, max_m, mean_m).
    """
    import math
    s = SRC_CAM["position_rig_m"]
    base = [c["mount"]["position_ego_m"] for c in RIG["cameras"]]
    out = []
    for k in scales:
        ds = [math.dist(s, (p[0] * k, p[1] * k, p[2])) for p in base]
        out.append((k, VEH["length_m"] * k, min(ds), max(ds), sum(ds) / len(ds)))
    return out


def dataset_panels():
    """How many delivered frames the dataset figures actually plot.

    `figures/make_figs_dataset.py` writes this when it runs, so the count the
    manuscript quotes is the count that was drawn rather than a literal typed
    into the prose and left to rot when a figure gains or loses a panel.
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "figures", "dataset_panels.json")
    if not os.path.exists(p):
        raise ValueError("run figures/make_figs_dataset.py first: %s is missing" % p)
    d = _load(p)
    parts = {k: v for k, v in d.items() if k != "total"}
    if sum(parts.values()) != d["total"]:
        raise ValueError("dataset panel counts do not sum: %r" % d)
    return d


def abstention_split(fps=30.0):
    """Split the no-actor camera-frames into 'no candidate' and 'gates refused'.

    The headline no-actor fraction pools two different events: frames where no
    actor expert is even temporally valid, and frames where one is valid but no
    gate admitted it. Only the second is a refusal, so quoting the pooled figure
    as a rejection rate overstates what the gates actually withheld.

    Each expert carries a validity window in microseconds. Those windows span
    2.05-9.42 s against a clip of about ten s, so they are clip-relative, and frame f
    sits at f/fps seconds. A frame is *covered* when at least one expert window
    contains it. Coverage is a property of the source expert's timeline, not of
    any target camera, so every camera shares the same uncovered frames.

    Returns (uncovered_frames, no_candidate_cam_frames, refused_cam_frames).
    """
    n_cov = sum(
        1 for f in range(NFRAMES)
        if any(e["timestamp_start_us"] <= f / fps * 1e6 <= e["timestamp_end_us"]
               for e in EXPERTS))
    uncovered = NFRAMES - n_cov
    no_candidate = NCAM * uncovered
    refused = ABST["frames_zero_actor"] - no_candidate
    if refused < 0:
        raise ValueError("more uncovered camera-frames than no-actor frames: "
                         "%d > %d" % (no_candidate, ABST["frames_zero_actor"]))
    return uncovered, no_candidate, refused


def holdout_by_camera():
    out = []
    for c in HOLD_CAMS:
        sub = [r for r in HOLD_VIEWS if r["cam"] == c]
        m, px = wmae(sub)
        med = st.median(r["median_m"] for r in sub)
        out.append((c, len(sub), px, m, med))
    return sorted(out, key=lambda r: r[3])


HOLD_STATS = dict(
    views=HOLD["views"], pixels=HOLD["pixels"],
    mae=HOLD["mae_m"], median=HOLD["median_m"], p90=HOLD["p90_m"],
    rel_mae=HOLD["relative_mae"], iteration=HOLD["iteration"],
    n_frames=len(HOLD_FRAMES), n_cams=len(HOLD_CAMS),
    tiles_per_frame=_tiles_per_frame(), tile_split=_tile_split(),
    view_median=st.median(HOLD_MAE), view_min=min(HOLD_MAE), view_max=max(HOLD_MAE),
    view_q1=st.quantiles(HOLD_MAE, n=4)[0], view_q3=st.quantiles(HOLD_MAE, n=4)[2],
)


# --------------------------------------------------------------- registry
SEL = REG["selection"]
EXPERTS = REG["experts"]
REJECTED = REG["rejected"]
TRACKS = sorted({e["track_id"] for e in EXPERTS})


def expert_rows():
    out = []
    for e in EXPERTS:
        m = e["metrics"]
        out.append((e["track_id"], e["label"], e["source_camera"],
                    (e["timestamp_end_us"] - e["timestamp_start_us"]) / 1e6,
                    m["positive_frames"], m["masked_rgb_mae"],
                    m["alpha_iou_0_5"], m["alpha_area_ratio"]))
    return out


def rejected_rows():
    out = []
    for e in REJECTED:
        m = e["metrics"]
        out.append((e["name"], e["reason"], m["masked_rgb_mae"],
                    m["alpha_iou_0_5"], m["alpha_area_ratio"]))
    return out


REG_STATS = dict(
    n_accepted=len(EXPERTS), n_rejected=len(REJECTED), n_tracks=len(TRACKS),
    max_mae=SEL["max_masked_rgb_mae"], min_iou=SEL["min_alpha_iou"],
    area_lo=SEL["alpha_area_ratio"][0], area_hi=SEL["alpha_area_ratio"][1],
    overlap=SEL["overlap_fraction"],
    mae_lo=min(e["metrics"]["masked_rgb_mae"] for e in EXPERTS),
    mae_hi=max(e["metrics"]["masked_rgb_mae"] for e in EXPERTS),
    iou_lo=min(e["metrics"]["alpha_iou_0_5"] for e in EXPERTS),
    iou_hi=max(e["metrics"]["alpha_iou_0_5"] for e in EXPERTS),
    area_min=min(e["metrics"]["alpha_area_ratio"] for e in EXPERTS),
    area_max=max(e["metrics"]["alpha_area_ratio"] for e in EXPERTS),
)


# --------------------------------------------------------------- gates
GATE = dict(
    angle=COMP["max_view_angle_deg"],
    dist_lo=COMP["distance_ratio"][0], dist_hi=COMP["distance_ratio"][1],
    fade_ms=COMP["temporal_fade_us"] / 1000.0,
    fade_frames=COMP["temporal_fade_us"] / 1e6 * 30.0,
    depth_quant=COMP["background_depth_quantization_m"],
    occl_margin=COMP["static_occlusion_margin_m"],
)
EXPERT_NAMES = [e["name"] for e in COMP["experts"]]
EXPERT_LABEL = {}
for _e in COMP["experts"]:
    _cam = _e["name"].split("_", 1)[1].rsplit("_", 1)[0]
    EXPERT_LABEL[_e["name"]] = "t%d/%s" % (
        _e["track_id"], _cam.replace("camera_", "").replace("_120fov", "")
                            .replace("_70fov", ""))
PROTO = {e["name"]: dict(dist=e["prototype_distance_m"],
                         window=e["observation_timestamps_us"])
         for e in COMP["experts"]}


# --------------------------------------------------------------- per camera
def cam_metrics(c):
    return CAMJ[c]["metrics"]


def cam_rows():
    out = []
    for c in CAMS:
        m = cam_metrics(c)
        out.append((c, m["actor_active_frames"],
                    1 - m["actor_active_frames"] / float(NFRAMES),
                    m["actor_alpha_area_mean"], m["actor_alpha_area_active_mean"],
                    m["outside_actor_rgb_mae"], m["outside_actor_rgb_mae_max"],
                    m["adjacent_alpha_iou_mean"], m["adjacent_alpha_iou_p05"]))
    return out


def live_experts(cam):
    """Registry members that are admitted at least once in this target view."""
    counts = COMP["counts"][cam]
    return [n for n in EXPERT_NAMES if counts.get(n, 0) > 0]


def gate_attribution():
    """Which (target camera, expert) pairs the gates admit at all.

    A pair that never activates is refused for every frame of the clip, so this
    isolates structural refusal from merely intermittent refusal.
    """
    rows = [(c, len(live_experts(c)), [EXPERT_LABEL[n] for n in live_experts(c)])
            for c in CAMS]
    dead = sum(len(EXPERT_NAMES) - n for _c, n, _l in rows)
    return rows, dead, len(CAMS) * len(EXPERT_NAMES)


def activation_rows():
    out = []
    for c in CAMS:
        counts = COMP["counts"][c]
        out.append((c, [counts.get(n, 0) for n in EXPERT_NAMES],
                    cam_metrics(c)["actor_active_frames"]))
    return out


_ACT = [cam_metrics(c)["actor_active_frames"] for c in CAMS]
_AREA = [cam_metrics(c)["actor_alpha_area_mean"] for c in CAMS]
_IOU = [cam_metrics(c)["adjacent_alpha_iou_mean"] for c in CAMS]
_OUT = [cam_metrics(c)["outside_actor_rgb_mae"] for c in CAMS]
_OUTX = [cam_metrics(c)["outside_actor_rgb_mae_max"] for c in CAMS]

ABST = dict(
    frames_zero_actor=NTOTAL - sum(_ACT),
    frac_zero_actor=(NTOTAL - sum(_ACT)) / float(NTOTAL),
    rate_lo=1 - max(_ACT) / float(NFRAMES),
    rate_hi=1 - min(_ACT) / float(NFRAMES),
    act_lo=min(_ACT), act_hi=max(_ACT),
    area_lo=min(_AREA), area_hi=max(_AREA),
    area_mean=sum(_AREA) / len(_AREA),
    iou_lo=min(_IOU), iou_hi=max(_IOU),
    out_lo=min(_OUT), out_hi=max(_OUT), out_max=max(_OUTX),
    n_p05_zero=sum(1 for c in CAMS if cam_metrics(c)["adjacent_alpha_iou_p05"] == 0),
    cams_p05_zero=[c for c in CAMS if cam_metrics(c)["adjacent_alpha_iou_p05"] == 0],
)

DELIVERY = {r["metric"]: r["value"] for r in QUALITY if r["scope"] == "global"}


# --------------------------------------------------------------- formatting
_SUP = {"-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
        "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078",
        "9": "\u2079"}


def sci(x, sig=3):
    """1.79e-06 -> '1.79 x 10\u207b\u2076'. Unicode superscripts keep the exponent legible
    in a plain DOCX run, where a real superscript would need its own run."""
    mant, exp = ("%.*e" % (sig - 1, x)).split("e")
    return "%s \u00d7 10%s" % (mant, "".join(_SUP[c] for c in str(int(exp))))


def pct(x, d=1):
    return "%.*f%%" % (d, 100 * x)


def provenance():
    """Audit trail: claim -> file -> value."""
    P = []
    a = P.append
    a(("delivery.frames", "quality_metrics.csv", DELIVERY["total_rgb_frames"]))
    a(("delivery.fps", "quality_metrics.csv", DELIVERY["fps"]))
    a(("delivery.duration_s", "quality_metrics.csv", DELIVERY["duration"]))
    a(("delivery.completion", "quality_metrics.csv", DELIVERY["completion_rate"]))
    for k in ("views", "pixels", "mae", "median", "p90", "rel_mae", "iteration"):
        a(("holdout." + k, "evidence/road_lidar_holdout_same_static_model.json",
           HOLD_STATS[k]))
    for k in ("view_median", "view_min", "view_max", "n_frames", "n_cams"):
        a(("holdout." + k, "derived from per_view[240]", HOLD_STATS[k]))
    for k, v in REG_STATS.items():
        a(("registry." + k, "evidence/accepted_actor_registry.json", v))
    for k, v in GATE.items():
        a(("gate." + k, "evidence/actor_composite_1080p_manifest.json", v))
    for c in CAMS:
        for k, v in cam_metrics(c).items():
            a(("camera.%s.%s" % (c, k), "evidence/%s.json" % c, v))
    for k, v in ABST.items():
        a(("abstention." + k, "derived from evidence/<camera>.json", v))
    for c in RIG["cameras"]:
        a(("rig.%s.position_ego_m" % c["camera_id"], "configs/7fd4_neolix_x3_size_7v.json",
           c["mount"]["position_ego_m"]))
        a(("rig.%s.yaw_pitch_roll_deg" % c["camera_id"], "configs/7fd4_neolix_x3_size_7v.json",
           c["mount"]["yaw_pitch_roll_deg"]))
    a(("static.iteration", "evidence/static_2dgs_1080p_manifest.json", STATIC["iteration"]))
    a(("static.sky_conf_threshold", "evidence/static_2dgs_1080p_manifest.json",
       STATIC["sky_confidence_threshold"]))
    for c in CAMS:
        a(("static.counts.%s" % c, "evidence/static_2dgs_1080p_manifest.json",
           STATIC["counts"][c]))
    return P


if __name__ == "__main__":
    for name, src, val in provenance():
        print("%-46s %-58s %s" % (name, src, val))
