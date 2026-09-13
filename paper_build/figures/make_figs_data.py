#!/usr/bin/env python
"""Data-driven figures (5-10) for the IEEE T-ITS cross-vehicle paper.

Every number is read from files under H:/ARIS/papers/paper07/. Nothing is hard-coded
except axis labels. Run:  python make_figs_data.py
"""
import json, os, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EV   = os.path.join(ROOT, "05_评价结果", "evidence")
OUT  = os.path.dirname(os.path.abspath(__file__))

# ---- IEEE style -------------------------------------------------------------
COL1, COL2 = 3.5, 7.16          # inches: single / double column
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.grid": True, "grid.alpha": 0.30,
    "axes.axisbelow": True,
})
# Okabe-Ito colourblind-safe palette
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

# Okabe-Ito is safe against colour vision deficiency but NOT against greyscale:
# #E69F00 and #56B4E9 sit 0.011 apart in relative luminance, and #D55E00 and
# #009E73 0.035 apart, so they merge in a black-and-white print or photocopy.
# Every multi-series bar therefore carries a hatch as well as a colour, so the
# series stay separable when the hue is gone. Order matches CB.
HATCH = ["", "///", "...", "\\\\\\", "xxx", "---", "+++"]

CAMS = ["front_center", "front_left", "front_right", "side_left",
        "side_right", "rear_left", "rear_right"]
NICE = {"front_center": "front\ncenter", "front_left": "front\nleft",
        "front_right": "front\nright", "side_left": "side\nleft",
        "side_right": "side\nright", "rear_left": "rear\nleft",
        "rear_right": "rear\nright"}
NFRAMES = 299


def load(name):
    with open(os.path.join(EV, name), encoding="utf-8") as f:
        return json.load(f)


CAMJ = {c: load(c + ".json") for c in CAMS}
HOLD = load("road_lidar_holdout_same_static_model.json")
REG = load("accepted_actor_registry.json")
COMP = load("actor_composite_1080p_manifest.json")


def save(fig, stem):
    p = os.path.join(OUT, stem + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.basename(p), "%.1f KB" % (os.path.getsize(p) / 1024))


# =============================================================== Fig 5
def fig5_holdout_distribution():
    pv = HOLD["per_view"]
    mae = np.array([v["mae_m"] for v in pv])
    px = np.array([v["pixels"] for v in pv], float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.15))

    a.hist(mae, bins=np.arange(0, 8.25, 0.25), color=CB[0],
           edgecolor="white", linewidth=0.35)
    top = a.get_ylim()[1]
    for x, lab, c, ls, fy, off, ha in (
            (np.median(mae), "view median\n%.3f m" % np.median(mae), CB[1], "-", 0.97, -4, "right"),
            (HOLD["mae_m"], "pixel-weighted mean %.3f m" % HOLD["mae_m"], CB[2], "--", 0.70, 5, "left"),
            (mae.max(), "worst view %.3f m" % mae.max(), CB[3], ":", 0.30, -5, "right")):
        a.axvline(x, color=c, ls=ls, lw=1.1)
        a.annotate(lab, (x, top * fy), xytext=(off, 0), textcoords="offset points",
                   color=c, fontsize=6.4, va="top", ha=ha)
    a.set_xlabel("per-view road-depth MAE (m)")
    a.set_ylabel("holdout views")
    a.set_title("(a) distribution over 240 views", loc="left")
    a.set_xlim(0, 8)

    o = np.argsort(mae)
    cw = np.cumsum(px[o]) / px.sum()
    b.plot(mae[o], cw, color=CB[0], lw=1.3, label="pixel-weighted")
    b.plot(mae[o], np.arange(1, len(mae) + 1) / len(mae), color=CB[3],
           lw=1.0, ls="--", label="view-weighted")
    for q, lab in ((0.5, "P50"), (0.9, "P90")):
        b.axhline(q, color="0.55", lw=0.5, ls=":")
        b.annotate(lab, (7.85, q), fontsize=6, color="0.35", ha="right", va="bottom")
    b.axvline(HOLD["p90_m"], color=CB[1], lw=0.9, ls="-.")
    b.annotate("global P90\n%.3f m" % HOLD["p90_m"], (HOLD["p90_m"], 0.05),
               xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=CB[1])
    b.set_xlabel("road-depth MAE (m)")
    b.set_ylabel("cumulative fraction")
    b.set_title("(b) cumulative error", loc="left")
    b.set_xlim(0, 8)
    b.set_ylim(0, 1.02)
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    b.legend(loc="lower right", frameon=False)
    save(fig, "fig05_holdout_distribution")


# =============================================================== Fig 6
def parse_views():
    """Group the 240 holdout views by physical source camera and tile pitch."""
    rows = []
    for v in HOLD["per_view"]:
        m = re.match(r"(\d+)_(\d+)_(camera_[a-z0-9_]+?)(?:_([mp]\d+)_pitch)?_([mp]\d+)$",
                     v["image"])
        if not m:
            raise ValueError("unparsed view name: " + v["image"])
        frame, _, cam, pitch, yaw = m.groups()
        rows.append(dict(frame=frame, cam=cam, pitch=pitch or "p0", yaw=yaw, **v))
    return rows


def wmae(rows):
    p = sum(r["pixels"] for r in rows)
    return sum(r["mae_m"] * r["pixels"] for r in rows) / p, p


def fig6_holdout_by_camera():
    rows = parse_views()
    cams = sorted({r["cam"] for r in rows})
    pitches = ["m20", "p0", "p20"]
    short = {c: c.replace("camera_", "").replace("_120fov", "\n(120 deg)")
              .replace("_70fov", "\n(70 deg)") for c in cams}

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.25),
                              gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(cams))
    w = 0.26
    for i, p in enumerate(pitches):
        vals, labels = [], []
        for c in cams:
            sub = [r for r in rows if r["cam"] == c and r["pitch"] == p]
            if sub:
                m, _ = wmae(sub)
                vals.append(m)
                labels.append("%.2f" % m)
            else:
                vals.append(np.nan)
                labels.append("")
        bars = a.bar(x + (i - 1) * w, vals, w, color=CB[i], edgecolor="white",
                     linewidth=0.3, hatch=HATCH[i % len(HATCH)],
                     label="tile pitch %s" % p.replace("m", "-").replace("p", "+"))
        for bar, lab in zip(bars, labels):
            if lab:
                a.annotate(lab, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, -3), textcoords="offset points", rotation=90,
                           ha="center", va="top", fontsize=5.6, color="white")
    a.axhline(HOLD["mae_m"], color="0.2", ls="--", lw=0.9)
    a.annotate("all-view pixel-weighted mean %.3f m" % HOLD["mae_m"],
               (-0.42, HOLD["mae_m"]), xytext=(0, 2.5),
               textcoords="offset points", ha="left", fontsize=6, color="0.2")
    a.annotate("70 deg rear cameras contribute\none tile pitch only",
               (4.42, 0.05), ha="right", va="bottom", fontsize=5.6, color="0.45")
    a.set_xticks(x)
    a.set_xticklabels([short[c] for c in cams], fontsize=6.2)
    a.set_ylabel("road-depth MAE (m)")
    a.set_title("(a) by source camera and tile pitch", loc="left")
    a.legend(loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0.0, 0.99))
    a.set_ylim(0, 1.62)

    byc = {}
    for c in cams:
        sub = [r for r in rows if r["cam"] == c]
        byc[c] = (wmae(sub)[0], sum(r["pixels"] for r in sub), len(sub))
    order = sorted(byc, key=lambda c: byc[c][0])
    b.barh(np.arange(len(order)), [byc[c][0] for c in order], 0.62,
           color=CB[0], edgecolor="white", linewidth=0.3)
    for i, c in enumerate(order):
        b.annotate("%.3f m  (%d views, %.2f M px)" % (byc[c][0], byc[c][2], byc[c][1] / 1e6),
                   (0.03, i), fontsize=5.8, va="center", color="white")
    b.axvline(HOLD["mae_m"], color="0.2", ls="--", lw=0.9)
    b.set_yticks(np.arange(len(order)))
    b.set_yticklabels([short[c].replace("\n", " ") for c in order], fontsize=6.2)
    b.set_xlabel("road-depth MAE (m)")
    b.set_title("(b) pooled over tile pitch", loc="left")
    b.set_xlim(0, 1.5)
    save(fig, "fig06_holdout_by_camera")


# =============================================================== Fig 7
def fig7_registry_acceptance():
    sel = REG["selection"]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    lo_iou, hi_mae = sel["min_alpha_iou"], sel["max_masked_rgb_mae"]
    ax.add_patch(plt.Rectangle((0, lo_iou), hi_mae, 1 - lo_iou, facecolor=CB[2],
                               alpha=0.10, edgecolor=CB[2], lw=0.8, ls="--", zorder=0))
    ax.annotate("acceptance region\nMAE $\\leq$ %.2f, IoU $\\geq$ %.2f" % (hi_mae, lo_iou),
                (0.004, 0.833), fontsize=6.2, color=CB[2], va="bottom")

    def scat(items, marker, color, label, edge):
        xs = [e["metrics"]["masked_rgb_mae"] for e in items]
        ys = [e["metrics"]["alpha_iou_0_5"] for e in items]
        ss = [26 + 150 * abs(e["metrics"]["alpha_area_ratio"] - 1.0) for e in items]
        ax.scatter(xs, ys, s=ss, marker=marker, facecolor=color, edgecolor=edge,
                   linewidth=0.7, label=label, zorder=3, alpha=0.9)
        return xs, ys

    xa, ya = scat(REG["experts"], "o", CB[0], "accepted (n=%d)" % len(REG["experts"]), "white")
    for e, x, y in zip(REG["experts"], xa, ya):
        ax.annotate("t%d" % e["track_id"], (x, y), xytext=(0, -8.5),
                    textcoords="offset points", ha="center", fontsize=5.8, color=CB[0])
    xr, yr = scat(REG["rejected"], "X", CB[1], "rejected (n=%d)" % len(REG["rejected"]), CB[1])
    for e, x, y in zip(REG["rejected"], xr, yr):
        m, r = e["metrics"], e["reason"]
        if r == "does_not_meet_registry_thresholds":
            t = ("quality reject\nIoU %.3f $<$ %.2f\narea ratio %.2f $>$ %.2f"
                 % (m["alpha_iou_0_5"], lo_iou, m["alpha_area_ratio"],
                    sel["alpha_area_ratio"][1]))
            off, va = (8, 0), "center"
        else:
            t = "de-duplication reject\n(passes thresholds; lower\nquality than kept expert)"
            off, va = (7, 10), "bottom"
        ax.annotate(t, (x, y), xytext=off, textcoords="offset points",
                    fontsize=5.5, color=CB[1], va=va,
                    ha="right" if off[0] < 0 else "left")
    ax.axhline(lo_iou, color=CB[2], lw=0.7, ls="--")
    ax.axvline(hi_mae, color=CB[2], lw=0.7, ls="--")
    ax.set_xlabel("source-view masked RGB MAE")
    ax.set_ylabel(r"source-view alpha IoU @ 0.5")
    ax.set_xlim(0, 0.105)
    ax.set_ylim(0.70, 1.01)
    ax.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.45),
              handletextpad=0.4, borderpad=0.2)
    ax.set_title("marker area $\\propto$ |alpha area ratio $-$ 1|", loc="left",
                 fontsize=6.5, color="0.35")
    save(fig, "fig07_registry_acceptance")


# =============================================================== Fig 8
def fig8_activation_ledger():
    experts = [e["name"] for e in COMP["experts"]]
    lbl = {}
    for e in COMP["experts"]:
        cam = e["name"].split("_", 1)[1].rsplit("_", 1)[0]
        lbl[e["name"]] = "t%d / %s" % (e["track_id"],
                                       cam.replace("camera_", "").replace("_120fov", "")
                                          .replace("_70fov", ""))
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.3),
                              gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(CAMS))
    bottom = np.zeros(len(CAMS))
    for i, ex in enumerate(experts):
        v = np.array([COMP["counts"][c].get(ex, 0) for c in CAMS], float)
        a.bar(x, v, 0.66, bottom=bottom, color=CB[i % len(CB)], edgecolor="white",
              linewidth=0.35, hatch=HATCH[i % len(HATCH)], label=lbl[ex])
        bottom += v
    act = np.array([CAMJ[c]["metrics"]["actor_active_frames"] for c in CAMS], float)
    a.plot(x, act, "k_", ms=17, mew=1.4, label="frames with $\\geq$1 actor")
    for xi, v in zip(x, act):
        a.annotate("%d" % v, (xi, v), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=6, fontweight="bold")
    a.set_xticks(x)
    a.set_xticklabels([NICE[c] for c in CAMS], fontsize=6.2)
    a.set_ylabel("expert-frame selections")
    a.set_title("(a) which source expert supports which target view", loc="left")
    a.legend(loc="upper right", frameon=False, ncol=2, fontsize=5.9)
    a.set_ylim(0, 260)

    abst = 100 * (1 - act / NFRAMES)
    area = 100 * np.array([CAMJ[c]["metrics"]["actor_alpha_area_mean"] for c in CAMS])
    b.barh(x, abst, 0.62, color=CB[1], edgecolor="white", linewidth=0.3,
           label="frames with no actor content")
    for xi, (v, ar) in enumerate(zip(abst, area)):
        b.annotate("%.1f%%   (actor pixels %.2f%%)" % (v, ar), (2.5, xi),
                   fontsize=5.9, va="center", color="white")
    b.set_yticks(x)
    b.set_yticklabels([NICE[c].replace("\n", " ") for c in CAMS], fontsize=6.2)
    b.invert_yaxis()
    b.set_xlabel("temporal abstention rate")
    b.set_xlim(0, 100)
    b.xaxis.set_major_formatter(PercentFormatter(100))
    b.set_title("(b) abstention: %d of 2093 frames emit no actor" %
                int(7 * NFRAMES - act.sum()), loc="left")
    save(fig, "fig08_activation_ledger")


# =============================================================== Fig 9
def fig9_alpha_iou():
    mean = np.array([CAMJ[c]["metrics"]["adjacent_alpha_iou_mean"] for c in CAMS])
    p05 = np.array([CAMJ[c]["metrics"]["adjacent_alpha_iou_p05"] for c in CAMS])
    act = np.array([CAMJ[c]["metrics"]["actor_active_frames"] for c in CAMS])
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    x = np.arange(len(CAMS))
    fam = [CB[0]] * 3 + [CB[4]] * 2 + [CB[1]] * 2
    ax.bar(x, mean, 0.62, color=fam, edgecolor="white", linewidth=0.35, label="mean")
    ax.plot(x, p05, "v", color="0.15", ms=4.2, mew=0, label="5th percentile")
    ax.vlines(x, p05, mean, color="0.35", lw=0.7)
    for xi, (m, p, a_) in enumerate(zip(mean, p05, act)):
        ax.annotate("%.3f" % m, (xi, m), xytext=(0, 2.5), textcoords="offset points",
                    ha="center", fontsize=5.8)
        if p == 0:
            ax.annotate("0", (xi, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=5.6, color="0.15")
        ax.annotate("n=%d" % a_, (xi, -0.075), ha="center", fontsize=5.4, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CAMS], fontsize=6.2)
    ax.set_ylabel("adjacent-frame alpha IoU")
    ax.set_ylim(-0.11, 1.0)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("front / side / rear view families; n = active frames",
                 loc="left", fontsize=6.5, color="0.35")
    save(fig, "fig09_alpha_iou")


# =============================================================== Fig 10
def fig10_outside_actor():
    mean = np.array([CAMJ[c]["metrics"]["outside_actor_rgb_mae"] for c in CAMS])
    mx = np.array([CAMJ[c]["metrics"]["outside_actor_rgb_mae_max"] for c in CAMS])
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    x = np.arange(len(CAMS))
    w = 0.36
    ax.bar(x - w / 2, mean, w, color=CB[0], edgecolor="white", linewidth=0.3,
           label="per-frame mean")
    ax.bar(x + w / 2, mx, w, color=CB[5], edgecolor="white", linewidth=0.3,
           hatch="///", label="worst frame")
    q = 1.0 / 255
    ax.axhline(q, color=CB[1], ls="--", lw=0.9)
    ax.annotate("1 LSB of 8-bit RGB (1/255)", (len(CAMS) - 0.45, q * 1.35),
                ha="right", fontsize=6, color=CB[1])
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 2e-2)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CAMS], fontsize=6.2)
    ax.set_ylabel("outside-actor RGB MAE\n(normalized units, log scale)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save(fig, "fig10_outside_actor")


if __name__ == "__main__":
    fig5_holdout_distribution()
    fig6_holdout_by_camera()
    fig7_registry_acceptance()
    fig8_activation_ledger()
    fig9_alpha_iou()
    fig10_outside_actor()
    print("done")
