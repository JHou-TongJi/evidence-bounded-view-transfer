#!/usr/bin/env python
"""Target platform elevations: side and front views with sensor placement.

Every dimension and every sensor position is read from
03_工程代码/configs/7fd4_neolix_x3_size_7v.json. Nothing is hand-placed, so the
drawing cannot disagree with the rig the renderer actually used.

This is a dimensioned engineering drawing rather than a product photograph, for
two reasons stated in the manuscript: the rig is an engineering proxy and not any
vendor's calibration, and a drawing can carry the mounting geometry that a photo
cannot.

    python make_fig_platform.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "03_工程代码", "configs", "7fd4_neolix_x3_size_7v.json")

COL2 = 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
BODY_FC, BODY_EC = "#eef1f4", "#4a5560"


def dim(ax, p, q, text, off=0.0, vertical=False, fs=5.8, color="0.30"):
    """Dimension line with arrows and a centred label."""
    if vertical:
        x = p[0] + off
        ax.annotate("", (x, p[1]), (x, q[1]),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=color))
        ax.plot([p[0], x], [p[1], p[1]], lw=0.4, color=color, ls=":")
        ax.plot([q[0], x], [q[1], q[1]], lw=0.4, color=color, ls=":")
        ax.text(x + 0.045, (p[1] + q[1]) / 2, text, rotation=90, ha="left",
                va="center", fontsize=fs, color=color)
    else:
        y = p[1] + off
        ax.annotate("", (p[0], y), (q[0], y),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=color))
        ax.plot([p[0], p[0]], [p[1], y], lw=0.4, color=color, ls=":")
        ax.plot([q[0], q[0]], [q[1], y], lw=0.4, color=color, ls=":")
        ax.text((p[0] + q[0]) / 2, y + 0.035, text, ha="center", va="bottom",
                fontsize=fs, color=color)


def main():
    with open(CFG, encoding="utf-8") as f:
        rig = json.load(f)
    v, intr, hw = rig["vehicle"], rig["intrinsics"], rig["camera_hardware_reference"]
    L, W, H, WB = (v["length_m"], v["width_m"],
                   v["height_without_lidar_m"], v["wheelbase_m"])
    cams = rig["cameras"]
    lidar = rig["lidars"][0]

    front_x = max(c["mount"]["position_ego_m"][0] for c in cams) + 0.01
    rear_x = front_x - L

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, 2.60),
                                 gridspec_kw={"width_ratios": [1.62, 1]})

    # ---------------------------------------------------------- (a) side
    ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((rear_x, 0), L, H,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=BODY_FC, edgecolor=BODY_EC,
                                linewidth=1.0, zorder=1))
    ax.plot([rear_x - 0.35, front_x + 0.35], [0, 0], color="0.35", lw=1.1)
    for wx in (0.0, WB):                       # rear axle at origin, front axle
        ax.add_patch(Circle((wx, 0.30), 0.30, facecolor="0.80",
                            edgecolor="0.35", linewidth=0.8, zorder=2))
        ax.add_patch(Circle((wx, 0.30), 0.11, facecolor="0.60",
                            edgecolor="0.35", linewidth=0.5, zorder=3))

    for i, c in enumerate(cams):
        x, y, z = c["mount"]["position_ego_m"]
        pitch = c["mount"]["yaw_pitch_roll_deg"][1]
        col = CB[i % len(CB)]
        ax.add_patch(Wedge((x, z), 0.42, pitch - 8, pitch + 8, facecolor=col,
                           alpha=0.22, edgecolor="none", zorder=4))
        ax.plot([x], [z], "o", ms=3.6, color=col, mec="white", mew=0.5, zorder=6)

    lx, ly, lz = lidar["position_ego_m"]
    ax.add_patch(Rectangle((lx - 0.10, H), 0.20, lz - H, facecolor="0.55",
                           edgecolor="0.3", linewidth=0.7, zorder=5))
    ax.annotate("roof LiDAR %.2f m\n(configuration metadata only)" % lz,
                (lx, lz), xytext=(16, 6), textcoords="offset points",
                ha="left", fontsize=5.6, color="0.35",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55"))

    dim(ax, (rear_x, H), (front_x, H), "length %.2f m" % L, off=0.80)
    dim(ax, (0.0, 0.0), (WB, 0.0), "wheelbase %.2f m" % WB, off=-0.52)
    dim(ax, (front_x, 0.0), (front_x, H), "height %.2f m" % H, vertical=True,
        off=0.46)
    zmin = min(c["mount"]["position_ego_m"][2] for c in cams)
    zmax = max(c["mount"]["position_ego_m"][2] for c in cams)
    dim(ax, (rear_x, zmin), (rear_x, zmax),
        "cameras %.2f-%.2f m" % (zmin, zmax), vertical=True, off=-0.62)
    ax.plot(0, 0, marker="+", ms=8, mew=1.0, color="0.3", zorder=7)
    ax.annotate("ego origin\n(rear axle, ground)", (0, 0), xytext=(-6, -13),
                textcoords="offset points", ha="right", va="top", fontsize=5.5,
                color="0.35")
    ax.set_xlim(rear_x - 1.35, front_x + 0.95)
    ax.set_ylim(-1.05, 3.10)
    ax.axis("off")
    ax.text(rear_x - 1.30, -1.00, "(a) side elevation, +x forward",
            fontsize=7.0, ha="left", va="bottom")

    # ---------------------------------------------------------- (b) front
    bx.set_aspect("equal")
    bx.add_patch(FancyBboxPatch((-W / 2, 0), W, H,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=BODY_FC, edgecolor=BODY_EC,
                                linewidth=1.0, zorder=1))
    bx.plot([-W / 2 - 0.35, W / 2 + 0.35], [0, 0], color="0.35", lw=1.1)
    for i, c in enumerate(cams):
        x, y, z = c["mount"]["position_ego_m"]
        bx.plot([-y], [z], "o", ms=3.6, color=CB[i % len(CB)], mec="white",
                mew=0.5, zorder=6)
    bx.add_patch(Rectangle((-0.10, H), 0.20, lz - H, facecolor="0.55",
                           edgecolor="0.3", linewidth=0.7, zorder=5))
    dim(bx, (-W / 2, H), (W / 2, H), "width %.2f m" % W, off=0.42)
    ymax = max(abs(c["mount"]["position_ego_m"][1]) for c in cams)
    dim(bx, (-ymax, 0.10), (ymax, 0.10),
        "lateral spread %.2f m" % (2 * ymax), off=-0.55)
    bx.set_xlim(-1.50, 1.50)
    bx.set_ylim(-1.05, 3.10)
    bx.axis("off")
    bx.text(-1.47, -1.00, "(b) front elevation, +y left", fontsize=7.0,
            ha="left", va="bottom")

    fig.text(0.5, -0.030,
             "Seven rectified pinhole cameras, %d x %d at %d FPS, %.0f deg "
             "horizontal / %.2f deg vertical FOV, $f_x=f_y$ = %.2f px.\n"
             "Raw-lens reference: %s, %d x %d, %.0f deg H / %.0f deg V, "
             "up to %d FPS.\n"
             "Payload %d kg, cargo %.1f m$^3$, design speed %d km/h."
             % (intr["width"], intr["height"], rig["output"]["fps"],
                cams[0]["mount"]["horizontal_fov_deg"],
                cams[0]["mount"]["vertical_fov_deg"], intr["fx"],
                hw["model"], hw["raw_sensor_resolution"][0],
                hw["raw_sensor_resolution"][1], hw["raw_lens_horizontal_fov_deg"],
                hw["raw_lens_vertical_fov_deg"], hw["maximum_raw_frame_rate_fps"],
                v["payload_kg"], v["cargo_volume_m3"],
                v["maximum_design_speed_kph"]),
             ha="center", fontsize=5.9, color="0.32", linespacing=1.5)

    out = os.path.join(HERE, "fig05_platform.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote %s (%.1f KB)" % (os.path.basename(out),
                                  os.path.getsize(out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
