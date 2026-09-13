#!/usr/bin/env python
"""Schematic figures (1-4) for the IEEE T-ITS cross-vehicle paper.

Fig. 2 is drawn directly from 03_工程代码/configs/7fd4_neolix_x3_size_7v.json --
positions, yaw and horizontal FOV are never hard-coded.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Rectangle, Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = os.path.join(ROOT, "03_工程代码", "configs", "7fd4_neolix_x3_size_7v.json")
OUT = os.path.dirname(os.path.abspath(__file__))

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK = "#1a1a1a"
with open(CFG, encoding="utf-8") as f:
    RIG = json.load(f)


def save(fig, stem):
    p = os.path.join(OUT, stem + ".png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.basename(p), "%.1f KB" % (os.path.getsize(p) / 1024))


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=6.6, lw=0.8, ls="-", tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.35,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=INK, lw=0.9, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, color=color, linestyle=ls, zorder=4,
                                 connectionstyle="arc3,rad=%.2f" % rad,
                                 shrinkA=1.5, shrinkB=1.5))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ============================================================ Fig 1
def fig1_problem():
    """Cross-vehicle transfer, drawn to a shared metre scale.

    The target outline is the exact envelope from the rig configuration
    (length x width, plan view). The source outline is a nominal passenger-car
    envelope drawn to the same scale and labelled as indicative, because NCore
    does not publish the collection vehicle's body dimensions -- only one camera
    mounting, which anchors the two rigs in a common frame.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paper_data as D

    veh = D.VEH
    TGT_L, TGT_W = veh["length_m"], veh["width_m"]
    SRC_L, SRC_W = 4.80, 1.90          # nominal passenger car; see caption
    M = 0.045                          # plot units per metre (shared scale)

    fig, ax = plt.subplots(figsize=(COL2, 2.75))
    blank(ax)

    def veh_plan(cx, cy, L, W, color, cams, dashed=False):
        w, h = L * M, W * M
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=color, alpha=0.13, edgecolor=color,
                                    linewidth=1.1,
                                    linestyle="--" if dashed else "-", zorder=2))
        for (fx, fy, ang) in cams:
            px, py = cx - w / 2 + fx * w, cy + fy * h
            a = np.deg2rad(ang)
            ax.add_patch(Wedge((px, py), 0.050, ang - 27, ang + 27,
                               facecolor=color, alpha=0.20, edgecolor="none",
                               zorder=3))
            ax.add_patch(Circle((px, py), 0.0075, facecolor=color,
                                edgecolor="white", linewidth=0.4, zorder=5))
        return w, h

    ycar = 0.635
    sw, sh = veh_plan(0.160, ycar, SRC_L, SRC_W, CB[0],
                      [(0.86, 0.0, 0), (0.72, 0.42, 55), (0.72, -0.42, -55),
                       (0.20, 0.42, 145), (0.20, -0.42, -145)], dashed=True)
    tw, th = veh_plan(0.800, ycar, TGT_L, TGT_W, CB[1],
                      [(0.90, 0.0, 0), (0.80, 0.44, 50), (0.80, -0.44, -50),
                       (0.45, 0.46, 100), (0.45, -0.46, -100),
                       (0.09, 0.42, 150), (0.09, -0.42, -150)])

    ax.text(0.160, ycar + sh / 2 + 0.098, "SOURCE  passenger car", ha="center",
            fontsize=7.4, color=CB[0], fontweight="bold")
    ax.text(0.160, ycar + sh / 2 + 0.046,
            "7 x FTheta, rolling shutter, measured", ha="center", fontsize=6.0,
            color="0.35")
    ax.text(0.160, ycar - sh / 2 - 0.022,
            "envelope indicative (~%.1f x %.1f m): NCore does not publish it"
            % (SRC_L, SRC_W), ha="center", va="top", fontsize=5.7, color="0.45")

    ax.text(0.800, ycar + sh / 2 + 0.098, "TARGET  proxy L4 rig",
            ha="center", fontsize=7.4, color=CB[1], fontweight="bold")
    ax.text(0.800, ycar + sh / 2 + 0.046,
            "7 x rectified pinhole; not the deployment vehicle", ha="center",
            fontsize=6.0, color="0.35")
    ax.text(0.800, ycar - th / 2 - 0.022,
            "%.2f x %.2f m, wheelbase %.2f m (rig configuration)"
            % (TGT_L, TGT_W, veh["wheelbase_m"]), ha="center", va="top",
            fontsize=5.7, color="0.45")

    arrow(ax, (0.160 + sw / 2 + 0.045, ycar), (0.800 - tw / 2 - 0.045, ycar),
          color=CB[3], lw=1.6)
    ax.text(0.480, ycar + 0.030, r"rig substitution  $\mathcal{S}$", ha="center",
            fontsize=7.6, color=CB[3], fontweight="bold")

    off = D.source_to_target_offsets()
    near, far = off[0], off[-1]
    ax.text(0.5, 0.470,
            "Same world trajectory, different rig. From the one source mounting "
            "NCore publishes, the target cameras sit\n"
            "%.2f m (%s) to %.2f m (%s) away, and rise from %.2f m to %.2f-%.2f m "
            "above the ground."
            % (near[1], near[0].replace("_", " "), far[1], far[0].replace("_", " "),
               D.SRC_CAM["position_rig_m"][2],
               min(c["mount"]["position_ego_m"][2] for c in D.RIG["cameras"]),
               max(c["mount"]["position_ego_m"][2] for c in D.RIG["cameras"])),
            ha="center", va="top", fontsize=6.2, color="0.32", linespacing=1.5)

    # metre scale bar, shared by both outlines
    bx, by = 0.040, 0.330
    ax.plot([bx, bx + 2 * M], [by, by], color="0.25", lw=1.4,
            solid_capstyle="butt")
    for t in (0, 1, 2):
        ax.plot([bx + t * M] * 2, [by - 0.011, by + 0.011], color="0.25", lw=1.0)
    ax.text(bx + M, by + 0.020, "2 m", ha="center", fontsize=5.9, color="0.25")
    ax.text(bx + 2 * M + 0.012, by, "both outlines to scale", va="center",
            fontsize=5.8, color="0.45")

    # observability trichotomy
    ax.plot([0.02, 0.98], [0.275, 0.275], color="0.85", lw=0.6)
    ax.text(0.02, 0.243,
            "every target pixel is assigned exactly one observability class:",
            fontsize=6.3, color="0.30", va="top")
    trio = [("MEASURED",
             "road surface with a LiDAR return,\nor source pixels seen from an\nadmissible view",
             CB[2], 0.175),
            ("CONDITIONAL",
             "actor surface observed by a source\ncamera whose expert passes every\nadmissibility gate",
             CB[4], 0.500),
            ("UNKNOWN",
             "no admissible source view  ->  no\nactor content is emitted and an\ninvalid bit is set",
             CB[1], 0.825)]
    for name, desc, c, x in trio:
        ax.add_patch(FancyBboxPatch((x - 0.145, 0.150), 0.29, 0.040,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=c, edgecolor="none", zorder=2))
        ax.text(x, 0.170, name, ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=3)
        ax.text(x, 0.130, desc, ha="center", va="top", fontsize=5.9,
                color="0.28", linespacing=1.42)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.005, 0.80)
    save(fig, "fig01_problem")


# ============================================================ Fig 2
def fig2_rig_layout():
    cams = RIG["cameras"]
    v = RIG["vehicle"]
    L, W = v["length_m"], v["width_m"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, 2.6),
                                 gridspec_kw={"width_ratios": [1.28, 1]})

    # ---- (a) top-down plan, ego +x forward / +y left -> plot x=+y_ego, y=+x_ego
    ax.set_aspect("equal")
    front_x = max(c["mount"]["position_ego_m"][0] for c in cams) + 0.01
    ax.add_patch(Rectangle((-W / 2, front_x - L), W, L, facecolor="0.92",
                           edgecolor="0.55", linewidth=0.8, zorder=1))
    ax.plot(0, 0, marker="+", ms=8, mew=1.0, color="0.35", zorder=4)
    ax.annotate("rear-axle midpoint\n(ego origin)", (0, 0), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                fontsize=5.6, color="0.35")
    R = 2.35
    for i, c in enumerate(cams):
        px, py, pz = c["mount"]["position_ego_m"]
        yaw = c["mount"]["yaw_pitch_roll_deg"][0]
        fov = c["mount"]["horizontal_fov_deg"]
        col = CB[i % len(CB)]
        # plot frame: X = -y_ego (so +y_ego = left appears on the left), Y = +x_ego
        X, Y = -py, px
        th = 90.0 + yaw           # +x_ego is up (90 deg); +yaw rotates toward +y_ego (left)
        ax.add_patch(Wedge((X, Y), R, th - fov / 2, th + fov / 2, facecolor=col,
                           alpha=0.13, edgecolor=col, linewidth=0.45, zorder=2))
        ax.plot([X], [Y], "o", ms=3.4, color=col, mec="white", mew=0.5, zorder=6)
        a = np.deg2rad(th)
        lx, ly = X + 1.02 * R * np.cos(a), Y + 1.02 * R * np.sin(a)
        ytxt = "0 deg" if abs(yaw) < 1e-6 else "%+.0f deg" % yaw
        ax.text(lx, ly, "%s\n%s, %.2f m" % (c["camera_id"], ytxt, pz),
                ha="center", va="center", fontsize=5.3, color=col, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=col, lw=0.35, alpha=0.92),
                zorder=7)
    ax.arrow(0, 0, 0, 0.55, head_width=0.09, head_length=0.11, fc="0.3", ec="0.3", lw=0.7, zorder=5)
    ax.text(0.07, 0.45, "$+x$ fwd", fontsize=5.8, color="0.3")
    ax.arrow(0, 0, 0.5, 0, head_width=0.09, head_length=0.11, fc="0.3", ec="0.3", lw=0.7, zorder=5)
    ax.text(0.30, -0.20, "$-y$", fontsize=5.8, color="0.3")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.15, 4.05)
    ax.axis("off")
    ax.text(-3.2, -3.05, "(a) plan view: mounting points and 100 deg horizontal FOV",
            fontsize=7.2, ha="left", va="bottom")

    # ---- (b) azimuth coverage ring
    bx.set_aspect("equal")
    bx.axis("off")
    for i, c in enumerate(cams):
        yaw = c["mount"]["yaw_pitch_roll_deg"][0]
        fov = c["mount"]["horizontal_fov_deg"]
        r0 = 0.62 + 0.052 * i
        bx.add_patch(Wedge((0, 0), r0 + 0.046, 90 + yaw - fov / 2, 90 + yaw + fov / 2,
                           width=0.046, facecolor=CB[i % len(CB)], alpha=0.85,
                           edgecolor="white", linewidth=0.35))
    for ang, lab in ((90, "front"), (0, "right"), (180, "left"), (270, "rear")):
        a = np.deg2rad(ang)
        bx.text(1.10 * np.cos(a), 1.10 * np.sin(a), lab, ha="center", va="center",
                fontsize=6.2, color="0.35")
    bx.add_patch(Circle((0, 0), 0.55, facecolor="0.95", edgecolor="0.7", lw=0.6))
    bx.text(0, 0.10, "7 x 100 deg", ha="center", fontsize=7.0, fontweight="bold")
    bx.text(0, -0.03, "= 700 deg of", ha="center", fontsize=6.2, color="0.35")
    bx.text(0, -0.15, "azimuth over 360 deg", ha="center", fontsize=6.2, color="0.35")
    bx.text(0, -0.30, "(1.94x overlap)", ha="center", fontsize=6.0, color="0.5")
    bx.set_xlim(-1.35, 1.35)
    bx.set_ylim(-1.55, 1.32)
    bx.text(-1.35, -1.30, "(b) azimuth coverage", fontsize=7.2, ha="left", va="bottom")
    fig.text(0.5, -0.015,
             "vehicle envelope %.2f x %.2f x %.2f m, wheelbase %.2f m; all intrinsics "
             "$f_x=f_y=%.1f$ px, %d x %d. Engineering proxy, not an OEM calibration."
             % (v["length_m"], v["width_m"], v["height_without_lidar_m"], v["wheelbase_m"],
                RIG["intrinsics"]["fx"], RIG["intrinsics"]["width"], RIG["intrinsics"]["height"]),
             ha="center", fontsize=5.8, color="0.35")
    save(fig, "fig02_rig_layout")


# ============================================================ Fig 3
def fig3_frames():
    fig, ax = plt.subplots(figsize=(COL2, 1.95))
    blank(ax)
    stages = [
        (0.015, "source pixel\n$p=(u,v)$\nFTheta image", CB[0]),
        (0.178, "normalized\n$\\rho=\\sqrt{x^2+y^2}$\n$\\theta=\\sum a_i\\rho^i$", CB[0]),
        (0.341, "source camera ray\n$r_{\\mathrm{cam}}$\n(unit, OpenCV)", CB[0]),
        (0.504, "world point\n$T_{\\mathrm{cam}\\rightarrow world}^{src}(t_{row})$\nrolling-corrected", CB[2]),
        (0.667, "target camera\n$T_{cam\\rightarrow rig}^{k}$ then\n$T_{rig\\rightarrow world}(t)$", CB[1]),
        (0.830, "target pixel\n$K_k\\,[R|t]$\n1920 x 1080", CB[1]),
    ]
    w, h, y = 0.155, 0.40, 0.42
    for x, txt, c in stages:
        box(ax, x, y, w, h, txt, fc=c, ec=c, fs=5.9, tc="white", lw=0)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=c, edgecolor="none", zorder=1))
    for x, _, _ in stages[:-1]:
        arrow(ax, (x + w + 0.001, y + h / 2), (x + w + 0.0075, y + h / 2), lw=1.0)

    ax.annotate("", (0.504, y - 0.035), (0.504 + 0.155, y - 0.035),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.4"))
    ax.text(0.582, y - 0.10, "shared NCore world frame\n(the invariant of the transfer)",
            ha="center", va="top", fontsize=5.9, color="0.32", linespacing=1.3)

    ax.text(0.015, y + h + 0.13, "SOURCE DOMAIN  (measured, FTheta + rolling shutter)",
            fontsize=6.3, color=CB[0], fontweight="bold")
    ax.text(0.667, y + h + 0.13, "TARGET DOMAIN  (proxy pinhole)",
            fontsize=6.3, color=CB[1], fontweight="bold")
    ax.plot([0.008, 0.50], [y + h + 0.10, y + h + 0.10], color=CB[0], lw=0.9)
    ax.plot([0.660, 0.992], [y + h + 0.10, y + h + 0.10], color=CB[1], lw=0.9)

    ax.text(0.5, 0.09,
            "sign / handedness invariants tested before training:  a point 1 m ahead of a camera "
            "must project near the principal point;\n"
            "a point 1 m to camera right must move toward $+u$.  "
            "Ego frame: $+x$ fwd, $+y$ left, $+z$ up.  Camera frame (OpenCV): $+x$ right, $+y$ down, $+z$ fwd.",
            ha="center", va="center", fontsize=5.8, color="0.30", linespacing=1.45)
    save(fig, "fig03_frames")


# ============================================================ Fig 4
def fig4_pipeline():
    fig, ax = plt.subplots(figsize=(COL2, 3.05))
    blank(ax)
    G, A, R = CB[2], CB[4], CB[1]     # admit / conditional / abstain colours

    box(ax, 0.015, 0.60, 0.148, 0.30,
        "NCore clip\n7 x FTheta RGB\nposes, LiDAR\ntracks, masks", fc="#eef2f7", ec=CB[0], fs=6.0)
    box(ax, 0.196, 0.60, 0.148, 0.30,
        "source\nnormalization\nFTheta -> rays\n20 pinhole tiles", fc="#eef2f7", ec=CB[0], fs=6.0)

    box(ax, 0.383, 0.755, 0.20, 0.145,
        "static 2DGS  $G_s$\nmasked RGB + road-LiDAR depth\n12 000 iter, 5 cams x 20 tiles",
        fc="#e9f5ef", ec=G, fs=5.9)
    box(ax, 0.383, 0.585, 0.20, 0.145,
        "temporal sky  $G_y$\nworld-direction cubemap\n7 cams x 18 time slots",
        fc="#e9f5ef", ec=G, fs=5.9)
    box(ax, 0.383, 0.345, 0.20, 0.185,
        "actor experts  $E_j$\nrigid 2DGS per track and source cam\nsource-view temporal holdout:\n"
        "MAE $\\leq$ 0.07, IoU $\\geq$ 0.82,\narea ratio $\\in$ [0.85, 1.20]",
        fc="#fdf3e3", ec=A, fs=5.7)

    arrow(ax, (0.164, 0.75), (0.194, 0.75))
    for yy in (0.828, 0.658):
        arrow(ax, (0.345, 0.75), (0.381, yy), rad=0.10)
    arrow(ax, (0.345, 0.72), (0.381, 0.44), rad=0.14)

    # registry freeze gate
    box(ax, 0.383, 0.235, 0.20, 0.075,
        "REGISTRY FREEZE\n6 accepted, 2 rejected", fc=A, ec=A, fs=5.5, tc="white", lw=0)
    arrow(ax, (0.483, 0.343), (0.483, 0.313))

    # target-side gates
    gx = 0.625
    box(ax, gx, 0.60, 0.155, 0.30,
        "TARGET RIG $R$\n7 rectified pinhole\n$K_k$, $T_{cam\\rightarrow rig}^k$\n1920x1080 @ 30 FPS",
        fc="#fdeee6", ec=R, fs=6.0)
    gates = [("view angle $\\leq$ 25 deg", 0.455),
             ("distance ratio $\\in$ [0.60, 1.60]", 0.365),
             ("$t$ within source window", 0.275),
             ("$\\pm$100 ms alpha fade", 0.185)]
    for txt, yy in gates:
        box(ax, gx, yy, 0.155, 0.072, txt, fc="white", ec=A, fs=5.7, lw=0.7, ls="--")
    arrow(ax, (0.585, 0.272), (gx - 0.003, 0.40), rad=-0.12, color=A)
    for i in range(3):
        arrow(ax, (gx + 0.0775, gates[i][1]), (gx + 0.0775, gates[i + 1][1] + 0.072),
              lw=0.7, color=A)

    # compositor + outputs
    box(ax, 0.825, 0.60, 0.162, 0.30,
        "fail-closed\ncompositor\n$C_{out}=C_{bg}(1-A_e)+C_eA_e$\ndepth-sorted admitted actors",
        fc="#fdeee6", ec=R, fs=5.2)
    arrow(ax, (0.782, 0.75), (0.823, 0.75))
    arrow(ax, (0.782, 0.42), (0.860, 0.598), rad=0.16, color=A)

    outs = [("ADMIT\nactor\ncomposited", G, 0.825),
            ("ABSTAIN\nstatic /\nsky only", R, 0.910)]
    for txt, c, x in outs:
        box(ax, x, 0.40, 0.0745, 0.125, txt, fc=c, ec=c, fs=5.2, tc="white", lw=0)
    arrow(ax, (0.862, 0.598), (0.862, 0.527))
    arrow(ax, (0.947, 0.598), (0.947, 0.527))

    box(ax, 0.825, 0.235, 0.162, 0.115,
        "7 x 299 = 2093 frames\nRGB, depth, alpha,\ninvalid mask, provenance",
        fc="#f4f4f4", ec="0.45", fs=5.8)
    arrow(ax, (0.906, 0.398), (0.906, 0.352))

    ax.text(0.5, 0.115,
            "Two independent freezes protect the evidence chain: the static model is frozen before the road-LiDAR holdout is scored,\n"
            "and the actor registry is frozen before any target frame is rendered.  No target-side measurement can revise either.",
            ha="center", va="center", fontsize=6.0, color="0.28", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="0.82", lw=0.5))
    ax.text(0.015, 0.955, "SOURCE DOMAIN", fontsize=6.5, color=CB[0], fontweight="bold")
    ax.text(0.383, 0.955, "SCENE MODELS  (frozen)", fontsize=6.5, color=G, fontweight="bold")
    ax.text(0.625, 0.955, "TARGET DOMAIN  (gates + abstention)", fontsize=6.5,
            color=R, fontweight="bold")
    ax.plot([0.010, 0.350], [0.938, 0.938], color=CB[0], lw=0.8)
    ax.plot([0.380, 0.590], [0.938, 0.938], color=G, lw=0.8)
    ax.plot([0.620, 0.992], [0.938, 0.938], color=R, lw=0.8)
    ax.set_ylim(0.05, 1.0)
    save(fig, "fig04_pipeline")


if __name__ == "__main__":
    fig1_problem()
    fig2_rig_layout()
    fig3_frames()
    fig4_pipeline()
    print("done")
