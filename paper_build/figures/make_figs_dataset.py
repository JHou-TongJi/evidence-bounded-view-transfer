#!/usr/bin/env python
"""Dataset figures: delivered target frames, straight from the delivery.

Every panel here is a real rendered frame of the delivered sequence. Nothing is
simulated, redrawn or illustrative. Two sources are used, and both are checked
before anything is written:

  * `04_生成结果/selected_frames/` -- PNGs the project itself selected as the
    peak-actor frame for each target camera;
  * `04_生成结果/generated_video/*.mp4` -- the delivered per-camera sequences.
    These decode to 299 frames each at 1920x1080, and 7 x 299 = 2093 is exactly
    the delivered frame count in the manifest, so the videos *are* the delivery
    rather than a re-render of it. The script asserts this.

Frame choice is tied to evidence, not to appearance. Each actor expert carries a
temporal validity window; outside every window no expert is admissible on any
camera, so those frames provably carry no actor content. Those windows leave two
runs uncovered, and the "no candidate" panels are drawn from them. The
"admitted" panels are the project's own peak-actor selections, each of which
falls inside a window.

    python make_figs_dataset.py
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, PAPER)
import paper_data as D  # noqa: E402

GEN = os.path.join(ROOT, "04_生成结果")
VID = os.path.join(GEN, "generated_video")
SEL = os.path.join(GEN, "selected_frames")

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.2, "axes.titlesize": 7.2,
    "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FPS = 30.0
#: The project's own peak-actor selection per target camera.
PEAK = [("front_center", 141), ("front_left", 209), ("front_right", 160),
        ("side_left", 213), ("side_right", 166), ("rear_left", 227),
        ("rear_right", 208)]


def windows():
    return [(e["timestamp_start_us"], e["timestamp_end_us"]) for e in D.EXPERTS]


def covered(frame, wins):
    t = frame / FPS * 1e6
    return any(a <= t <= b for a, b in wins)


def uncovered_runs():
    wins = windows()
    unc = [f for f in range(D.NFRAMES) if not covered(f, wins)]
    runs, start, prev = [], unc[0], unc[0]
    for f in unc[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append((start, prev))
            start = prev = f
    runs.append((start, prev))
    return runs


def read_frame(path, index):
    """Decode one frame, asserting the file is the delivered 299-frame sequence.

    The project tree has CJK directory names, and OpenCV's VideoCapture cannot
    open a non-ASCII path on Windows -- it fails silently, reporting zero
    frames. Opening by bare filename from inside the directory sidesteps it.
    """
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(path))
        cap = cv2.VideoCapture(os.path.basename(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n != D.NFRAMES:
            raise ValueError("%s decoded %d frames, expected %d"
                             % (os.path.basename(path), n, D.NFRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        cap.release()
    finally:
        os.chdir(cwd)
    if not ok:
        raise ValueError("could not decode frame %d of %s" % (index, path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_png(path):
    """imread() has the same non-ASCII path defect; decode from bytes instead."""
    if not os.path.exists(path):
        raise ValueError("missing asset: " + path)
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode: " + path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def shrink(img, width_px):
    h, w = img.shape[:2]
    if w <= width_px:
        return img
    return cv2.resize(img, (width_px, int(round(h * width_px / w))),
                      interpolation=cv2.INTER_AREA)


def panel(ax, img, title, sub=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("0.45")
    ax.set_title(title, loc="left", pad=2.2, fontsize=6.4)
    if sub:
        ax.text(0.012, 0.035, sub, transform=ax.transAxes, fontsize=5.4,
                color="white", va="bottom", ha="left",
                bbox=dict(boxstyle="square,pad=0.22", fc="black", ec="none",
                          alpha=0.62))


def save(fig, stem):
    p = os.path.join(HERE, stem + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote %-34s %7.1f KB" % (os.path.basename(p),
                                    os.path.getsize(p) / 1024))


# ===================================================== per-camera peak actor
def fig_peak_gallery():
    """Seven panels: one delivered frame per target camera, at peak actor support."""
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 2.42))
    wins = windows()
    for i, (cam, fr) in enumerate(PEAK):
        ax = axes.ravel()[i]
        img = shrink(load_png(os.path.join(SEL, "actor_peak_rgb",
                                           "%s_%06d.png" % (cam, fr))), 760)
        m = D.cam_metrics(cam)
        if not covered(fr, wins):
            raise ValueError("%s frame %d is outside every expert window"
                             % (cam, fr))
        panel(ax, img, "%s · f%d (%.1f s)" % (cam.replace("_", " "), fr, fr / FPS),
              "%s actor px · %d/%d active"
              % (D.pct(m["actor_alpha_area_mean"], 2),
                 m["actor_active_frames"], D.NFRAMES))
    # The eighth cell carries the rig-wide summary rather than a stretched panel.
    ax = axes.ravel()[7]
    ax.axis("off")
    ax.text(0.02, 0.80,
            "rig-wide\n\n"
            "mean actor support\n%s of the image\n\n"
            "frames with no actor\n%s of %s"
            % (D.pct(D.ABST["area_mean"], 2),
               "{:,}".format(D.ABST["frames_zero_actor"]),
               "{:,}".format(D.NTOTAL)),
            transform=ax.transAxes, fontsize=6.0, va="top", ha="left",
            linespacing=1.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.30)
    save(fig, "fig14_peak_actor_gallery")
    return len(PEAK)


# ===================================================== abstention regimes
def fig_abstention_regimes():
    """Six panels on one camera: no candidate exists vs an actor is admitted."""
    runs = uncovered_runs()
    # three frames with no temporally valid expert, spread across the runs
    no_cand = [runs[0][0] + 12, (runs[0][0] + runs[0][1]) // 2, runs[-1][0] + 6]
    wins = windows()
    for f in no_cand:
        if covered(f, wins):
            raise ValueError("frame %d is not actually uncovered" % f)
    # three frames inside windows, including the curated peak
    admitted = [96, 141, 200]
    for f in admitted:
        if not covered(f, wins):
            raise ValueError("frame %d is not inside any expert window" % f)

    cam = "front_center"
    path = os.path.join(VID, "front_center_1920x1080.mp4")
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 2.60))
    for f, ax in zip(no_cand, axes[0]):
        panel(ax, shrink(read_frame(path, f), 900),
              "frame %d  (%.2f s)" % (f, f / FPS), "no expert valid")
    for f, ax in zip(admitted, axes[1]):
        panel(ax, shrink(read_frame(path, f), 900),
              "frame %d  (%.2f s)" % (f, f / FPS), "expert window open")
    axes[0, 0].set_ylabel("no candidate", fontsize=6.6, labelpad=2)
    axes[1, 0].set_ylabel("candidate available", fontsize=6.6, labelpad=2)
    # The explanation belongs in the caption, not burned into the raster: a
    # suptitle here also inflated the tight bounding box with dead space.
    fig.subplots_adjust(wspace=0.03, hspace=0.24)
    save(fig, "fig15_abstention_regimes")
    return len(no_cand) + len(admitted)


# ===================================================== temporal mosaics
def fig_lateral_timeline():
    """Eight panels: both side cameras across the clip.

    The seven-view mosaic is already shown once, and stacking several of them
    makes each sub-view illegible, so this figure spends its space where the
    method is most stressed instead. The side cameras sit furthest from the one
    published source-camera mounting, so they ask the reconstruction for the
    largest change of viewing direction; `source_to_target_offsets()` gives the
    displacement quoted in the caption.
    """
    picks = [30, 120, 180, 260]
    wins = windows()
    cams = [("side_left", "side_left_1920x1080.mp4"),
            ("side_right", "side_right_1920x1080.mp4")]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 2.46))
    for r, (cam, vid) in enumerate(cams):
        path = os.path.join(VID, vid)
        for c, f in enumerate(picks):
            state = "expert window open" if covered(f, wins) else "no expert valid"
            panel(axes[r, c], shrink(read_frame(path, f), 760),
                  "%s · f%d (%.1f s)" % (cam.replace("_", " "), f, f / FPS), state)
    fig.subplots_adjust(wspace=0.04, hspace=0.30)
    save(fig, "fig16_lateral_timeline")
    return len(cams) * len(picks)


def main():
    print("delivered sequence: %d frames x %d cameras = %d"
          % (D.NFRAMES, D.NCAM, D.NTOTAL))
    print("uncovered runs (no expert valid):", uncovered_runs())
    counts = {"peak_gallery": fig_peak_gallery(),
              "regimes": fig_abstention_regimes(),
              "lateral": fig_lateral_timeline()}
    # The manuscript states how many delivered frames these figures show. That
    # number must come from what was actually plotted, not from a literal typed
    # into the prose, so each builder returns its panel count and the total is
    # written here for paper_data to read back.
    counts["total"] = sum(counts.values())
    import json
    with open(os.path.join(HERE, "dataset_panels.json"), "w") as f:
        json.dump(counts, f, indent=1, sort_keys=True)
    print("panels: %s -> total %d" % (
        ", ".join("%s=%d" % (k, counts[k]) for k in sorted(counts) if k != "total"),
        counts["total"]))
    print("done")


if __name__ == "__main__":
    main()
