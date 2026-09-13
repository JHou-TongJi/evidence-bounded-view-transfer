#!/usr/bin/env python
"""Prepare Figs. 12-13 from the delivered qualitative frames.

Figs. 12 and 13 are delivered assets, not plots, so this script does not draw
them -- it copies them from the evidence package and asserts their provenance
(source path, exact size, SHA-256 prefix) so that every figure in the manuscript
has a runnable generator and a checkable origin. Fig. 11 needs relabelling and
therefore has its own script, make_fig11_english.py.

    python make_figs_photo.py
"""
import hashlib
import os
import shutil
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "05_评价结果", "src"))

# (source filename, figure filename, expected size)
ASSETS = [
    ("target_l4_front_center_frame020.png", "fig12_front_center_1080p.png", (1920, 1080)),
    ("target_l4_7v_mosaic_frame000.png", "fig13_mosaic_7v.png", (1920, 1080)),
]


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    problems = []
    for src_name, fig_name, want_size in ASSETS:
        src = os.path.join(SRC_DIR, src_name)
        dst = os.path.join(HERE, fig_name)
        if not os.path.exists(src):
            problems.append("missing source asset: %s" % src)
            continue
        with Image.open(src) as im:
            size = im.size
        if size != want_size:
            problems.append("%s is %s, expected %s" % (src_name, size, want_size))
            continue
        shutil.copyfile(src, dst)
        print("  %-32s -> %-30s %s  sha256:%s"
              % (src_name, fig_name, "%dx%d" % size, sha16(dst)))

    if problems:
        for p in problems:
            print("  ERROR " + p)
        return 1
    print("wrote %d delivered figures, provenance verified" % len(ASSETS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
