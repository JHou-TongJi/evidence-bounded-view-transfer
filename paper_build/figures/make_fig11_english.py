#!/usr/bin/env python
"""Rebuild Fig. 11 with English panel labels.

The delivered comparison frame carries burned-in Chinese labels. This script
reads the ORIGINAL asset (never modifies it), paints over each label plate and
redraws it in English using the same terminology as the manuscript body:
"front wide" / "left cross" / "right cross" for the source cameras and
"front centre" / "side left" / "side right" for the target cameras.

    python make_fig11_english.py
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "05_评价结果", "src",
                   "source_vehicle_vs_target_l4_frame270.png")
OUT = os.path.join(HERE, "fig11_source_to_target.png")

# Label plate geometry measured from the original raster (1920x1080).
BAND_Y = [(10, 49), (370, 409), (730, 769)]     # three panel rows
LEFT_X0, LEFT_ERASE_X1 = 330, 664               # source column
RIGHT_X0, RIGHT_ERASE_X1 = 962, 1392            # target column
PANEL_LEFT_X1, PANEL_RIGHT_X1 = 955, 1601       # panel right edges (content)

PLATE = (45, 45, 45)
INK = (255, 255, 255)
PAD_X, PAD_Y = 8, 4

LABELS = [
    # (row, column, text)
    (0, "L", "Source passenger car: front wide FTheta"),
    (0, "R", "Target L4: front centre pinhole (different mounting)"),
    (1, "L", "Source passenger car: left cross wide FTheta"),
    (1, "R", "Target L4: side left pinhole (different mounting)"),
    (2, "L", "Source passenger car: right cross wide FTheta"),
    (2, "R", "Target L4: side right pinhole (different mounting)"),
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise SystemExit("no usable TrueType font found; tried %s" % FONT_CANDIDATES)


def fit_font(draw, text, max_w, start=25, floor=15):
    """Largest size at which the label still fits inside its panel."""
    for size in range(start, floor - 1, -1):
        f = load_font(size)
        if draw.textlength(text, font=f) <= max_w:
            return f, size
    return load_font(floor), floor


def main():
    if not os.path.exists(SRC):
        raise SystemExit("original asset not found: %s" % SRC)
    im = Image.open(SRC).convert("RGB")
    if im.size != (1920, 1080):
        raise SystemExit("unexpected source size %s; geometry below assumes 1920x1080"
                         % (im.size,))
    d = ImageDraw.Draw(im)

    for row, col, text in LABELS:
        y0, y1 = BAND_Y[row]
        if col == "L":
            x0, erase_x1, panel_x1 = LEFT_X0, LEFT_ERASE_X1, PANEL_LEFT_X1
        else:
            x0, erase_x1, panel_x1 = RIGHT_X0, RIGHT_ERASE_X1, PANEL_RIGHT_X1

        avail = panel_x1 - x0 - 2 * PAD_X
        font, size = fit_font(d, text, avail)
        tw = d.textlength(text, font=font)

        # The new plate must cover the old one completely, so take the wider of
        # the measured original extent and the new text box.
        plate_x1 = max(erase_x1, int(x0 + tw + 2 * PAD_X))
        plate_x1 = min(plate_x1, panel_x1)
        d.rectangle([x0, y0, plate_x1, y1], fill=PLATE)

        ty = y0 + (y1 - y0 - size) / 2 - 1
        d.text((x0 + PAD_X, ty), text, font=font, fill=INK)
        print("  row %d %s  size %2d  plate x %4d..%4d  %s"
              % (row, col, size, x0, plate_x1, text))

    im.save(OUT)
    print("wrote %s (%.1f KB)" % (os.path.basename(OUT),
                                  os.path.getsize(OUT) / 1024))
    print("original untouched: %s" % SRC)


if __name__ == "__main__":
    sys.exit(main())
