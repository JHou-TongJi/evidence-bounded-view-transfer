#!/usr/bin/env python
"""Format compliance check for the IEEE Access rendering.

Measures the ink bounding box of every rendered page against the Access text
area (the DOCX analogue of an Overfull hbox) and reports the page budget.

    python docx_to_pdf.py Cross_Vehicle_7V_IEEE_Access.docx --pages
    python format_check_access.py
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = sorted(glob.glob(os.path.join(HERE, "build", "pages_access", "p*.png")))

# from Access-Template-2024.docx: 11520 x 15660 twips, margins 1300/1040/740/740
PAGE_W, PAGE_H = 8.0, 10.875
ML = MR = 740 / 1440.0
MT, MB = 1300 / 1440.0, 1040 / 1440.0
HEADER_TOP = 360 / 1440.0        # the running head sits above the text block
FOOTER_BOT = 640 / 1440.0
TARGET = 25
TOL_IN = 0.03


def ink_bbox(path):
    im = Image.open(path).convert("L")
    a = np.asarray(im)
    dpi = im.width / PAGE_W
    dark = a < 245
    cols, rows = dark.any(axis=0), dark.any(axis=1)
    if not cols.any():
        return None
    x0, x1 = np.argmax(cols), len(cols) - 1 - np.argmax(cols[::-1])
    y0, y1 = np.argmax(rows), len(rows) - 1 - np.argmax(rows[::-1])
    return (x0 / dpi, y0 / dpi, x1 / dpi, y1 / dpi)


def _assert_fresh():
    """Refuse to grade rasters older than the PDF they claim to depict.

    docx_to_pdf.py only rewrites build/pages_access/ when passed --pages. Run
    without it and this check silently scored the previous build's pages: it
    reported 26 pages against a 27-page PDF, which is a pass that means nothing.
    Staleness must be an error, not a quieter kind of success.
    """
    pdf = os.path.join(HERE, "Cross_Vehicle_7V_IEEE_Access.pdf")
    if not os.path.exists(pdf):
        return
    newest = max(os.path.getmtime(p) for p in PAGES)
    if os.path.getmtime(pdf) > newest + 1:
        sys.exit("rendered pages are older than the PDF; "
                 "rerun: python docx_to_pdf.py Cross_Vehicle_7V_IEEE_Access.docx --pages")


def main():
    if not PAGES:
        print("no rendered pages found; run docx_to_pdf.py --pages first")
        return 1
    _assert_fresh()

    print("== page geometry (IEEE Access) ==")
    print("  page  %.3f x %.3f in" % (PAGE_W, PAGE_H))
    print("  text  x %.3f-%.3f in, y %.3f-%.3f in"
          % (ML, PAGE_W - MR, MT, PAGE_H - MB))
    print("  running head from y %.3f, footer to y %.3f"
          % (HEADER_TOP, PAGE_H - FOOTER_BOT))

    problems = []
    for p in PAGES:
        bbox = ink_bbox(p)
        n = os.path.basename(p)
        if bbox is None:
            problems.append("%s: blank page" % n)
            continue
        x0, y0, x1, y1 = bbox
        over = []
        if x0 < ML - TOL_IN:
            over.append("left by %.3f in" % (ML - x0))
        if x1 > PAGE_W - MR + TOL_IN:
            over.append("right by %.3f in" % (x1 - (PAGE_W - MR)))
        if y0 < HEADER_TOP - TOL_IN:
            over.append("top by %.3f in" % (HEADER_TOP - y0))
        if y1 > PAGE_H - FOOTER_BOT + TOL_IN:
            over.append("bottom by %.3f in" % (y1 - (PAGE_H - FOOTER_BOT)))
        if over:
            problems.append("%s: ink outside the page block - %s"
                            % (n, "; ".join(over)))

    print("\n== overflow check (%d pages) ==" % len(PAGES))
    if problems:
        for q in problems:
            print("  BLOCK  " + q)
    else:
        print("  no page bleeds past a margin")

    print("\n== page budget ==")
    print("  %d pages; target was about %d" % (len(PAGES), TARGET))
    print("  IEEE Access has no hard page limit; over 10 pages incurs the "
          "overlength charge")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
