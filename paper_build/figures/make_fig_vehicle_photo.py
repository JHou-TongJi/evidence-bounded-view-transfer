#!/usr/bin/env python
"""Prepare the deployment-vehicle photograph for publication.

The manuscript shows the intended deployment platform. Before publication the
image is de-identified: the licence plate, the vendor wordmark and the vendor
badge are blurred, so the figure illustrates the vehicle class without carrying
a registration number or a commercial endorsement. The numbered locker labels
are left intact -- they are a functional feature of the vehicle, not an
identifier.

The original photograph is never modified; this reads it and writes a separate
figure file.

    python make_fig_vehicle_photo.py
"""
import os
import sys

from PIL import Image, ImageFilter, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "photo", "微信图片_20260901182743_2_53.jpg")
OUT = os.path.join(HERE, "fig06_vehicle.png")

EXPECT_SIZE = (1706, 1279)

#: Regions to obscure, in original pixel coordinates (left, top, right, bottom).
REDACT = [
    ((313, 481, 412, 612), "vendor wordmark on cab side"),
    ((288, 936, 370, 1008), "licence plate"),
    ((322, 700, 358, 758), "vendor badge"),
]

#: Crop that removes empty sky and foreground paving.
CROP = (250, 170, 1430, 1210)


def main():
    if not os.path.exists(SRC):
        raise SystemExit("photograph not found: %s" % SRC)
    im = Image.open(SRC).convert("RGB")
    if im.size != EXPECT_SIZE:
        raise SystemExit("unexpected source size %s; REDACT boxes assume %s"
                         % (im.size, EXPECT_SIZE))

    for (box, what) in REDACT:
        patch = im.crop(box)
        # heavy blur then a light flat overlay, so nothing is recoverable by
        # sharpening and the redaction reads as deliberate
        patch = patch.resize((max(1, patch.width // 22),
                             max(1, patch.height // 22)), Image.BILINEAR)
        patch = patch.resize(
            (box[2] - box[0], box[3] - box[1]), Image.NEAREST)
        patch = patch.filter(ImageFilter.GaussianBlur(7))
        im.paste(patch, box)
        print("  redacted %-32s %s" % (what, box))

    im = im.crop(CROP)
    im.save(OUT)
    print("wrote %s  %dx%d  (%.1f KB)"
          % (os.path.basename(OUT), im.width, im.height,
             os.path.getsize(OUT) / 1024))
    print("original untouched: %s" % SRC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
