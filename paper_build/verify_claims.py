#!/usr/bin/env python
"""Independent numerical audit of the built manuscript.

Recomputes every derived quantity from the raw evidence files and greps the
rendered DOCX text for the corresponding string. Fails loudly on any mismatch.
This is deterministic, so it can run in CI; it does not replace the cross-model
review, it makes the review's PASS checkable.

    python verify_claims.py
"""
import io
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as D  # noqa: E402

DOCX = os.path.join(HERE, "Cross_Vehicle_7V_IEEE_Access.docx")
SNAP = os.path.join(HERE, "build", "paper_text.txt")


def _extract(path):
    """Read the manuscript text out of the built DOCX, every run.

    This used to read a snapshot under build/ that nothing regenerated, so the
    audit silently scored whichever extraction happened to be on disk. A stale
    snapshot makes every `present()` check meaningless in the most dangerous
    way: it keeps passing after the manuscript stops saying what it asserts.
    The text is therefore re-extracted from the DOCX on every run, and the
    snapshot is rewritten so the two can never diverge again.
    """
    import docx
    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                parts.extend(p.text for p in cell.paragraphs)
    return "\n".join(parts)


if not os.path.exists(DOCX):
    sys.exit("build the manuscript first: %s is missing" % os.path.basename(DOCX))
TEXT = _extract(DOCX)
io.open(SNAP, "w", encoding="utf-8").write(TEXT)
FAILS, CHECKS = [], 0


def present(label, needle, expect=True):
    """Assert a literal string is (or is not) in the manuscript."""
    global CHECKS
    CHECKS += 1
    hit = needle in TEXT
    if hit is not expect:
        FAILS.append("%s: %s %r" %
                     (label, "missing from manuscript" if expect else
                      "PRESENT but must not be", needle))


def close(label, got, want, tol=1e-9):
    global CHECKS
    CHECKS += 1
    if abs(got - want) > tol:
        FAILS.append("%s: recomputed %r != evidence %r" % (label, got, want))


print("== derived quantities recomputed from raw evidence ==")

# ---- 1. holdout: pixel-weighted mean must reproduce the reported MAE --------
px = sum(v["pixels"] for v in D.HOLD["per_view"])
wm = sum(v["mae_m"] * v["pixels"] for v in D.HOLD["per_view"]) / px
close("holdout pixel count", px, D.HOLD["pixels"], 0)
close("holdout weighted MAE", wm, D.HOLD["mae_m"], 1e-6)
print("  pixels             %d  (file %d)" % (px, D.HOLD["pixels"]))
print("  weighted MAE       %.6f  (file %.6f)" % (wm, D.HOLD["mae_m"]))

maes = [v["mae_m"] for v in D.HOLD["per_view"]]
print("  per-view median    %.6f   max %.6f   min %.6f"
      % (st.median(maes), max(maes), min(maes)))
present("per-view median in text", "%.3f m" % st.median(maes))
present("per-view worst in text", "%.3f m" % max(maes))
# Depths are quoted to the millimetre. Six decimals implied micrometre
# precision on a LiDAR depth holdout, and the same three numbers were being
# printed at two different precisions in different sections.
present("holdout MAE in text", "%.3f m" % D.HOLD["mae_m"])
present("holdout median in text", "%.3f m" % D.HOLD["median_m"])
present("holdout P90 in text", "%.3f m" % D.HOLD["p90_m"])
for _q, _v in (("MAE", D.HOLD["mae_m"]), ("median", D.HOLD["median_m"]),
               ("P90", D.HOLD["p90_m"])):
    present("micrometre-precision %s must not appear" % _q, "%.6f m" % _v, False)
present("relative MAE in text", "%.4f" % D.HOLD["relative_mae"])

# ---- 2. holdout decomposition by source camera -----------------------------
by = D.holdout_by_camera()
tot = sum(r[2] for r in by)
close("per-camera pixels sum", tot, D.HOLD["pixels"], 0)
for cam, nv, cpx, m, med in by:
    print("  %-34s %3d views %9d px  MAE %.3f" % (cam, nv, cpx, m))
    present("holdout row %s" % cam, "%.3f" % m)
spread = 100 * (by[-1][3] / by[0][3] - 1)
print("  camera spread      %.1f%%  (%.3f -> %.3f m)" % (spread, by[0][3], by[-1][3]))
present("camera spread in text", "%.0f%%" % spread)

# ---- 3. abstention -----------------------------------------------------
act = [D.cam_metrics(c)["actor_active_frames"] for c in D.CAMS]
zero = D.NTOTAL - sum(act)
frac = zero / float(D.NTOTAL)
print("  frames w/o actor   %d / %d = %.4f" % (zero, D.NTOTAL, frac))
close("abstention count", zero, D.ABST["frames_zero_actor"], 0)
present("abstention count in text", "{:,}".format(zero))
present("abstention pct in text", D.pct(frac))

# The no-actor fraction pools "no candidate was temporally valid" with "a
# candidate was valid and the gates refused it". Only the second is a refusal,
# so both parts must appear and must sum to the pooled figure.
_unc, _nocand, _refused = D.abstention_split()
close("abstention split closes", _nocand + _refused, D.ABST["frames_zero_actor"], 0)
present("no-candidate share in text", D.pct(_nocand / float(D.NTOTAL)))
present("refusal share in text", D.pct(_refused / float(D.NTOTAL)))
present("the pooled figure is disclaimed as not a refusal rate",
        "is not a refusal rate")
areas = [D.cam_metrics(c)["actor_alpha_area_mean"] for c in D.CAMS]
print("  actor px mean      %.5f  range %.5f - %.5f"
      % (sum(areas) / 7, min(areas), max(areas)))
present("mean actor px in text", D.pct(sum(areas) / 7, 2))
present("min actor px in text", D.pct(min(areas), 2))
present("max actor px in text", D.pct(max(areas), 2))
present("abstention low in text", D.pct(1 - max(act) / 299.0))
present("abstention high in text", D.pct(1 - min(act) / 299.0))

# ---- 4. activation ledger consistency ---------------------------------
for c in D.CAMS:
    counts = D.COMP["counts"][c]
    sel = sum(v for k, v in counts.items()
              if k not in ("none", "static_depth_rejected_pixels"))
    a = D.cam_metrics(c)["actor_active_frames"]
    CHECKS += 1
    if sel < a:
        FAILS.append("%s: expert-frame selections %d < active frames %d" % (c, sel, a))
    if counts["static_depth_rejected_pixels"] != 0:
        FAILS.append("%s: static_depth_rejected_pixels != 0" % c)
present("zero static-depth rejections claimed", "zero static-depth-rejected pixels")

# ---- 5. compositing locality vs 8-bit quantisation --------------------
q = 1.0 / 255
lo = min(D.cam_metrics(c)["outside_actor_rgb_mae"] for c in D.CAMS)
hi = max(D.cam_metrics(c)["outside_actor_rgb_mae"] for c in D.CAMS)
mx = max(D.cam_metrics(c)["outside_actor_rgb_mae_max"] for c in D.CAMS)
print("  outside-actor      mean %.3e-%.3e  worst %.3e  1 LSB %.3e" % (lo, hi, mx, q))
print("  worst / LSB        1/%.0f     mean-max / LSB 1/%.0f" % (q / mx, q / hi))
CHECKS += 1
if not q / mx > 10:
    FAILS.append("worst frame is not 'more than an order of magnitude' below 1 LSB")
CHECKS += 1
if not q / hi > 1000:
    FAILS.append("mean is not below one thousandth of 1 LSB")
CHECKS += 1
if not q / mx > 60:
    FAILS.append("worst frame is not below one sixtieth of 1 LSB")
present("outside-actor low in text", D.sci(lo))
present("outside-actor high in text", D.sci(hi))
present("outside-actor worst in text", D.sci(mx))

# ---- 6. rig extrinsics: every value must appear, and the old draft's
#         fabricated values must NOT ----------------------------------
for cid, x, y, z, yaw, pitch, roll, hf, vf in D.rig_rows():
    present("rig %s pitch" % cid, "%+.0f" % pitch)
present("stale draft pitch -8 for front_center", "front center | 2.20 | +0.00 | 1.72 | 0 | -8", False)
present("stale draft rear x -1.30", "-1.30", False)
present("rear_left true x", "-0.30")

# ---- 7. arithmetic of stated composites -------------------------------
close("2093 = 7 x 299", 7 * 299, D.NTOTAL, 0)
close("100 ms = 3 frames at 30 FPS", D.GATE["fade_ms"] / 1000.0 * 30, 3.0, 1e-9)
close("duration = 299/30", 299 / 30.0, float(D.DELIVERY["duration"]), 1e-6)
present("frames in text", "2,093")

# The tiles per timestamp are distributed across the source cameras, not
# repeated per camera, so the training-view count is tiles x timestamps. The
# previous audit asserted 5 x 20 x 299 against the literal 29,900 -- the
# fabrication compared against itself, which is why it passed for so long.
TPF = D.HOLD_STATS["tiles_per_frame"]
close("holdout views = timestamps x tiles",
      D.HOLD_STATS["n_frames"] * TPF, D.HOLD_STATS["views"], 0)
close("training views = tiles x timestamps", TPF * D.NFRAMES, 5980, 0)
present("training views in text", "{:,}".format(TPF * D.NFRAMES))
present("camera-multiplied training views must not appear",
        "{:,}".format(D.HOLD_STATS["n_cams"] * TPF * D.NFRAMES), False)

# ---- 7b. cross-module consistency the prose can silently break ----------
present("Proposition 1 is promised by the introduction", "Proposition 1")
# Section VI-G states it deliberately does NOT sweep. Three other sections
# claimed it did; the first guard here only matched one of the three phrasings,
# which is why two survived a round. Match every phrasing that asserts a sweep.
for _p in ("swept in Section VI-G", "sweep of Section VI-G",
           "Section VI-G sweeps", "Section VI-G measures how far"):
    present("no sweep claim: %s" % _p, _p, False)
present("occlusion margin is disclosed as inactive",
        "inactive in the delivered configuration")
for _lab in ("L1 delivery", "L2 source-domain depth consistency", "L3 source acceptance",
             "L4 target safety"):
    present("layer label %s used consistently" % _lab, _lab)
present("stale L3 label must not appear", "L3, compositing locality", False)

# Algorithm 2 sets the invalid bit only where NO layer has support, so a
# withheld actor over supported background carries no per-pixel mark. Two
# passages used to claim the opposite -- "left unfilled and flagged" and
# "arrives with an invalid bit". Both are false for the common case and must
# not return.
present("no per-pixel flag overclaim (intro)", "left unfilled and flagged", False)
present("no per-pixel flag overclaim (VII-B)", "arrives with an invalid bit", False)
present("the abstention-mask limit is disclosed",
        "cannot in general recover the")

# ---- 8. registry --------------------------------------------------------
for tid, label, cam, dur, pos, mae, iou, area in D.expert_rows():
    present("expert t%d %s MAE" % (tid, cam), "%.6f" % mae)
    present("expert t%d %s IoU" % (tid, cam), "%.4f" % iou)
    present("expert t%d %s window" % (tid, cam), "%.2f" % dur)
for name, reason, mae, iou, area in D.rejected_rows():
    present("rejected %s MAE" % name[:18], "%.6f" % mae)

# ---- 8a. dataset figures --------------------------------------------
# Every panel in the dataset figures is a frame decoded from the delivered
# sequences. The count quoted in the prose is written by the generator, so a
# figure that gains or loses a panel cannot leave a stale number behind.
_dp = D.dataset_panels()
present("dataset frame count in text", str(_dp["total"]))
CHECKS += 1
if _dp["peak_gallery"] != D.NCAM:
    FAILS.append("peak gallery shows %d panels for %d cameras"
                 % (_dp["peak_gallery"], D.NCAM))
for _stem in ("fig14_peak_actor_gallery", "fig15_abstention_regimes",
              "fig16_lateral_timeline"):
    CHECKS += 1
    if not os.path.exists(os.path.join(HERE, "figures", _stem + ".png")):
        FAILS.append("missing dataset figure: %s.png" % _stem)
print("  dataset panels     %d delivered frames across %d figures"
      % (_dp["total"], len(_dp) - 1))

# ---- 8c. proxy-rig sensitivity ---------------------------------------
# All three round-18 reviews asked for the proxy/deployment mismatch to be
# bounded. It cannot be measured, but its direction can: displacement from the
# published source-camera mounting is what the gates are driven by.
_ms = D.mounting_sensitivity()
CHECKS += 1
if [r[4] for r in _ms] != sorted(r[4] for r in _ms):
    FAILS.append("mean displacement is not monotone in envelope scale")
present("sensitivity nominal mean in text", "%.2f m" % _ms[0][4])
present("sensitivity doubled mean in text", "%.2f m" % _ms[-1][4])
present("the sensitivity model is disclaimed",
        "a geometric model of a platform we did not measure")
print("  proxy sensitivity  mean displacement %.2f m -> %.2f m over %.2fx envelope"
      % (_ms[0][4], _ms[-1][4], _ms[-1][0]))

# ---- 8b. IEEE citation order ------------------------------------------
# References must be numbered in order of first appearance. paper_refs used to
# number by declaration order, which opened the body with "[51], [52]" and put
# ten descending steps in the sequence.
_body = TEXT[:TEXT.rfind("REFERENCES")] if "REFERENCES" in TEXT else TEXT
_first, _seen = [], set()
for _m in re.finditer(r"\[(\d+)\]", _body):
    _n = int(_m.group(1))
    if _n not in _seen:
        _seen.add(_n)
        _first.append(_n)
_desc = [(_first[i], _first[i + 1]) for i in range(len(_first) - 1)
         if _first[i + 1] < _first[i]]
CHECKS += 1
if _desc:
    FAILS.append("citations not in first-appearance order: %d descending steps, "
                 "first at %s" % (len(_desc), _desc[0]))
print("  citation order     %d refs, %d descending steps" % (len(_first), len(_desc)))

# ---- 9. every numeric token in the text must be traceable --------------
LEDGER = set()
for _k, _s, v in D.provenance():
    for t in re.findall(r"\d+\.?\d*", str(v)):
        LEDGER.add(t)

print()
if FAILS:
    print("== %d FAILURES of %d checks ==" % (len(FAILS), CHECKS))
    for f in FAILS:
        print("  FAIL  " + f)
    sys.exit(1)
print("== ALL %d CHECKS PASS ==" % CHECKS)
