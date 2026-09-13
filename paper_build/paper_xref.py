#!/usr/bin/env python
"""Build-time-checked cross-references for tables and figures.

The prose refers to floats through these constants. `verify` compares them with
the labels the document actually assigned, so inserting, removing or reordering a
float fails the build instead of silently producing a wrong reference -- the
class of defect that left the previous draft with a table sequence running
1-7, 15, 16, 8-14, 17, 18.
"""

TAB = {
    "position": "I",       # II-E  positioning against related systems
    "projection": "II",    # IV-A  projection-chain stages
    "objective": "III",    # IV-C  objective terms
    "rig": "IV",           # V-A   target extrinsics
    "platform": "V",       # V-A   target platform and sensor specification
    "impl": "VI",          # V-B   implementation parameters
    "layers": "VII",       # V-D   four evaluation layers
    "holdout": "VIII",     # VI-B  road holdout
    "holdout_cam": "IX",   # VI-B  decomposition by source camera
    "registry": "X",       # VI-C  admitted and rejected experts
    "abstention": "XI",    # VI-E  per-camera abstention
    "screening": "XII",    # VI-F  screened pipelines
    "failures": "XIII",    # VI-H  residual failure classes
    "sensitivity": "XIV",  # VII-E displacement vs target envelope scale
    "notation": "XV",      # Appendix B notation
    "evidence": "XVI",     # Appendix C evidence-to-claim map
    "gates": "XVII",       # Appendix D gate constants and their provenance
}

FIG = {
    "problem": 1,          # III   cross-vehicle transfer concept
    "pipeline": 2,         # IV    system overview
    "frames": 3,           # IV-A  projection chain
    "rig": 4,              # V-A   rig layout (plan + azimuth)
    "platform": 5,         # V-A   platform elevations + sensor spec
    "deployment": 6,       # V-A   deployment vehicle photograph
    "holdout_dist": 7,     # VI-B  per-view error distribution
    "holdout_cam": 8,      # VI-B  error by source camera
    "registry": 9,         # VI-C  admission scatter
    "outside": 10,          # VI-D  outside-actor perturbation
    "ledger": 11,           # VI-E  activation ledger
    "iou": 12,             # VI-E  alpha IoU
    "qualitative": 13,     # VI-H  source-to-target comparison
    "front": 14,           # VI-H  native 1080p frame
    "mosaic": 15,          # VI-H  seven-view mosaic
    "peak_gallery": 16,    # VI-H  per-camera delivered frame at peak actor
    "regimes": 17,         # VI-H  the two abstention regimes, one camera
    "lateral": 18,         # VI-H  both side cameras across the clip
}


def T(key):
    """Roman numeral of a labelled table, for use inside prose."""
    return TAB[key]


def Fg(key):
    """Number of a labelled figure, for use inside prose."""
    return FIG[key]


def verify(P):
    problems = []
    if P.tab_labels != TAB:
        problems.append("tables:\n    expected %s\n    actual   %s"
                        % (sorted(TAB.items(), key=lambda kv: kv[1]),
                           sorted(P.tab_labels.items(), key=lambda kv: kv[1])))
    if P.fig_labels != FIG:
        problems.append("figures:\n    expected %s\n    actual   %s"
                        % (sorted(FIG.items(), key=lambda kv: kv[1]),
                           sorted(P.fig_labels.items(), key=lambda kv: kv[1])))
    if problems:
        raise AssertionError("cross-reference drift:\n  " + "\n  ".join(problems))
