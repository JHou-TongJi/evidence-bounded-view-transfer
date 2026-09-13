#!/usr/bin/env python
"""Export every table and figure in the manuscript, with its Python source.

The manuscript is generated, so each table is already produced by Python -- but
the definitions live inside the section modules, which makes them hard to find
and impossible to open in a spreadsheet. This script runs the same build against
a recording harness and emits, for every table:

    build/tables/TABLE_<roman>_<label>.csv    the data, editable anywhere
    build/tables/ALL_TABLES.md                every table rendered for review
    build/tables/SOURCE_INDEX.md              table/figure -> file, line, function

and for every figure, the generator script and function that draws it. Nothing
here re-derives a number: it captures exactly what the built document contains,
so the CSVs cannot drift from the manuscript.

    python make_tables.py
"""
import csv
import inspect
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ieee_docx import IEEEPaper, ROMAN                     # noqa: E402
import paper_refs, paper_xref                              # noqa: E402
import content_intro, content_method                       # noqa: E402
import content_results, content_ablation, content_discussion  # noqa: E402

OUT = os.path.join(HERE, "build", "tables")

#: Which script and function draws each figure file.
FIGURE_SOURCE = {
    "fig01_problem.png":        ("figures/make_figs_schematic.py", "fig1_problem"),
    "fig02_rig_layout.png":     ("figures/make_figs_schematic.py", "fig2_rig_layout"),
    "fig03_frames.png":         ("figures/make_figs_schematic.py", "fig3_frames"),
    "fig04_pipeline.png":       ("figures/make_figs_schematic.py", "fig4_pipeline"),
    "fig05_platform.png":       ("figures/make_fig_platform.py", "main"),
    "fig06_vehicle.png":        ("figures/make_fig_vehicle_photo.py", "main"),
    "fig05_holdout_distribution.png": ("figures/make_figs_data.py", "fig5_holdout_distribution"),
    "fig06_holdout_by_camera.png":    ("figures/make_figs_data.py", "fig6_holdout_by_camera"),
    "fig07_registry_acceptance.png":  ("figures/make_figs_data.py", "fig7_registry_acceptance"),
    "fig08_activation_ledger.png":    ("figures/make_figs_data.py", "fig8_activation_ledger"),
    "fig09_alpha_iou.png":            ("figures/make_figs_data.py", "fig9_alpha_iou"),
    "fig10_outside_actor.png":        ("figures/make_figs_data.py", "fig10_outside_actor"),
    "fig11_source_to_target.png":     ("figures/make_fig11_english.py", "main"),
    "fig12_front_center_1080p.png":   ("figures/make_figs_photo.py", "main"),
    "fig13_mosaic_7v.png":            ("figures/make_figs_photo.py", "main"),
}


def _caller():
    """Locate the content module, function and line that issued this float."""
    for fr in inspect.stack()[2:]:
        f = os.path.basename(fr.filename)
        if f.startswith("content_"):
            return f, fr.function, fr.lineno
    return "?", "?", 0


class RecordingPaper(IEEEPaper):
    """Builds the document normally while capturing every table and figure."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured_tables = []
        self.captured_figures = []

    def table(self, caption, header, rows, **kw):
        src = _caller()
        n = super().table(caption, header, rows, **kw)
        self.captured_tables.append(dict(
            number=ROMAN[n], label=kw.get("label", ""), caption=caption,
            header=list(header),
            rows=[[c[0] if isinstance(c, tuple) else str(c) for c in r] for r in rows],
            note=kw.get("note"), span=bool(kw.get("span")), source=src))
        return n

    def figure(self, path, caption, **kw):
        src = _caller()
        n = super().figure(path, caption, **kw)
        self.captured_figures.append(dict(
            number=n, label=kw.get("label", ""), file=os.path.basename(path),
            caption=caption, span=bool(kw.get("span")), source=src))
        return n


def build_recording():
    P = RecordingPaper(cache_dir=os.path.join(HERE, "build", "eqcache"))
    content_intro.build(P)
    content_method.build(P)
    content_results.build_setup(P)
    content_results.build_results(P)
    content_results.build_results_2(P)
    content_ablation.build(P)
    content_discussion.build(P)
    content_method.verify_equations(P)
    paper_xref.verify(P)
    P.references(paper_refs.entries())
    return P


def main():
    os.makedirs(OUT, exist_ok=True)
    P = build_recording()

    # ---- one CSV per table -------------------------------------------
    for t in P.captured_tables:
        name = "TABLE_%s_%s.csv" % (t["number"], t["label"] or "unlabelled")
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["# Table %s. %s" % (t["number"], t["caption"])])
            w.writerow(t["header"])
            w.writerows(t["rows"])
            if t["note"]:
                w.writerow([])
                w.writerow(["# Note: " + t["note"]])
        print("  %-44s %d x %d" % (name, len(t["rows"]), len(t["header"])))

    # ---- all tables as one readable Markdown file ---------------------
    md = ["# All manuscript tables",
          "",
          "Exported from the built document by `make_tables.py`. Each table is also "
          "written as a CSV in this directory. The *source* line gives the module, "
          "function and line that defines it.",
          ""]
    for t in P.captured_tables:
        md.append("## Table %s. %s" % (t["number"], t["caption"]))
        md.append("")
        md.append("*source*: `%s` → `%s()` line %d%s"
                  % (t["source"][0], t["source"][1], t["source"][2],
                     " · full-width float" if t["span"] else ""))
        md.append("")
        md.append("| " + " | ".join(t["header"]) + " |")
        md.append("|" + "|".join(["---"] * len(t["header"])) + "|")
        for r in t["rows"]:
            md.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
        if t["note"]:
            md.append("")
            md.append("*Note*: " + t["note"])
        md.append("")
    with open(os.path.join(OUT, "ALL_TABLES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---- source index for every float --------------------------------
    idx = ["# Source index: every figure and table",
           "",
           "Generated by `make_tables.py`. Every float in the manuscript is produced "
           "by Python; this maps each one to the code that produces it.",
           "",
           "## Figures", "",
           "| Fig. | File | Generator script | Function | Caption defined in |",
           "|---|---|---|---|---|"]
    for fg in P.captured_figures:
        gen = FIGURE_SOURCE.get(fg["file"], ("(unregistered)", "?"))
        idx.append("| %d | `%s` | `%s` | `%s()` | `%s` → `%s()` line %d |"
                   % (fg["number"], fg["file"], gen[0], gen[1],
                      fg["source"][0], fg["source"][1], fg["source"][2]))
    idx += ["", "## Tables", "",
            "| Table | Label | Defined in | Function | Line | CSV |",
            "|---|---|---|---|---|---|"]
    for t in P.captured_tables:
        idx.append("| %s | `%s` | `%s` | `%s()` | %d | `TABLE_%s_%s.csv` |"
                   % (t["number"], t["label"], t["source"][0], t["source"][1],
                      t["source"][2], t["number"], t["label"] or "unlabelled"))
    idx += ["", "## Data layer", "",
            "No table or figure hard-codes a measured value. Every number is read "
            "from the evidence package through `paper_data.py`, whose "
            "`provenance()` lists each claim key, its source file and its value. "
            "`verify_claims.py` re-derives them independently.", ""]
    with open(os.path.join(OUT, "SOURCE_INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx))

    unregistered = [fg["file"] for fg in P.captured_figures
                    if fg["file"] not in FIGURE_SOURCE]
    print("\ntables exported : %d" % len(P.captured_tables))
    print("figures indexed : %d" % len(P.captured_figures))
    if unregistered:
        print("FIGURES WITH NO REGISTERED GENERATOR: %s" % unregistered)
        return 1
    print("every figure has a registered generator script")
    print("wrote %s" % os.path.relpath(OUT, HERE))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
