#!/usr/bin/env python
"""Build the manuscript as an IEEE Access submission.

    python build_access_paper.py

Same content modules and same build-time consistency checks as
`build_ieee_paper.py`, but laid out with `access_docx.AccessPaper`, which builds
on the official Access-Template-2024.docx rather than synthesising a Transactions
page. Writes a timestamped file plus a stable copy, and refuses to finish if any
cross-reference drifted or any bibliography entry is uncited.
"""
import datetime
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from access_docx import AccessPaper                  # noqa: E402
import paper_refs                                    # noqa: E402
import paper_xref                                    # noqa: E402
import content_intro, content_method                 # noqa: E402
import content_results, content_ablation             # noqa: E402
import content_discussion                            # noqa: E402
import content_backmatter                            # noqa: E402

BUILD = os.path.join(HERE, "build")
STEM = "Cross_Vehicle_7V_IEEE_Access"

PUB_LINE = ("Date of publication xxxx 00, 0000, date of current version "
            "xxxx 00, 0000.")
DOI_LINE = "Digital Object Identifier 10.1109/ACCESS.2026.DOI"

AUTHORS = [("Ruizhi Zhong", "1", ""),
           ("Silei Chen", "1", "")]
AFFILIATIONS = [("1", "Perception Data Reconstruction Team")]
CORRESPONDING = ("Corresponding author: Ruizhi Zhong "
                 "(e-mail: TO BE SUPPLIED BY THE AUTHORS).")
FUNDING = ("FUNDING STATEMENT TO BE SUPPLIED BY THE AUTHORS. "
           "The source recording is the NVIDIA PhysicalAI Autonomous Vehicles "
           "NCore dataset, used under its access terms. The target vehicle rig "
           "is an engineering proxy defined for this study and is not an OEM "
           "calibration of any commercial product.")

RUNNING_AUTHOR = "R. Zhong and S. Chen"
RUNNING_TITLE = "Evidence-Bounded Cross-Vehicle View Transfer"
VOLUME_LINE = "VOLUME XX, 2026"


def main():
    os.makedirs(BUILD, exist_ok=True)
    P = AccessPaper(cache_dir=os.path.join(BUILD, "eqcache"))
    P.set_running_head(RUNNING_AUTHOR, RUNNING_TITLE)
    P.set_footer(VOLUME_LINE)

    P.front_matter(PUB_LINE, DOI_LINE, content_intro.TITLE, AUTHORS,
                   AFFILIATIONS, CORRESPONDING, FUNDING)

    content_intro.build(P, front_matter=False)
    content_method.build(P)
    content_results.build_setup(P)
    content_results.build_results(P)
    content_results.build_results_2(P)
    content_ablation.build(P)
    content_discussion.build(P)

    # ---- build-time consistency checks --------------------------------
    # equation numbers shift when equations move; literal "(n)" references
    # in the prose silently go stale, so refuse to build with any present
    import check_eq_refs
    if check_eq_refs.main() != 0:
        raise AssertionError("hard-coded equation references; see output above")

    content_method.verify_equations(P)
    paper_xref.verify(P)

    orphans = paper_refs.unused()
    if orphans:
        raise AssertionError(
            "bibliography entries never cited in the text: %s" % ", ".join(orphans))

    content_backmatter.acknowledgment(P)
    P.references(paper_refs.entries())
    content_backmatter.biographies(P)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = os.path.join(BUILD, "%s_%s.docx" % (STEM, stamp))
    stable = os.path.join(HERE, "%s.docx" % STEM)
    P.save(versioned)
    shutil.copyfile(versioned, stable)

    print("sections   %d" % P.sec_no)
    print("equations  %d" % P.eq_no)
    print("tables     %d" % P.tab_no)
    print("figures    %d" % P.fig_no)
    print("algorithms %d" % P.alg_no)
    print("references %d (all cited)" % len(paper_refs.REFS))
    print("wrote      %s" % versioned)
    print("wrote      %s" % stable)
    return stable


if __name__ == "__main__":
    main()
