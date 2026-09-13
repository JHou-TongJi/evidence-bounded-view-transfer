#!/usr/bin/env python
"""Fresh, context-naive review of the manuscript via the Codex CLI.

This calls `codex exec` as a subprocess. The earlier transports are both gone:
the MCP server mode does not exist in this Codex build, and the direct HTTP call
to the relay's Responses API needed a credential this script had to hunt for in
~/.codex/auth.json, which the CLI rewrites and sometimes deletes out from under a
running session. The CLI reads its own config.toml, so routing, provider and
credentials live in one place that Codex itself maintains.

`codex exec` starts a new session on every invocation -- resuming is a separate
subcommand -- so REVIEWER_BIAS_GUARD holds by construction: no threadId is
carried, no prior review is in scope, and `--ephemeral` keeps the session off
disk so a later run cannot pick it up.

The manuscript goes in on stdin rather than as an argument. Windows caps a
command line near 32k characters and a typeset paper's extracted text runs to
several hundred thousand, so passing it as argv would truncate the manuscript
silently -- the reviewer would score whatever fitted.

    python run_review.py <out.json> [--pdf FILE] [--pages N] [--dpi N]
                         [--venue "MDPI World Electric Vehicle Journal"]

The venue is a parameter because it is a review criterion, not decoration.
A verification run judged a WEVJ manuscript against IEEE Access standards
because the venue was hard-coded here, and the resulting score answered a
question nobody asked.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

SP = os.path.dirname(os.path.abspath(__file__))
PAPER = r"H:\ARIS\papers\paper07\paper"

# Fixed deliberately. The reviewer model is part of what makes a score series
# comparable across rounds; changing it mid-series would make the progression
# meaningless, so it is not a command-line argument.
MODEL = "gpt-6-astra"

PROMPT = """You are reviewing a manuscript submitted to %(venue)s. This is a \
fresh, zero-context review. There are no prior review rounds to consider. Judge \
the paper only from the material below.

The manuscript is supplied as (a) the complete extracted text of the typeset \
PDF, page by page, and (b) rasterised images of every page, so you can assess \
the visual presentation directly.

Act as a senior reviewer for %(venue)s working in autonomous driving, neural \
rendering and sensor simulation. Provide:

1. **Overall Score** (1-10, where 6 = weak accept, 7 = accept)
2. **Summary** (2-3 sentences)
3. **Strengths** (bullet list, ranked)
4. **Weaknesses** (bullet list, ranked: CRITICAL > MAJOR > MINOR)
5. **For each CRITICAL/MAJOR weakness**: a specific, actionable fix
6. **Missing References** (if any)
7. **Visual Review** (from the page images):
   - Figure quality: readable? labels legible? colours distinguishable in greyscale?
   - Figure-caption alignment: does each caption match its figure?
   - Layout: orphaned headers, awkward page breaks, floats far from their references?
   - Table formatting: aligned columns, consistent decimals, best results marked?
   - Visual consistency: same colour scheme across all figures?
8. **Verdict**: Ready for submission? Yes / Almost / No

Focus on: theoretical rigour, claims-versus-evidence alignment, whether the \
evaluation actually supports what the abstract promises, writing clarity, \
self-containedness, notation consistency, and visual presentation quality.

Be specific and be hard on it. If the evidence does not support a claim, say so \
plainly and name the claim.

Answer with the review itself. Do not inspect the working directory, run \
commands, or ask questions; everything you need is below.

=== MANUSCRIPT TEXT ===
"""


def read_pdf(pdf, n_pages, dpi, imgdir):
    """Extract the text AND the page rasters from the same open document.

    These used to come from different places: the text from a file written by a
    separate step, the images from the PDF. Nothing forced the two to describe
    the same build, so a reviewer could be shown one manuscript's prose beside
    another's pages. Reading both from one handle makes that impossible.
    """
    import pymupdf
    doc = pymupdf.open(pdf)
    text, paths = [], []
    for i in range(doc.page_count):
        text.append("\n=== PAGE %d ===\n" % (i + 1) + doc[i].get_text())
        if i < n_pages:
            p = os.path.join(imgdir, "page_%03d.jpg" % (i + 1))
            doc[i].get_pixmap(dpi=dpi).save(p, jpg_quality=72)
            paths.append(p)
    return "".join(text), paths, doc.page_count


def run_codex(prompt, images, out_last):
    """One `codex exec` run. Returns (returncode, stdout, stderr).

    The executable is resolved rather than named. npm installs codex on Windows
    as codex.CMD beside an extensionless shell shim, and CreateProcess appends
    only .exe when resolving a bare name -- so passing "codex" fails with
    WinError 2 even though the command works in a shell.
    """
    exe = shutil.which("codex") or "codex"
    cmd = [exe, "exec",
           "--skip-git-repo-check",
           "--ephemeral",                  # leave no session for a later run
           "-m", MODEL,
           "-s", "read-only"]              # a reviewer has nothing to write
    for p in images:                       # in page order; the reviewer is told
        cmd += ["-i", p]                   # the pages are sequential
    cmd += ["-o", out_last,
            "-"]                           # prompt arrives on stdin
    return subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=5400)


def main():
    args = [a for a in sys.argv[1:]]
    dst = os.path.join(SP, "review.json")
    if args and not args[0].startswith("-"):
        dst = args[0]
    npages, dpi = 40, 100
    venue = "IEEE Access"
    pdf = os.path.join(PAPER, "Cross_Vehicle_7V_IEEE_Access.pdf")
    for flag, cast in (("--pages", int), ("--dpi", int), ("--pdf", str),
                       ("--venue", str)):
        if flag in args:
            v = cast(args[args.index(flag) + 1])
            if flag == "--pages":
                npages = v
            elif flag == "--dpi":
                dpi = v
            elif flag == "--venue":
                venue = v
            else:
                pdf = v
    if not os.path.exists(pdf):
        sys.exit("no such PDF: %s" % pdf)
    if not shutil.which("codex"):
        sys.exit("the codex CLI is not on PATH; this reviewer calls it directly")

    tmp = tempfile.mkdtemp(prefix="review_pages_")
    try:
        text, images, npg = read_pdf(pdf, npages, dpi, tmp)
        prompt = (PROMPT % {"venue": venue}) + text
        print("reviewing %s for %s: %d pages, %d chars, %d page images, model %s"
              % (os.path.basename(pdf), venue, npg, len(text),
                 len(images), MODEL))

        out_last = dst.replace(".json", ".md")
        # The relay behind the CLI intermittently drops large multimodal
        # requests. That is a transport failure, not a review outcome, so retry
        # rather than record a missing review.
        r = None
        for attempt in range(1, 4):
            try:
                r = run_codex(prompt, images, out_last)
            except subprocess.TimeoutExpired:
                print("attempt %d: timed out after 90 min" % attempt)
                r = None
            else:
                if r.returncode == 0 and os.path.exists(out_last) \
                        and os.path.getsize(out_last) > 0:
                    break
                print("attempt %d: exit %s" % (attempt, r.returncode))
                if r.stderr:
                    print("   stderr: %s" % r.stderr.strip()[:400])
            if attempt < 3:
                wait = 20 * attempt
                print("   retrying in %ds" % wait)
                time.sleep(wait)

        if r is None or r.returncode != 0 or not os.path.exists(out_last):
            print("reviewer unreachable after 3 attempts -- no score recorded")
            return 1

        txt = io.open(out_last, encoding="utf-8").read()
        io.open(dst, "w", encoding="utf-8").write(json.dumps(
            {"model": MODEL, "venue": venue, "pdf": pdf, "pages": npg,
             "page_images": len(images),
             "review": txt}, ensure_ascii=False, indent=1))
        print("review chars: %d  -> %s" % (len(txt), os.path.basename(out_last)))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
