#!/usr/bin/env python
"""Guard against hard-coded equation references in the prose.

Equation numbers shift whenever an equation is inserted or removed. A reference
written as a literal -- "gate (9)", "predicate, (12)" -- silently becomes wrong
and points the reader at an unrelated equation. This has happened twice in this
manuscript: Algorithm 2 pointed at the LiDAR depth loss instead of the
admissibility predicate, and the Appendix B notation table named equation (12)
after the predicate had moved to (13).

Legitimate references are always written as "(%d)" % EQ[...], so any literal
parenthesised one- or two-digit number inside a string in a content module is
suspicious. This scans for them and fails the build.

    python check_eq_refs.py

Add a genuinely non-equation case to ALLOW below if one arises; keep the list
short, and prefer rewording over allow-listing.
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: Literal "(n)" occurrences that are demonstrably not equation references.
ALLOW = {
    # (none at present)
}

STR_LIT = re.compile(r'"([^"\n]*\(\d{1,2}\)[^"\n]*)"')


def main():
    hits = []
    for path in sorted(glob.glob(os.path.join(HERE, "content_*.py"))):
        src = io.open(path, encoding="utf-8").read()
        for m in STR_LIT.finditer(src):
            frag = m.group(1)
            if frag.strip() in ALLOW:
                continue
            line = src[:m.start()].count("\n") + 1
            hits.append((os.path.basename(path), line, frag))

    if hits:
        print("hard-coded equation references found (%d):" % len(hits))
        for f, line, frag in hits:
            print("  %-24s L%-5d %s" % (f, line, frag[:100]))
        print()
        print("Equation numbers shift when equations are added or removed.")
        print('Write these as "(%d)" % EQ["<label>"] so they follow the numbering.')
        return 1

    print("no hard-coded equation references in any content module")
    return 0


if __name__ == "__main__":
    sys.exit(main())
