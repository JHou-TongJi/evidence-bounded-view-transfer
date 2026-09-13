#!/usr/bin/env python
"""Render a DOCX to PDF with Word, and optionally rasterise pages for the
format check.

    python docx_to_pdf.py Cross_Vehicle_7V_IEEE_Access.docx [--pages]
"""
import os
import sys

import win32com.client as win32

HERE = os.path.dirname(os.path.abspath(__file__))
WD_FORMAT_PDF = 17


def convert(docx, pdf=None):
    docx = os.path.abspath(docx)
    pdf = pdf or os.path.splitext(docx)[0] + ".pdf"
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(docx, ReadOnly=False)
        doc.Fields.Update()
        doc.Repaginate()
        n = doc.ComputeStatistics(2)          # wdStatisticPages
        words = doc.ComputeStatistics(0)      # wdStatisticWords
        doc.SaveAs(pdf, FileFormat=WD_FORMAT_PDF)
        doc.Close(SaveChanges=0)
    finally:
        word.Quit()
    return pdf, n, words


def rasterise(pdf, outdir, dpi=110):
    import pymupdf
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.startswith("p") and f.endswith(".png"):
            os.remove(os.path.join(outdir, f))
    d = pymupdf.open(pdf)
    for i, page in enumerate(d, 1):
        page.get_pixmap(dpi=dpi).save(os.path.join(outdir, "p%03d.png" % i))
    return d.page_count


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Cross_Vehicle_7V_IEEE_Access.docx"
    pdf, pages, words = convert(os.path.join(HERE, src))
    print("pdf    %s" % pdf)
    print("pages  %d" % pages)
    print("words  %d" % words)
    if "--pages" in sys.argv:
        n = rasterise(pdf, os.path.join(HERE, "build", "pages_access"))
        print("raster %d pages" % n)
