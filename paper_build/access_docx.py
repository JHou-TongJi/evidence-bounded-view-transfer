#!/usr/bin/env python
"""IEEE Access two-column DOCX layout engine (python-docx).

Unlike `ieee_docx`, which synthesises the IEEE Transactions look from scratch,
this engine *opens the official IEEE Access 2024 Word template* and builds the
manuscript on top of it. The template's own styles, numbering definitions, page
geometry (8.0 x 10.875 in), IEEE Access header logos and footers are therefore
used verbatim rather than reimplemented, which is what the venue asks for.

Only the body content is replaced. Automatic list numbering inherited from the
template styles is suppressed on every heading, caption and reference so that the
numbers the reader sees are the same ones the build-time cross-reference checks
in `paper_xref` compare against.
"""
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, RGBColor, Twips

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "Access-Template-2024.docx")

SERIF = "Times New Roman"
SANS = "Helvetica"          # the template's own face; Word substitutes Arial
BLACK = RGBColor(0, 0, 0)
ACCESS_BLUE = RGBColor(0x00, 0x62, 0x9B)
ACCESS_GREY = RGBColor(0x58, 0x59, 0x5B)

# Page geometry, in twips, read from the template's own sectPr.
PG_W, PG_H = 11520, 15660
M_TOP, M_BOT, M_LR = 1300, 1040, 740
COL_SPACE = 400

_TEXT_W_TW = PG_W - 2 * M_LR                       # 10040
_COL_W_TW = (_TEXT_W_TW - COL_SPACE) // 2          # 4820

FULL_W = _TEXT_W_TW / 1440.0 - 0.05                # 6.92 in, with a safety strip
COL_W = _COL_W_TW / 1440.0 - 0.05                  # 3.30 in

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


# ----------------------------------------------------------------- low level
def _set_cols(section, num, space_twips=COL_SPACE):
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")


def _kill_numbering(p):
    """Suppress the numPr a template style would otherwise contribute.

    numId 0 is the documented "no numbering" sentinel; setting it on the
    paragraph overrides the style-level definition.
    """
    pPr = p._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:numPr")):
        pPr.remove(old)
    numPr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    nid = OxmlElement("w:numId")
    nid.set(qn("w:val"), "0")
    numPr.append(ilvl)
    numPr.append(nid)
    pPr.insert(0, numPr)
    return p


#: template styles that carry a numPr. With numbering suppressed their list
#: indents are dead space, so they are zeroed; every other style keeps the
#: indentation the template defines (notably PARA_Indent's first line).
NUMBERED_STYLES = {"H1_List (No Space)", "H1_List (Space)", "H2_First",
                   "H2_Cont", "H3", "Fig Caption", "References"}


def _keep_with_next(p):
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    return p


def _cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for name, sz in (("top", top), ("bottom", bottom),
                     ("left", left), ("right", right)):
        el = borders.find(qn("w:" + name))
        if el is None:
            el = OxmlElement("w:" + name)
            borders.append(el)
        if sz is None:
            el.set(qn("w:val"), "nil")
        else:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")


def _cell_pad(cell, top=1.2, bottom=1.2, left=3.0, right=3.0):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcMar")):
        tcPr.remove(old)
    mar = OxmlElement("w:tcMar")
    for name, v in (("top", top), ("bottom", bottom),
                    ("left", left), ("right", right)):
        el = OxmlElement("w:" + name)
        el.set(qn("w:w"), str(int(v * 20)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _superscript(run):
    rPr = run._element.get_or_add_rPr()
    va = OxmlElement("w:vertAlign")
    va.set(qn("w:val"), "superscript")
    rPr.append(va)
    return run


# ----------------------------------------------------------------- math
_TEX_REWRITES = [
    (r"\Bigl", r"\left"), (r"\Bigr", r"\right"),
    (r"\bigl", r"\left"), (r"\bigr", r"\right"),
    (r"\lVert", r"\|"), (r"\rVert", r"\|"),
    (r"\tfrac", r"\frac"),
    (r"\textsf", r"\mathsf"), (r"\text{", r"\mathrm{"),
    (r"\!\left", r"\left"), (r"\!\right", r"\right"),
]


def normalize_tex(tex):
    for a, b in _TEX_REWRITES:
        tex = tex.replace(a, b)
    return tex


class MathRenderer:
    """Render LaTeX-ish math to PNG via matplotlib mathtext, with a disk cache."""

    def __init__(self, cache_dir, dpi=900):
        self.dir = cache_dir
        self.dpi = dpi
        os.makedirs(cache_dir, exist_ok=True)
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"

    def render(self, tex, fontsize=10.0, color="black"):
        tex = normalize_tex(tex)
        key = hashlib.md5(("%s|%.2f|%s|%d" % (tex, fontsize, color, self.dpi))
                          .encode("utf-8")).hexdigest()[:16]
        path = os.path.join(self.dir, "eq_%s.png" % key)
        if os.path.exists(path):
            return path
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, "$%s$" % tex, fontsize=fontsize, color=color)
        fig.savefig(path, dpi=self.dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.012)
        plt.close(fig)
        return path

    def width_in(self, path):
        from PIL import Image
        with Image.open(path) as im:
            return im.width / float(self.dpi)

    def height_in(self, path):
        from PIL import Image
        with Image.open(path) as im:
            return im.height / float(self.dpi)


# ----------------------------------------------------------------- document
class AccessPaper:
    """Builds an IEEE Access manuscript on top of the official 2024 template."""

    def __init__(self, cache_dir, template=TEMPLATE):
        self.doc = Document(template)
        self.math = MathRenderer(cache_dir)
        self.eq_no = 0
        self.eq_labels = {}
        self.tab_no = 0
        self.tab_labels = {}
        self.fig_no = 0
        self.fig_labels = {}
        self.alg_no = 0
        self.sec_no = 0
        self.sub_no = 0
        self.subsub_no = 0
        self._clear_body()
        self._setup_section(self.doc.sections[0], ncols=1, first_page=True)

    # -- template surgery ---------------------------------------------------
    def _clear_body(self):
        """Drop the template's sample content, keeping styles, numbering and the
        header/footer parts the final sectPr points at."""
        body = self.doc.element.body
        sectPr = body.find(qn("w:sectPr"))
        for child in list(body):
            if child is not sectPr:
                body.remove(child)

    def _setup_section(self, section, ncols, first_page=False):
        sectPr = section._sectPr
        # the template's last sectPr is a "continuous" one; the document's first
        # section must be a normal page section or Word ignores the page setup
        for t in sectPr.findall(qn("w:type")):
            sectPr.remove(t)
        if not first_page:
            t = OxmlElement("w:type")
            t.set(qn("w:val"), "continuous")
            sectPr.insert(0, t)

        section.page_width, section.page_height = Twips(PG_W), Twips(PG_H)
        section.top_margin, section.bottom_margin = Twips(M_TOP), Twips(M_BOT)
        section.left_margin, section.right_margin = Twips(M_LR), Twips(M_LR)
        section.header_distance, section.footer_distance = Twips(360), Twips(640)
        _set_cols(section, ncols)

        # first page carries the tall IEEE Access logo, later pages the running
        # head; both header parts come from the template itself
        self._attach_hf(sectPr, "headerReference", "first", "rId9")
        self._attach_hf(sectPr, "headerReference", "default", "rId12")
        self._attach_hf(sectPr, "footerReference", "first", "rId10")
        self._attach_hf(sectPr, "footerReference", "default", "rId10")
        if not sectPr.findall(qn("w:titlePg")):
            tp = OxmlElement("w:titlePg")
            cols = sectPr.find(qn("w:cols"))
            cols.addprevious(tp)
        return section

    @staticmethod
    def _attach_hf(sectPr, tag, kind, rid):
        for el in sectPr.findall(qn("w:" + tag)):
            if el.get(qn("w:type")) == kind:
                sectPr.remove(el)
        el = OxmlElement("w:" + tag)
        el.set(qn("w:type"), kind)
        el.set(qn("r:id"), rid)
        sectPr.insert(0, el)

    # -- running head and footer -------------------------------------------
    def _part(self, rid):
        return self.doc.part.related_parts[rid]

    def set_running_head(self, author_short, title_short):
        """Replace the sample text after the small logo in the page-2+ header."""
        hdr = self._part("rId12").element
        p = hdr.find(qn("w:p"))
        keep = []
        for r in p.findall(qn("w:r")):
            if r.findall(qn("w:drawing")) or r.findall(qn("w:tab")):
                keep.append(r)
            p.remove(r)
        for r in keep:
            p.append(r)
        for text in ("%s: " % author_short, title_short):
            r = OxmlElement("w:r")
            rPr = OxmlElement("w:rPr")
            f = OxmlElement("w:rFonts")
            f.set(qn("w:ascii"), SANS)
            f.set(qn("w:hAnsi"), SANS)
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), "14")
            rPr.append(f)
            rPr.append(sz)
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = text
            r.append(rPr)
            r.append(t)
            p.append(r)

    def set_footer(self, volume_line):
        """`VOLUME xx, 2026` at the left, a live PAGE field at the right."""
        ftr = self._part("rId10").element
        p = ftr.find(qn("w:p"))
        for r in p.findall(qn("w:r")):
            p.remove(r)

        def _rpr():
            rPr = OxmlElement("w:rPr")
            f = OxmlElement("w:rFonts")
            f.set(qn("w:ascii"), SANS)
            f.set(qn("w:hAnsi"), SANS)
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), "12")
            rPr.append(f)
            rPr.append(sz)
            return rPr

        r = OxmlElement("w:r")
        r.append(_rpr())
        t = OxmlElement("w:t")
        t.text = volume_line
        r.append(t)
        p.append(r)

        r = OxmlElement("w:r")
        r.append(_rpr())
        r.append(OxmlElement("w:tab"))
        p.append(r)

        for kind, payload in (("begin", None), ("instr", "PAGE"), ("end", None)):
            r = OxmlElement("w:r")
            r.append(_rpr())
            if kind == "instr":
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = " %s " % payload
                r.append(it)
            else:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                r.append(fc)
            p.append(r)

    # -- generic helpers ----------------------------------------------------
    def _sp(self, style, before=None, after=None, line=None, align=None,
            keep=False, nonum=True):
        """A paragraph in one of the template's own named styles."""
        p = self.doc.add_paragraph(style=style)
        if nonum:
            _kill_numbering(p)
            if style in NUMBERED_STYLES:
                p.paragraph_format.left_indent = Pt(0)
                p.paragraph_format.first_line_indent = Pt(0)
        pf = p.paragraph_format
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        if line is not None:
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(line)
        if align is not None:
            pf.alignment = align
        if keep:
            _keep_with_next(p)
        return p

    def _p(self, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, line=None,
           indent=0.0, container=None, style=None):
        """A bare paragraph, used inside tables and floats where a body style
        would drag in unwanted indents."""
        if container is None:
            p = self.doc.add_paragraph(style=style) if style else self.doc.add_paragraph()
        else:
            p = container.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        if line is None:
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.0
        else:
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = line
        if indent:
            pf.first_line_indent = Inches(indent)
        return p

    def _run(self, p, text, size=10, bold=False, italic=False, caps=False,
             color=None, font=SERIF, spacing=None):
        r = p.add_run(text)
        r.font.name = font
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        r.font.color.rgb = color or BLACK
        if caps:
            r.font.small_caps = True
        rpr = r._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is not None:
            rf.set(qn("w:cs"), font)
        if spacing is not None:
            el = OxmlElement("w:spacing")
            el.set(qn("w:val"), str(int(spacing * 20)))
            rpr.append(el)
        return r

    def rich(self, p, spans, size=10, body_line_pt=12.0):
        """spans: list of (text, style), style in {'', 'i', 'b', 'bi', 'm'}.

        The template's body styles set an *exact* 12 pt leading, which clips any
        inline math taller than the line. A paragraph that carries inline math is
        therefore switched to "at least" leading: ordinary lines keep the 12 pt
        grid and only the lines holding a tall glyph grow.
        """
        has_math = False
        for text, style in spans:
            if style == "m":
                has_math = True
                path = self.math.render(text, fontsize=size * 1.02)
                h = self.math.height_in(path)
                r = p.add_run()
                r.add_picture(path, height=Inches(min(h, 0.30)))
            else:
                self._run(p, text, size=size,
                          bold="b" in style, italic="i" in style)
        if has_math:
            pf = p.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.AT_LEAST
            pf.line_spacing = Pt(body_line_pt)
        return p

    # -- front matter -------------------------------------------------------
    def front_matter(self, pub_line, doi_line, title, authors, affiliations,
                     corresponding, funding):
        """The IEEE Access first-page block.

        `authors` is a list of (name, superscript, postnominal); `affiliations`
        a list of (superscript, text).
        """
        p = self._sp("DOP")
        self._run(p, pub_line, size=7, font=SANS)

        p = self._sp("DOI")
        self._run(p, doi_line, size=6, italic=True)

        p = self._sp("Paper Title")
        self._run(p, title, size=22, bold=True, font=SANS, color=ACCESS_BLUE)

        p = self._sp("AU", after=0)
        for i, (name, sup, post) in enumerate(authors):
            if i:
                if i == len(authors) - 1:
                    sep = " and " if len(authors) == 2 else ", and "
                else:
                    sep = ", "
                self._run(p, sep, size=10, bold=True, font=SANS)
            self._run(p, name, size=10, bold=True, font=SANS)
            if sup:
                _superscript(self._run(p, sup, size=10, bold=True, font=SANS))
            if post:
                self._run(p, ", " + post, size=10, bold=True, font=SANS)

        for j, (sup, text) in enumerate(affiliations):
            last = (j == len(affiliations) - 1)
            p = self._sp("PI" if last else "PI_No Space", after=0)
            p.paragraph_format.first_line_indent = Pt(0)
            if sup:
                _superscript(self._run(p, sup, size=7))
            self._run(p, text, size=7)

        p = self._sp("CA", before=5, after=5)
        self._run(p, corresponding, size=7.5)

        p = self._sp("footnote text", after=27)
        p.paragraph_format.first_line_indent = Pt(0)
        self._run(p, funding, size=8)

    def title_block(self, title, authors, affiliation, note=None):
        """Compatibility shim for the Transactions content modules."""
        self.front_matter(
            "Date of publication xxxx 00, 0000, date of current version "
            "xxxx 00, 0000.",
            "Digital Object Identifier 10.1109/ACCESS.2026.DOI",
            title, [(authors, "", "")], [("", affiliation)],
            "Corresponding author: (e-mail: ).", note or "")

    def abstract(self, text):
        p = self._sp("Abstract")
        self._run(p, "ABSTRACT ", size=10, bold=True, font=SANS,
                  color=ACCESS_BLUE)
        self._run(p, text, size=10)

    def index_terms(self, text):
        p = self._sp("IT")
        self._run(p, "INDEX TERMS ", size=10, bold=True, font=SANS,
                  color=ACCESS_BLUE)
        self._run(p, text, size=10)

    # -- section machinery --------------------------------------------------
    def _continuous(self, ncols):
        s = self.doc.add_section(WD_SECTION.CONTINUOUS)
        return self._setup_section(s, ncols)

    def start_two_column(self):
        """Deferred: in IEEE Access the abstract and index terms are still part
        of the full-width first-page block, so the break is emitted lazily, just
        before the first body element."""
        self._pending_2col = True

    def _flush_2col(self):
        if getattr(self, "_pending_2col", False):
            self._pending_2col = False
            self._continuous(2)

    def _span_open(self):
        self._continuous(1)

    def _span_close(self):
        self._continuous(2)

    # -- headings -----------------------------------------------------------
    def section(self, title):
        self._flush_2col()
        self.sec_no += 1
        self.sub_no = 0
        style = "H1_List (No Space)" if self.sec_no == 1 else "H1_List (Space)"
        p = self._sp(style, keep=True)
        self._run(p, "%s. %s" % (ROMAN[self.sec_no], title.upper()),
                  size=9, bold=True, font=SANS, color=ACCESS_BLUE)
        return self.sec_no

    def subsection(self, title):
        self.sub_no += 1
        self.subsub_no = 0
        style = "H2_First" if self.sub_no == 1 else "H2_Cont"
        p = self._sp(style, keep=True)
        self._run(p, "%s. %s" % (chr(64 + self.sub_no), title.upper()),
                  size=9, bold=True, italic=True, font=SANS, color=ACCESS_GREY)
        return "%s-%s" % (ROMAN[self.sec_no], chr(64 + self.sub_no))

    def subsubsection(self, title):
        self.subsub_no += 1
        p = self._sp("H3", keep=True)
        self._run(p, "%d) %s" % (self.subsub_no, title.upper()),
                  size=9, font=SANS, color=ACCESS_GREY)
        return p

    def unnumbered_section(self, title):
        self._flush_2col()
        p = self._sp("H1", keep=True)
        self._run(p, title.upper(), size=9, bold=True, font=SANS,
                  color=ACCESS_BLUE)
        return p

    def appendix(self, letter, title):
        self._flush_2col()
        p = self._sp("H1", keep=True)
        self._run(p, "APPENDIX %s" % letter, size=9, bold=True, font=SANS,
                  color=ACCESS_BLUE)
        p2 = self._sp("H1", before=0, keep=True)
        self._run(p2, title.upper(), size=9, bold=True, font=SANS,
                  color=ACCESS_BLUE)

    # -- body ---------------------------------------------------------------
    def para(self, spans, first=False, size=10):
        self._flush_2col()
        p = self._sp("PARA" if first else "PARA_Indent")
        if isinstance(spans, str):
            self._run(p, spans, size=size)
        else:
            self.rich(p, spans, size=size)
        return p

    def bullet(self, spans, size=9.5):
        p = self._sp("PARA")
        p.paragraph_format.left_indent = Inches(0.20)
        p.paragraph_format.first_line_indent = Inches(-0.11)
        if isinstance(spans, str):
            self._run(p, "\u2022  " + spans, size=size)
        else:
            self._run(p, "\u2022  ", size=size)
            self.rich(p, spans, size=size)
        return p

    # -- equations ----------------------------------------------------------
    def equation(self, tex, number=True, width_limit=COL_W - 0.55, fontsize=10.5,
                 label=None):
        self.eq_no += 1 if number else 0
        if label:
            if label in self.eq_labels:
                raise ValueError("duplicate equation label: " + label)
            self.eq_labels[label] = self.eq_no
        path = self.math.render(tex, fontsize=fontsize)
        w, h = self.math.width_in(path), self.math.height_in(path)
        if w > width_limit:
            h *= width_limit / w
            w = width_limit

        t = self.doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = (COL_W - 0.40, 0.40)
        for c, w_ in zip(t.rows[0].cells, widths):
            c.width = Inches(w_)
            _cell_borders(c)
            _cell_pad(c, top=3.0, bottom=3.0, left=0, right=0)
        for col, w_ in zip(t.columns, widths):
            col.width = Inches(w_)
        c0 = t.rows[0].cells[0].paragraphs[0]
        c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._tight(c0)
        c0.add_run().add_picture(path, width=Inches(w), height=Inches(h))
        c1 = t.rows[0].cells[1].paragraphs[0]
        c1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self._tight(c1)
        if number:
            self._run(c1, "(%d)" % self.eq_no, size=10)
        self._gap(2)
        return self.eq_no

    @staticmethod
    def _tight(p, before=0, after=0, line=1.0):
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = line
        return p

    def _gap(self, pts):
        p = self.doc.add_paragraph()
        self._tight(p, 0, 0, 0.35)
        p.paragraph_format.space_after = Pt(pts)
        return p

    def equation_cases(self, lhs, cases, number=True, fontsize=10.0, label=None):
        self.eq_no += 1 if number else 0
        if label:
            self.eq_labels[label] = self.eq_no
        n = len(cases)
        t = self.doc.add_table(rows=n, cols=4)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        w = [1.00, 0.10, 1.78, 0.40]
        for col, w_ in zip(t.columns, w):
            col.width = Inches(w_)
        for r in range(n):
            for j, cell in enumerate(t.rows[r].cells):
                cell.width = Inches(w[j])
                _cell_borders(cell)
                _cell_pad(cell, top=0.8, bottom=0.8, left=0, right=0)
                p = cell.paragraphs[0]
                self._tight(p)
                p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if j == 0 else
                               WD_ALIGN_PARAGRAPH.CENTER if j == 1 else
                               WD_ALIGN_PARAGRAPH.LEFT if j == 2 else
                               WD_ALIGN_PARAGRAPH.RIGHT)
        mid = (n - 1) // 2
        path = self.math.render(lhs + " =", fontsize=fontsize)
        p = t.rows[mid].cells[0].paragraphs[0]
        p.add_run().add_picture(path, height=Inches(self.math.height_in(path)))
        for r in range(n):
            ch = {0: "\u23a7", n - 1: "\u23a9"}.get(r, "\u23aa")
            if n == 3 and r == 1:
                ch = "\u23a8"
            self._run(t.rows[r].cells[1].paragraphs[0], ch, size=fontsize + 1)
        for r, (val, cond) in enumerate(cases):
            pv = self.math.render(val, fontsize=fontsize)
            pc = self.math.render(cond, fontsize=fontsize)
            cell = t.rows[r].cells[2].paragraphs[0]
            cell.add_run().add_picture(pv, height=Inches(self.math.height_in(pv)))
            self._run(cell, "   ")
            cell.add_run().add_picture(pc, height=Inches(self.math.height_in(pc)))
        if number:
            self._run(t.rows[mid].cells[3].paragraphs[0], "(%d)" % self.eq_no,
                      size=10)
        self._gap(2)
        return self.eq_no

    # -- figures ------------------------------------------------------------
    def figure(self, path, caption, width=COL_W, span=False, label=None):
        self._flush_2col()
        self.fig_no += 1
        if label:
            if label in self.fig_labels:
                raise ValueError("duplicate figure label: " + label)
            self.fig_labels[label] = self.fig_no
        if span:
            width = FULL_W
            self._span_open()
        p = self._p(WD_ALIGN_PARAGRAPH.CENTER, before=4, after=1.5)
        p.add_run().add_picture(path, width=Inches(width))
        c = self._sp("Fig Caption", after=8,
                     align=WD_ALIGN_PARAGRAPH.JUSTIFY if len(caption) > 95
                     else WD_ALIGN_PARAGRAPH.LEFT)
        self._run(c, "FIGURE %d. " % self.fig_no, size=7, bold=True, font=SANS,
                  color=ACCESS_BLUE)
        self._run(c, caption, size=7, bold=True, font=SANS)
        if span:
            self._span_close()
        return self.fig_no

    # -- tables -------------------------------------------------------------
    def table(self, caption, header, rows, widths=None, span=False,
              note=None, size=7.5, align=None, label=None):
        """IEEE Access table: `TABLE n` and the caption above, in the template's
        Table Title style; three horizontal rules; header row repeated on break."""
        self._flush_2col()
        self.tab_no += 1
        if label:
            if label in self.tab_labels:
                raise ValueError("duplicate table label: " + label)
            self.tab_labels[label] = ROMAN[self.tab_no]
        total = FULL_W if span else COL_W
        if span:
            self._span_open()

        cp = self._sp("Table Title", before=8, after=1, keep=True)
        self._run(cp, "TABLE %s" % ROMAN[self.tab_no], size=8, caps=True)
        cp2 = self._sp("Table Title", before=0, after=2, keep=True)
        self._run(cp2, caption, size=8, caps=True)

        ncol = len(header)
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        if widths is None:
            widths = [1.0 / ncol] * ncol
        ws = [Inches(total * w) for w in widths]
        for col, w_ in zip(t.columns, ws):
            col.width = w_

        for j, cell in enumerate(t.rows[0].cells):
            cell.width = ws[j]
            _cell_pad(cell, top=1.6, bottom=1.6)
            _cell_borders(cell, top=12, bottom=6)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self._tight(p)
            self._run(p, header[j], size=size, bold=True, font=SANS)
        _repeat_header(t.rows[0])

        for i, row in enumerate(rows):
            last = (i == len(rows) - 1)
            for j, cell in enumerate(t.rows[i + 1].cells):
                cell.width = ws[j]
                _cell_pad(cell, top=1.0, bottom=1.0)
                _cell_borders(cell, bottom=12 if last else None)
                p = cell.paragraphs[0]
                a = (align or ["l"] * ncol)[j]
                p.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT,
                               "c": WD_ALIGN_PARAGRAPH.CENTER,
                               "r": WD_ALIGN_PARAGRAPH.RIGHT}[a]
                self._tight(p)
                txt = row[j]
                if isinstance(txt, tuple):
                    self._run(p, txt[0], size=size, bold="b" in txt[1],
                              italic="i" in txt[1])
                else:
                    self._run(p, str(txt), size=size)

        if note:
            np_ = self._p(WD_ALIGN_PARAGRAPH.LEFT, before=1, after=7)
            self._run(np_, note, size=7)
        else:
            self._gap(7)
        if span:
            self._span_close()
        return self.tab_no

    # -- algorithms ---------------------------------------------------------
    def algorithm(self, title, io_lines, steps, span=False, size=8):
        self.alg_no += 1
        total = FULL_W if span else COL_W
        if span:
            self._span_open()
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        t.columns[0].width = Inches(total)
        cell = t.rows[0].cells[0]
        cell.width = Inches(total)
        _cell_pad(cell, top=3.0, bottom=3.0, left=2.0, right=2.0)
        _cell_borders(cell, top=12, bottom=12)

        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self._tight(p, 0, 1)
        self._run(p, "Algorithm %d  " % self.alg_no, size=size + 0.5, bold=True,
                  font=SANS)
        self._run(p, title, size=size + 0.5, bold=True, font=SANS)

        rule = cell.add_paragraph()
        self._tight(rule, 0, 0, 0.28)
        pPr = rule._p.get_or_add_pPr()
        bdr = OxmlElement("w:pBdr")
        bot = OxmlElement("w:bottom")
        bot.set(qn("w:val"), "single")
        bot.set(qn("w:sz"), "6")
        bot.set(qn("w:color"), "000000")
        bdr.append(bot)
        pPr.append(bdr)

        for label, text in io_lines:
            q = cell.add_paragraph()
            self._tight(q, 1.5, 0)
            q.paragraph_format.left_indent = Inches(0.42)
            q.paragraph_format.first_line_indent = Inches(-0.42)
            self._run(q, label + ": ", size=size, italic=True)
            self._run(q, text, size=size)

        for k, step in enumerate(steps, 1):
            q = cell.add_paragraph()
            self._tight(q, 0.6, 0)
            depth = len(step) - len(step.lstrip(" "))
            q.paragraph_format.left_indent = Inches(0.26 + 0.12 * (depth // 2))
            q.paragraph_format.first_line_indent = Inches(-0.26)
            self._run(q, "%2d:  " % k, size=size)
            self._run(q, step.strip(), size=size)

        self._gap(7)
        if span:
            self._span_close()
        return self.alg_no

    # -- back matter --------------------------------------------------------
    def acknowledgment(self, text):
        self.unnumbered_section("Acknowledgment")
        p = self._sp("PARA")
        self._run(p, text, size=10)

    def references(self, entries):
        self.unnumbered_section("References")
        for i, e in enumerate(entries, 1):
            p = self._sp("References", after=0.6)
            p.paragraph_format.left_indent = Inches(0.22)
            p.paragraph_format.first_line_indent = Inches(-0.22)
            self._run(p, "[%d]\t" % i, size=8)
            self._run(p, e, size=8)

    def biography(self, name, paragraphs, photo=None, photo_w=1.0):
        """Author biography block: 1 x 1.25 in photo, name in bold small caps,
        then the biography paragraphs."""
        p = self._sp("AU_Bios")
        if photo and os.path.exists(photo):
            p.add_run().add_picture(photo, width=Inches(photo_w),
                                    height=Inches(photo_w * 1.25))
        self._run(p, name.upper() + " ", size=8, bold=True)
        if paragraphs:
            self._run(p, paragraphs[0], size=8)
        for extra in paragraphs[1:]:
            q = self._sp("AU_Bios_No Space")
            self._run(q, extra, size=8)

    def save(self, path):
        self.doc.save(path)
        return path
