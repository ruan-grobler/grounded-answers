"""Render the PDF documents in corpus/ from the plain text sources in scripts/pdf_sources/.

The PDFs are committed, so ingestion reads real PDF files. This script only
exists so anyone can see, and regenerate, exactly what is inside them.

Source format:
    title: ...          document title (first line of page 1)
    meta: ...           document control line
    banner: ...         the "fictional document" notice
    # 1 Heading         a numbered section heading
    - item              a bullet line
    ---page---          force a page break
    anything else       a paragraph

Run: .venv/bin/python scripts/make_pdfs.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "scripts" / "pdf_sources"
OUT = ROOT / "corpus"
# Fixed creation date so the files are byte for byte reproducible.
CREATED = dt.datetime(2026, 10, 1, 9, 0, 0, tzinfo=dt.UTC)


def render(source: Path, target: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_creation_date(CREATED)
    pdf.set_producer("grounded-answers scripts/make_pdfs.py")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    title = ""
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("title:"):
            title = line.removeprefix("title:").strip()
            pdf.set_title(title)
            pdf.set_font("Helvetica", "B", 17)
            pdf.multi_cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        elif line.startswith("meta:"):
            pdf.set_font("Helvetica", "", 8.5)
            pdf.multi_cell(0, 4.5, line.removeprefix("meta:").strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        elif line.startswith("banner:"):
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.multi_cell(0, 4.5, line.removeprefix("banner:").strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)
        elif line == "---page---":
            pdf.add_page()
        elif line.startswith("# "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, line.removeprefix("# ").strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(0.5)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.2, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.2, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)
    if not title:
        raise ValueError(f"{source.name} has no title: line")
    pdf.output(str(target))


def main() -> None:
    for source in sorted(SOURCES.glob("*.txt")):
        target = OUT / f"{source.stem}.pdf"
        render(source, target)
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
