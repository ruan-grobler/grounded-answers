from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from grounded_answers.ingest import IngestError, load_corpus, parse_markdown, parse_pdf


def _pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for lines in pages:
        pdf.add_page()
        for line in lines:
            pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(path))


def test_markdown_front_matter_sections_and_notice(mini_corpus: Path) -> None:
    doc = parse_markdown(mini_corpus / "fees.md", mini_corpus.parent)
    assert doc.doc_id == "fees"
    assert doc.title == "Fees"
    assert doc.kind == "markdown"
    assert doc.path == "corpus/fees.md"
    assert doc.meta["notice"] == "Fictional document."
    assert [s.heading for s in doc.sections] == ["Cancellations", "Price list"]
    price = doc.sections[1].paragraphs[0]
    assert price.lines is True
    assert price.text == "- Check-up: R650\n- Scale and polish: R890"


def test_html_comment_stays_one_paragraph(mini_corpus: Path) -> None:
    doc = parse_markdown(mini_corpus / "staff.md", mini_corpus.parent)
    newsletter = next(s for s in doc.sections if s.heading == "Newsletter")
    assert len(newsletter.paragraphs) == 2
    assert newsletter.paragraphs[1].text.startswith("<!--")
    assert newsletter.paragraphs[1].text.endswith("-->")


def test_markdown_without_front_matter_uses_h1(tmp_path: Path) -> None:
    path = tmp_path / "plain.md"
    path.write_text("# Plain title\n\nIntro text.\n\n## Part\n\nBody.\n", encoding="utf-8")
    doc = parse_markdown(path, tmp_path)
    assert doc.doc_id == "plain"
    assert doc.title == "Plain title"
    assert [s.heading for s in doc.sections] == ["Introduction", "Part"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("---\nid: x\n", "not closed"),
        ("---\n- a\n---\n# T\n\n## S\n\nx\n", "mapping"),
        ("## Only a section\n\ntext\n", "no title"),
        ("---\ntitle: T\n---\n\nno sections here\n", "Introduction"),
        ("---\ntitle: T\n---\n", "no sections"),
    ],
)
def test_markdown_errors(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.md"
    path.write_text(text, encoding="utf-8")
    if message == "Introduction":
        assert parse_markdown(path, tmp_path).sections[0].heading == "Introduction"
        return
    with pytest.raises(IngestError, match=message):
        parse_markdown(path, tmp_path)


def test_pdf_sections_pages_and_bullets(tmp_path: Path) -> None:
    path = tmp_path / "fees-pdf.pdf"
    _pdf(
        path,
        [
            [
                "Fee schedule",
                "Document ID fees-pdf. Owner: Practice manager.",
                "Fictional document.",
                "1 Consultations",
                "- Check-up: R650",
                "- New patient exam: R1,150",
                "2 Notes",
                "Fees include VAT and 2 more things are listed here.",
            ],
            ["3 Crowns", "A porcelain crown is R9,800."],
        ],
    )
    doc = parse_pdf(path, tmp_path)
    assert doc.title == "Fee schedule"
    assert doc.meta["control"].startswith("Document ID")
    assert doc.meta["notice"] == "Fictional document."
    assert doc.meta["pages"] == "2"
    assert [s.heading for s in doc.sections] == ["Consultations", "Notes", "Crowns"]
    assert doc.sections[0].paragraphs[0].lines is True
    assert doc.sections[0].page == 1
    assert doc.sections[2].page == 2


def test_pdf_heading_numbers_must_be_in_order(tmp_path: Path) -> None:
    path = tmp_path / "order.pdf"
    _pdf(path, [["Title", "1 First", "Body one.", "3 Not a heading", "2 Second", "Body two."]])
    doc = parse_pdf(path, tmp_path)
    assert [s.heading for s in doc.sections] == ["First", "Second"]
    assert "3 Not a heading" in doc.sections[0].text


def test_pdf_without_headings_fails(tmp_path: Path) -> None:
    path = tmp_path / "flat.pdf"
    _pdf(path, [["Title", "Just text, no numbered headings."]])
    with pytest.raises(IngestError, match="no numbered section headings"):
        parse_pdf(path, tmp_path)


def test_empty_pdf_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    _pdf(path, [[]])
    with pytest.raises(IngestError, match="no text"):
        parse_pdf(path, tmp_path)


def test_load_corpus(mini_corpus: Path) -> None:
    (mini_corpus / "notes.txt").write_text("ignored", encoding="utf-8")
    docs = load_corpus(mini_corpus)
    assert [d.doc_id for d in docs] == ["fees", "staff"]


def test_load_corpus_errors(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        load_corpus(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match=r"no \.md or \.pdf"):
        load_corpus(tmp_path / "empty")
    dup = tmp_path / "dup"
    dup.mkdir()
    for name in ("a.md", "b.md"):
        (dup / name).write_text("---\nid: same\ntitle: T\n---\n## S\n\nx\n", encoding="utf-8")
    with pytest.raises(IngestError, match="duplicate"):
        load_corpus(dup)


def test_relative_path_outside_root(mini_corpus: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    doc = parse_markdown(mini_corpus / "fees.md", other)
    assert doc.path == "fees.md"
