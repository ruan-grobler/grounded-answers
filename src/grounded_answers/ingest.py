"""Load the corpus: Markdown with front matter, and PDFs, split into titled sections.

Section titles are kept as metadata on every chunk, because a policy answer is
only useful if it can say where it came from ("Billing and payments >
Deposits for larger treatment, page 2").
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pypdf import PdfReader

from grounded_answers.documents import Document, Paragraph, Section
from grounded_answers.jsonutil import sha256_file

SUPPORTED = (".md", ".pdf")
_PDF_HEADING = re.compile(r"^(\d{1,2})\s+([A-Z][^.]{1,70})$")
_LIST_LINE = re.compile(r"^\s*(\||[-*] )")


class IngestError(ValueError):
    """A document could not be parsed into sections."""


def load_corpus(corpus_dir: Path, root: Path | None = None) -> list[Document]:
    """Every Markdown and PDF file in `corpus_dir`, sorted by file name."""
    if not corpus_dir.is_dir():
        raise IngestError(f"corpus folder not found: {corpus_dir}")
    base = root or corpus_dir.parent
    documents = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() == ".md":
            documents.append(parse_markdown(path, base))
        elif path.suffix.lower() == ".pdf":
            documents.append(parse_pdf(path, base))
    if not documents:
        raise IngestError(f"no .md or .pdf files in {corpus_dir}")
    ids = [d.doc_id for d in documents]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise IngestError(f"duplicate document ids: {', '.join(duplicates)}")
    return documents


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


# ------------------------------------------------------------------ markdown


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise IngestError("front matter is not closed with ---")
    raw = yaml.safe_load(text[4:end]) or {}
    if not isinstance(raw, dict):
        raise IngestError("front matter must be a mapping")
    return {str(k): str(v) for k, v in raw.items()}, text[end + 5 :]


def _paragraphs(lines: list[str]) -> list[Paragraph]:
    """Split on blank lines, but never inside an HTML comment."""
    blocks: list[list[str]] = []
    current: list[str] = []
    in_comment = False
    for line in lines:
        if not line.strip() and not in_comment:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line.rstrip())
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            in_comment = True
        elif in_comment and "-->" in line:
            in_comment = False
    if current:
        blocks.append(current)
    out = []
    for block in blocks:
        is_list = all(_LIST_LINE.match(line) for line in block)
        text = "\n".join(block) if is_list else " ".join(line.strip() for line in block)
        out.append(Paragraph(text=text.strip(), lines=is_list))
    return out


def parse_markdown(path: Path, base: Path) -> Document:
    meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
    doc_id = meta.get("id", path.stem)
    title = meta.get("title", "")
    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    notice: list[str] = []

    def flush() -> None:
        if heading is not None:
            paragraphs = tuple(p for p in _paragraphs(buffer) if p.text)
            if paragraphs:
                sections.append(Section(heading=heading, paragraphs=paragraphs))
        elif any(line.strip() for line in buffer):
            sections.append(Section(heading="Introduction", paragraphs=tuple(_paragraphs(buffer))))

    for line in body.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("# "):
            continue
        if heading is None and line.startswith(">"):
            notice.append(line.lstrip("> ").strip())
            continue
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buffer = []
            continue
        buffer.append(line)
    flush()
    if not title:
        raise IngestError(f"{path.name}: no title in front matter or # heading")
    if not sections:
        raise IngestError(f"{path.name}: no sections")
    if notice:
        meta["notice"] = " ".join(notice)
    return Document(
        doc_id=doc_id,
        title=title,
        kind="markdown",
        path=_relative(path, base),
        sha256=sha256_file(path),
        sections=tuple(sections),
        meta=meta,
    )


# ----------------------------------------------------------------------- pdf


def parse_pdf(path: Path, base: Path) -> Document:
    """Numbered headings ("3 Sick leave") start sections. Page numbers are kept for citations."""
    reader = PdfReader(path)
    title = ""
    meta: dict[str, str] = {}
    notice: list[str] = []
    sections: list[Section] = []
    heading: str | None = None
    expected = 1
    paragraphs: list[Paragraph] = []
    prose: list[str] = []
    prose_page: int | None = None
    bullets: list[str] = []
    bullets_page: int | None = None

    def close_prose() -> None:
        nonlocal prose, prose_page
        if prose:
            paragraphs.append(Paragraph(text=" ".join(prose), page=prose_page))
        prose, prose_page = [], None

    def close_bullets() -> None:
        nonlocal bullets, bullets_page
        if bullets:
            paragraphs.append(Paragraph(text="\n".join(bullets), page=bullets_page, lines=True))
        bullets, bullets_page = [], None

    def close_section() -> None:
        nonlocal paragraphs
        close_prose()
        close_bullets()
        if heading is not None and paragraphs:
            sections.append(Section(heading=heading, paragraphs=tuple(paragraphs)))
        paragraphs = []

    for page_number, page in enumerate(reader.pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            if not title:
                title = line
                continue
            match = _PDF_HEADING.match(line)
            if match and int(match.group(1)) == expected:
                close_section()
                heading = match.group(2).strip()
                expected += 1
                continue
            if heading is None:
                if line.startswith("Document ID"):
                    meta["control"] = line
                else:
                    notice.append(line)
                continue
            if line.startswith("- "):
                close_prose()
                if not bullets:
                    bullets_page = page_number
                bullets.append(line)
            else:
                close_bullets()
                if not prose:
                    prose_page = page_number
                prose.append(line)
    close_section()
    if not title:
        raise IngestError(f"{path.name}: no text found (is it a scanned PDF?)")
    if not sections:
        raise IngestError(f"{path.name}: no numbered section headings found")
    if notice:
        meta["notice"] = " ".join(notice)
    meta["pages"] = str(len(reader.pages))
    return Document(
        doc_id=path.stem,
        title=title,
        kind="pdf",
        path=_relative(path, base),
        sha256=sha256_file(path),
        sections=tuple(sections),
        meta=meta,
    )
