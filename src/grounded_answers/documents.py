"""Documents, sections and chunks: the units the pipeline reads, retrieves and cites."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Chunk flags set at ingest time by `scan.py`. The grounded pipeline never
# retrieves a flagged chunk. The naive pipeline ignores flags.
FLAG_INJECTION = "prompt_injection"
FLAG_HIDDEN_TEXT = "hidden_text"
FLAG_PERSONAL = "personal_data"
ALL_FLAGS = (FLAG_INJECTION, FLAG_HIDDEN_TEXT, FLAG_PERSONAL)


@dataclass(frozen=True)
class Paragraph:
    """One block of text inside a section. `lines` is True for tables and lists."""

    text: str
    page: int | None = None
    lines: bool = False


@dataclass(frozen=True)
class Section:
    heading: str
    paragraphs: tuple[Paragraph, ...]

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def page(self) -> int | None:
        return self.paragraphs[0].page if self.paragraphs else None


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    kind: str  # "markdown" or "pdf"
    path: str  # relative to the repo root
    sha256: str
    sections: tuple[Section, ...]
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    ordinal: int
    text: str
    page: int | None = None
    flags: tuple[str, ...] = ()

    @property
    def header(self) -> str:
        return f"{self.doc_title} > {self.section}"

    @property
    def embed_text(self) -> str:
        """What the embedder and reranker see: section path, then text, tables as sentences."""
        return f"{self.header}\n{linearize_tables(self.text)}"

    @property
    def location(self) -> str:
        page = f", page {self.page}" if self.page else ""
        return f"{self.doc_id}, {self.section}{page}"


_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def linearize_tables(text: str) -> str:
    """Rewrite markdown table rows as "Header: value; Header: value" lines.

    Embedding models and cross-encoders are trained on prose and read pipe
    tables badly. The original text is still what the model is shown and what
    quotes are checked against; only retrieval sees this version.
    """
    lines = text.splitlines()
    out: list[str] = []
    header: list[str] | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            out.append(line)
            continue
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if header is None and _TABLE_RULE.match(following):
            header = _cells(stripped)
            continue
        if _TABLE_RULE.match(stripped):
            continue
        cells = _cells(stripped)
        if header and len(header) == len(cells):
            out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells, strict=True)) + ".")
        else:
            out.append(", ".join(cells) + ".")
    return "\n".join(out)
