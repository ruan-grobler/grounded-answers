"""Documents, sections and chunks: the units the pipeline reads, retrieves and cites."""

from __future__ import annotations

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
        """What the embedder and reranker see: the section path, then the text."""
        return f"{self.header}\n{self.text}"

    @property
    def location(self) -> str:
        page = f", page {self.page}" if self.page else ""
        return f"{self.doc_id}, {self.section}{page}"
