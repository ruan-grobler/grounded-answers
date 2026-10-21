"""Split sections into overlapping chunks.

Rules:
- A chunk never crosses a section boundary, so every chunk has one section title.
- Prose is split on sentence boundaries, tables and lists on line boundaries, so a
  fee or a table row is never cut in half.
- Consecutive chunks of the same section share up to `overlap_words` words, so a
  fact that sits on a boundary is still whole in at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from grounded_answers.documents import Chunk, Document, Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 160
    overlap_words: int = 30

    def __post_init__(self) -> None:
        if self.max_words < 20:
            raise ValueError("max_words must be at least 20")
        if not 0 <= self.overlap_words < self.max_words // 2:
            raise ValueError("overlap_words must be between 0 and half of max_words")


@dataclass(frozen=True)
class _Unit:
    text: str
    sep: str  # how this unit joins the previous one: "\n" or " "
    page: int | None

    @property
    def words(self) -> int:
        return len(self.text.split())


def _units(section: Section, max_words: int) -> list[_Unit]:
    units: list[_Unit] = []
    for paragraph in section.paragraphs:
        pieces = (
            paragraph.text.splitlines() if paragraph.lines else _SENTENCE_END.split(paragraph.text)
        )
        for index, piece in enumerate(p.strip() for p in pieces):
            if not piece:
                continue
            sep = "\n" if index == 0 or paragraph.lines else " "
            words = piece.split()
            if len(words) <= max_words:
                units.append(_Unit(piece, sep, paragraph.page))
                continue
            for start in range(0, len(words), max_words):
                part = " ".join(words[start : start + max_words])
                units.append(_Unit(part, sep if start == 0 else " ", paragraph.page))
    return units


def _join(units: list[_Unit]) -> str:
    text = units[0].text
    for unit in units[1:]:
        text += unit.sep + unit.text
    return text


def _tail(units: list[_Unit], budget: int) -> list[_Unit]:
    tail: list[_Unit] = []
    words = 0
    for unit in reversed(units):
        if words + unit.words > budget:
            break
        tail.insert(0, unit)
        words += unit.words
    return tail


def _group(section: Section, config: ChunkingConfig) -> list[list[_Unit]]:
    units = _units(section, config.max_words)
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    fresh = 0  # words in `current` that are not overlap from the previous chunk
    for unit in units:
        size = sum(u.words for u in current)
        if current and fresh and size + unit.words > config.max_words:
            groups.append(current)
            current = _tail(current, config.overlap_words)
            fresh = 0
            if sum(u.words for u in current) + unit.words > config.max_words:
                current = []
        current.append(unit)
        fresh += unit.words
    if current and fresh:
        groups.append(current)
    return groups


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in document.sections:
        for group in _group(section, config):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{len(chunks):02d}",
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    section=section.heading,
                    ordinal=len(chunks),
                    text=_join(group),
                    page=group[0].page,
                )
            )
    return chunks
