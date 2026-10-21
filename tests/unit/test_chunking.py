from __future__ import annotations

import pytest

from grounded_answers.chunking import ChunkingConfig, chunk_document
from grounded_answers.documents import Document, Paragraph, Section, linearize_tables


def _doc(*sections: Section) -> Document:
    return Document("doc", "Doc", "markdown", "corpus/doc.md", "sha", tuple(sections))


def test_short_sections_are_one_chunk_each() -> None:
    doc = _doc(
        Section("A", (Paragraph("First sentence. Second sentence."),)),
        Section("B", (Paragraph("Other text."),)),
    )
    chunks = chunk_document(doc)
    assert [c.chunk_id for c in chunks] == ["doc#00", "doc#01"]
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[0].text == "First sentence. Second sentence."
    assert chunks[0].header == "Doc > A"
    assert chunks[0].embed_text.startswith("Doc > A\n")


def test_long_section_splits_on_sentences_with_overlap() -> None:
    sentences = [f"Sentence {i} has exactly six words." for i in range(12)]
    doc = _doc(Section("Long", (Paragraph(" ".join(sentences)),)))
    chunks = chunk_document(doc, ChunkingConfig(max_words=24, overlap_words=6))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.split()) <= 24
        assert chunk.section == "Long"
    # The last sentence of one chunk starts the next one.
    first_tail = chunks[0].text.split(". ")[-1]
    assert chunks[1].text.startswith(first_tail.rstrip("."))
    joined = " ".join(c.text for c in chunks)
    for sentence in sentences:
        assert sentence in joined


def test_very_long_sentence_is_split_by_words() -> None:
    doc = _doc(Section("Run on", (Paragraph(" ".join(["word"] * 50)),)))
    chunks = chunk_document(doc, ChunkingConfig(max_words=20, overlap_words=0))
    assert [len(c.text.split()) for c in chunks] == [20, 20, 10]


def test_split_table_repeats_its_header() -> None:
    rows = "\n".join(f"| Person {i} | Role number {i} | {100 + i} |" for i in range(12))
    table = "| Name | Role | Extension |\n|---|---|---|\n" + rows
    doc = _doc(Section("Directory", (Paragraph(table, lines=True),)))
    chunks = chunk_document(doc, ChunkingConfig(max_words=40, overlap_words=8))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("| Name | Role | Extension |\n|---|---|---|")
    assert "Name: Person 11; Role: Role number 11; Extension: 111." in chunks[-1].embed_text


def test_pages_come_from_the_first_unit() -> None:
    doc = _doc(Section("P", (Paragraph("On page two.", page=2), Paragraph("Also two.", page=2))))
    assert chunk_document(doc)[0].page == 2


@pytest.mark.parametrize(("max_words", "overlap"), [(10, 0), (100, 60), (100, -1)])
def test_bad_config(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_words=max_words, overlap_words=overlap)


def test_linearize_tables() -> None:
    text = "Intro line.\n| A | B |\n|---|---|\n| 1 | 2 |\n| odd row |\nAfter."
    assert linearize_tables(text) == "Intro line.\nA: 1; B: 2.\nodd row.\nAfter."


def test_linearize_leaves_prose_alone() -> None:
    assert linearize_tables("No tables | here really.") == "No tables | here really."
