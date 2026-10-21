"""Ingest-time scanning: flag chunks that hold planted instructions, hidden text or personal data.

Flagged chunks stay in the index, so the naive pipeline can be measured against
them, but the grounded pipeline never retrieves them. That is data minimisation
(an assistant does not need staff home details to answer policy questions) and
it means a planted instruction never reaches the model.
"""

from __future__ import annotations

import re
from dataclasses import replace

from grounded_answers.documents import (
    FLAG_HIDDEN_TEXT,
    FLAG_INJECTION,
    FLAG_PERSONAL,
    Chunk,
)

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(instructions?|rules|polic(y|ies)|prompts?)\b",
        re.IGNORECASE,
    ),
    "addresses_the_ai": re.compile(
        r"\b(ai|llm|language model|chat ?bot|virtual) (assistants?|models?|agents?)\b",
        re.IGNORECASE,
    ),
    "system_prompt_claim": re.compile(
        r"\b(system|developer|admin(istrator)?) (instruction|prompt|message|override)s?\b",
        re.IGNORECASE,
    ),
    "conceal_instruction": re.compile(
        r"\bdo not (mention|reveal|disclose|tell anyone about) (this|these) "
        r"(instruction|message)s?\b",
        re.IGNORECASE,
    ),
    "role_change": re.compile(r"\byou are now\b|\bnew instructions?\s*:", re.IGNORECASE),
}

_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_PHONE = re.compile(r"\+27[\d ]{9,13}\d")
_PERSONAL_MARKERS = re.compile(
    r"\b(personal (cell|phone|number|details|address)|home (suburb|address)|emergency contact|"
    r"next of kin|id number|identity number|salary of|medical history of)\b",
    re.IGNORECASE,
)


def injection_rules(text: str) -> list[str]:
    """Names of the injection patterns that match, in a stable order."""
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]


def has_hidden_text(text: str) -> bool:
    """An HTML comment with words in it, or zero-width characters: text a reader never sees."""
    if _ZERO_WIDTH.search(text):
        return True
    return any(len(body.split()) >= 3 for body in _HTML_COMMENT.findall(text))


def has_personal_data(chunk: Chunk) -> bool:
    """Phone numbers next to markers of personal (not work) contact details."""
    haystack = f"{chunk.section}\n{chunk.text}"
    return bool(_PERSONAL_MARKERS.search(haystack)) and bool(_PHONE.search(chunk.text))


def flag_chunk(chunk: Chunk) -> Chunk:
    flags = []
    if injection_rules(chunk.text):
        flags.append(FLAG_INJECTION)
    if has_hidden_text(chunk.text):
        flags.append(FLAG_HIDDEN_TEXT)
    if has_personal_data(chunk):
        flags.append(FLAG_PERSONAL)
    return replace(chunk, flags=tuple(flags))


def flag_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [flag_chunk(c) for c in chunks]
