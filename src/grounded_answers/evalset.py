"""The eval set: questions, gold evidence locations and exact checks (eval/*.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from grounded_answers.documents import Chunk

Category = Literal["answerable", "multi_hop", "unanswerable", "pii", "injection"]
CATEGORIES: tuple[Category, ...] = ("answerable", "multi_hop", "unanswerable", "pii", "injection")
Expect = Literal["answer", "refuse"]
REFUSAL_REASONS = ("not_in_documents", "personal_data")


class EvalSetError(ValueError):
    """eval/*.yaml does not match the documented format."""


@dataclass(frozen=True)
class Location:
    doc: str
    section: str
    fact: str

    def matches(self, chunk: Chunk) -> bool:
        """Section level match: robust to changes in chunk size."""
        return chunk.doc_id == self.doc and chunk.section == self.section

    def holds_fact(self, chunk: Chunk) -> bool:
        return self.matches(chunk) and normalise(self.fact) in normalise(chunk.text)


@dataclass(frozen=True)
class Hop:
    """One piece of evidence the answer needs. Any one of `locations` satisfies it."""

    locations: tuple[Location, ...]

    def matches(self, chunk: Chunk) -> bool:
        return any(loc.matches(chunk) for loc in self.locations)


Check = tuple[str, ...]  # any one of these strings


@dataclass(frozen=True)
class Question:
    id: str
    category: Category
    question: str
    expect: Expect
    reference: str
    evidence: tuple[Hop, ...] = ()
    must_include: tuple[Check, ...] = ()
    must_not_include: tuple[str, ...] = ()
    refusal_reason: str | None = None

    def is_gold(self, chunk: Chunk) -> bool:
        return any(hop.matches(chunk) for hop in self.evidence)


_SPACES = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_QUOTES = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})


def normalise(text: str) -> str:
    """Lower case, straight quotes, single spaces, and no thousands separators (R4,900 = R4900)."""
    text = text.translate(_QUOTES).lower()
    text = _THOUSANDS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def _location(raw: Any, qid: str) -> Location:
    if not isinstance(raw, dict) or not {"doc", "section", "fact"} <= raw.keys():
        raise EvalSetError(f"{qid}: each evidence location needs doc, section and fact")
    return Location(doc=str(raw["doc"]), section=str(raw["section"]), fact=str(raw["fact"]))


def _hop(raw: Any, qid: str) -> Hop:
    if isinstance(raw, dict) and "any_of" in raw:
        locations = tuple(_location(r, qid) for r in raw["any_of"])
        if not locations:
            raise EvalSetError(f"{qid}: any_of needs at least one location")
        return Hop(locations)
    return Hop((_location(raw, qid),))


def _check(raw: Any) -> Check:
    if isinstance(raw, list):
        return tuple(str(r) for r in raw)
    return (str(raw),)


def _question(raw: dict[str, Any]) -> Question:
    qid = str(raw.get("id", "?"))
    for key in ("id", "category", "question", "expect", "reference"):
        if key not in raw:
            raise EvalSetError(f"{qid}: missing {key}")
    if raw["category"] not in CATEGORIES:
        raise EvalSetError(f"{qid}: unknown category {raw['category']!r}")
    if raw["expect"] not in ("answer", "refuse"):
        raise EvalSetError(f"{qid}: expect must be answer or refuse")
    evidence = tuple(_hop(h, qid) for h in raw.get("evidence") or [])
    if raw["expect"] == "answer" and not evidence:
        raise EvalSetError(f"{qid}: an answerable question needs evidence")
    if raw["expect"] == "refuse":
        if evidence:
            raise EvalSetError(f"{qid}: a refuse question should not list evidence")
        if raw.get("refusal_reason") not in REFUSAL_REASONS:
            raise EvalSetError(f"{qid}: refusal_reason must be one of {REFUSAL_REASONS}")
    return Question(
        id=qid,
        category=raw["category"],
        question=str(raw["question"]).strip(),
        expect=raw["expect"],
        reference=str(raw["reference"]).strip(),
        evidence=evidence,
        must_include=tuple(_check(c) for c in raw.get("must_include") or []),
        must_not_include=tuple(str(s) for s in raw.get("must_not_include") or []),
        refusal_reason=raw.get("refusal_reason"),
    )


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("questions")
    if not isinstance(raw, list) or not raw:
        raise EvalSetError(f"{path} has no questions list")
    questions = [_question(r) for r in raw]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise EvalSetError(f"{path} has duplicate question ids")
    texts = [q.question for q in questions]
    if len(set(texts)) != len(texts):
        raise EvalSetError(f"{path} has duplicate question texts")
    return questions


def check_evidence(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Problems with the gold labels: sections that do not exist, or facts not in them."""
    problems = []
    for question in questions:
        for hop in question.evidence:
            for loc in hop.locations:
                in_section = [c for c in chunks if loc.matches(c)]
                if not in_section:
                    problems.append(f"{question.id}: no section {loc.doc} > {loc.section}")
                elif not any(loc.holds_fact(c) for c in in_section):
                    problems.append(f"{question.id}: fact {loc.fact!r} not in {loc.section}")
    return problems
