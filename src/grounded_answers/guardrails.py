"""Guardrails around the model: a personal data gate on the way in, an output check on the way out.

Both are plain code with no model calls, so they are fast, free, deterministic
and unit tested. They are a second line of defence: the grounded prompt asks
the model for the same behaviour, and these checks hold even when it slips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape

from grounded_answers.documents import Chunk
from grounded_answers.evalset import normalise
from grounded_answers.generator import Answer
from grounded_answers.prompts import CANARY

# ------------------------------------------------------------- query guard

_SENSITIVE = re.compile(
    r"\b(personal|private|home)\s+(cell|phone|number|address|email|details|suburb)\b"
    r"|\b(cell|cellphone|mobile)(\s+phone)?(\s+number)?\b"
    r"|\bhome\s+address\b|\bsuburb\b|\blives?\b|\bliving\b"
    r"|\bemergency\s+contacts?\b|\bnext\s+of\s+kin\b"
    r"|\b(id|identity|passport)\s+number\b|\bdate\s+of\s+birth\b|\bbirthday\b"
    r"|\bsalary\b|\bsalaries\b|\bearns?\b|\bearning\b|\bwages?\b|\bpaid\b"
    r"|\bmedical\s+(history|condition|aid\s+number)\b|\bdiagnos\w*|\bsick\b|\billness\b",
    re.IGNORECASE,
)
_PATIENT_LISTING = re.compile(
    r"\b(list|names?|details)\s+(of\s+)?(all\s+)?(the\s+)?(our\s+)?patients\b"
    r"|\bpatients\s+(who\s+are\s+)?(booked|scheduled|coming\s+in|seen)\b"
    r"|\bwho\s+(is|are)\s+booked\b|\bwhich\s+patients\b",
    re.IGNORECASE,
)
_RELATION = re.compile(
    r"\b(her|his|their)\s+(husband|wife|spouse|partner|mother|father|sister|brother|son|"
    r"daughter|family|kids?|children)\b",
    re.IGNORECASE,
)
_POSSESSIVE_ROLE = re.compile(
    r"\bthe\s+(receptionist|hygienist|dental\s+assistant|assistant|dentist|practice\s+manager|"
    r"manager|principal|billing\s+clerk)'s\b",
    re.IGNORECASE,
)
_TITLED = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Za-z][\w'-]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
# Capitalised words that start questions or name things that are not people.
_NOT_NAMES = frozenset(
    """What Who Whom Whose Which Where When Why How Can Could Is Are Do Does Did Give Tell Show
    Please If A An The Our We I My Seapoint Smile Dental Example Road Information Officer
    Regulator Bowie Dick POPIA Day Christmas Goodwill Reconciliation New Year Monday Tuesday
    Wednesday Thursday Friday Saturday Sunday January February March April May June July
    August September October November December Class Celsius Practice Reception Deputy
    Mondays Tuesdays Wednesdays Thursdays Fridays Saturdays Sundays Bowie-Dick Wi-Fi X-ray
    X-rays Cape Town Sea Point English Afrikaans Medical Aid Information Officer's""".split()
)


@dataclass(frozen=True)
class QueryVerdict:
    allowed: bool
    rule: str | None = None


def names_a_person(question: str) -> bool:
    """A title and name, or a capitalised word that is not a question word or a known non-name.

    The first word of each sentence is skipped (it is capitalised anyway), and so are
    acronyms like POPIA or AED.
    """
    if _TITLED.search(question):
        return True
    for sentence in re.split(r"(?<=[.?!])\s+", question.strip()):
        tokens = _WORD.findall(sentence)
        for token in tokens[1:]:
            word = token.removesuffix("'s")
            if word[:1].isupper() and not word.isupper() and word not in _NOT_NAMES:
                return True
    return False


def classify_query(question: str) -> QueryVerdict:
    """Block requests for an identifiable person's personal data, and for patient lists.

    Policy questions that mention personal data ("how long do we keep patient
    records?") are allowed: the rule needs both a sensitive attribute and a
    specific person.
    """
    if _PATIENT_LISTING.search(question):
        return QueryVerdict(False, "patient_listing")
    targets_person = (
        names_a_person(question)
        or bool(_RELATION.search(question))
        or bool(_POSSESSIVE_ROLE.search(question))
    )
    if targets_person and _SENSITIVE.search(question):
        return QueryVerdict(False, "personal_attribute")
    return QueryVerdict(True)


# ------------------------------------------------------------ output guard

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_PHONE = re.compile(r"\+?\d[\d ]{7,}\d")


@dataclass
class OutputCheck:
    blocked: bool
    reasons: list[str] = field(default_factory=list)
    claims: int = 0
    verified: int = 0
    unknown_sources: list[str] = field(default_factory=list)

    @property
    def all_verified(self) -> bool:
        return self.claims > 0 and self.verified == self.claims


def _clean(text: str) -> str:
    return normalise(unescape(text))


def check_output(answer: Answer, context: dict[str, Chunk]) -> OutputCheck:
    """Verify every quote against the chunk it cites, and scan for leaks.

    Blocks when: a claim cites a source that was not provided, a quote is not
    found word for word in its cited chunks, an answer has no claims, a contact
    detail (email, link, phone) is not in a cited chunk, or the canary leaks.
    """
    check = OutputCheck(blocked=False, claims=len(answer.claims))
    texts = [answer.answer, *(c.statement for c in answer.claims)]
    if any(CANARY in t for t in texts):
        check.reasons.append("canary_leak")
    if answer.refused:
        check.blocked = bool(check.reasons)
        return check
    if not answer.claims:
        check.reasons.append("no_claims")
    cited: list[Chunk] = []
    for claim in answer.claims:
        sources = [context[s] for s in claim.source_ids if s in context]
        check.unknown_sources.extend(s for s in claim.source_ids if s not in context)
        cited.extend(sources)
        quote = _clean(claim.quote)
        if quote and any(quote in _clean(src.text) for src in sources):
            check.verified += 1
    if check.unknown_sources:
        check.reasons.append("unknown_source")
    if check.verified < check.claims:
        check.reasons.append("unverified_quote")
    cited_text = " ".join(_clean(c.text) for c in cited)
    for pattern in (_EMAIL, _URL, _PHONE):
        for text in texts:
            for found in pattern.findall(text):
                if _clean(found) not in cited_text:
                    check.reasons.append("contact_not_in_sources")
    check.reasons = sorted(set(check.reasons))
    check.blocked = bool(check.reasons)
    return check
