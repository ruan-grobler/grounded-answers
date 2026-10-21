"""Choose the rerank thresholds from the dev split, by a written rule, never from the test set.

Rule (margin 0.5, rounded down to a 0.5 step):
- gate_min_score: the weakest top rerank score among dev questions that should be
  answered, minus the margin. So no dev question that should be answered is gated.
- keep_min_score: the weakest rerank score of any gold chunk a dev answer needs,
  minus the margin. So every needed dev chunk reaches the prompt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_answers.evalset import Question
from grounded_answers.guardrails import classify_query
from grounded_answers.retrieval import RetrievalConfig, Retriever

MARGIN = 0.5
STEP = 0.5


@dataclass(frozen=True)
class DevRow:
    id: str
    expect: str
    blocked_by_guard: bool
    top_score: float | None
    hop_scores: tuple[float | None, ...]


@dataclass(frozen=True)
class TuneReport:
    rows: list[DevRow]
    gate_min_score: float
    keep_min_score: float
    gated_refusals: int
    refusals_considered: int


def _floor(value: float) -> float:
    return math.floor((value - MARGIN) / STEP) * STEP


def tune(retriever: Retriever, dev: list[Question]) -> TuneReport:
    cfg = retriever.config
    if cfg.mode != "hybrid_rerank" or cfg.gate_min_score is not None or cfg.keep_min_score is not None:
        raise ValueError("tune needs a hybrid_rerank retriever with no thresholds set")
    rows = []
    for question in dev:
        blocked = not classify_query(question.question).allowed
        result = retriever.retrieve(question.question)
        hop_scores = []
        for hop in question.evidence:
            scores = [c.score for c in result.ranked if hop.matches(c.chunk)]
            hop_scores.append(max(scores) if scores else None)
        rows.append(
            DevRow(question.id, question.expect, blocked, result.top_score, tuple(hop_scores))
        )
    answer_rows = [r for r in rows if r.expect == "answer"]
    tops = [r.top_score for r in answer_rows if r.top_score is not None]
    needed = [s for r in answer_rows for s in r.hop_scores if s is not None]
    if not tops or not needed:
        raise ValueError("the dev split needs answerable questions with retrievable evidence")
    gate = _floor(min(tops))
    keep = _floor(min(needed))
    refusals = [r for r in rows if r.expect == "refuse" and not r.blocked_by_guard]
    gated = sum(1 for r in refusals if r.top_score is None or r.top_score < gate)
    return TuneReport(rows, gate, keep, gated, len(refusals))


def tune_config(base: RetrievalConfig) -> RetrievalConfig:
    """The grounded retrieval settings with the thresholds removed, ready for `tune`."""
    return RetrievalConfig(
        mode=base.mode,
        candidates=base.candidates,
        rrf_k=base.rrf_k,
        rerank_top=base.rerank_top,
        context_k=base.context_k,
        blocked_flags=base.blocked_flags,
    )
