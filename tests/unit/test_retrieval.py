from __future__ import annotations

from pathlib import Path

import pytest

from grounded_answers.retrieval import RetrievalConfig, Retriever, sub_queries
from tests.helpers import FakeEmbedder, FakeReranker, build_store, chunk


@pytest.fixture
def store(tmp_path: Path):
    chunks = [
        chunk("fees#00", "Late cancellation fee is R350 with less than 24 hours notice."),
        chunk("fees#01", "No show fee is R500 when a patient does not arrive."),
        chunk("staff#00", "Megan Adams practice manager extension 107."),
        chunk("staff#01", "Megan Adams personal cell number.", flags=("personal_data",)),
        chunk("waste#00", "Healthcare risk waste is collected every Tuesday."),
        chunk("lab#00", "Crowns take 7 working days at the lab."),
    ]
    s = build_store(tmp_path, chunks)
    yield s
    s.close()


def _retriever(store, **kwargs) -> Retriever:
    config = RetrievalConfig(candidates=5, context_k=3, rerank_top=5, **kwargs)
    reranker = FakeReranker() if config.mode == "hybrid_rerank" else None
    return Retriever(store, FakeEmbedder(), reranker, config)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_obvious_chunk(store, mode: str) -> None:
    result = _retriever(store, mode=mode).retrieve("What is the late cancellation fee?")
    assert result.ranked[0].chunk.chunk_id == "fees#00"
    assert result.context[0].chunk.chunk_id == "fees#00"
    assert len(result.context) <= 3
    assert result.timings_ms["retrieval_total"] >= 0
    assert result.queries == ["What is the late cancellation fee?"]


def test_hybrid_records_both_ranks(store) -> None:
    result = _retriever(store, mode="hybrid").retrieve("late cancellation fee")
    top = result.ranked[0]
    assert top.dense_rank == 1
    assert top.bm25_rank == 1


def test_rerank_scores_and_gate(store) -> None:
    result = _retriever(store, mode="hybrid_rerank", gate_min_score=0.0).retrieve(
        "late cancellation fee"
    )
    assert result.top_score == pytest.approx(3.0)  # 3 shared words: 2 * 3 - 3
    assert not result.gated
    gated = _retriever(store, mode="hybrid_rerank", gate_min_score=0.0).retrieve(
        "parking validation"
    )
    assert gated.gated
    assert gated.context == []


def test_keep_threshold_limits_context(store) -> None:
    result = _retriever(store, mode="hybrid_rerank", keep_min_score=0.0).retrieve(
        "late cancellation fee"
    )
    assert [c.chunk.chunk_id for c in result.context] == ["fees#00"]


def test_blocked_flags_are_never_retrieved(store) -> None:
    result = _retriever(store, mode="hybrid", blocked_flags=frozenset({"personal_data"})).retrieve(
        "Megan Adams personal cell number"
    )
    assert "staff#01" not in [c.chunk.chunk_id for c in result.ranked]
    open_result = _retriever(store, mode="hybrid").retrieve("Megan Adams personal cell number")
    assert open_result.ranked[0].chunk.chunk_id == "staff#01"


def test_sub_queries() -> None:
    assert sub_queries("What is the late fee?") == ["What is the late fee?"]
    assert sub_queries("Who is the lead, and what is their extension?") == [
        "Who is the lead, and what is their extension?",
        "Who is the lead",
        "what is their extension?",
    ]
    assert sub_queries("Do we take debit and credit cards?") == [
        "Do we take debit and credit cards?"
    ]
    assert sub_queries("A case went out on Monday. When is it due?") == [
        "A case went out on Monday. When is it due?",
        "A case went out on Monday.",
        "When is it due?",
    ]


def test_decomposition_covers_every_part(store) -> None:
    question = "What is the no show fee, and how long do crowns take at the lab?"
    plain = _retriever(store, mode="hybrid_rerank", keep_min_score=0.0).retrieve(question)
    split = _retriever(store, mode="hybrid_rerank", keep_min_score=0.0, decompose=True).retrieve(
        question
    )
    ids = [c.chunk.chunk_id for c in split.context]
    assert "fees#01" in ids
    assert "lab#00" in ids
    assert len(split.queries) == 3
    assert {c.best_query for c in split.ranked[:2]} <= {0, 1, 2}
    assert len(plain.queries) == 1


def test_decomposition_without_rerank_merges_scores(store) -> None:
    result = _retriever(store, mode="hybrid", decompose=True).retrieve(
        "What is the no show fee, and how long do crowns take at the lab?"
    )
    ids = [c.chunk.chunk_id for c in result.ranked[:3]]
    assert "fees#01" in ids
    assert "lab#00" in ids


def test_bad_configs(store) -> None:
    with pytest.raises(ValueError, match="unknown"):
        RetrievalConfig(mode="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="context_k"):
        RetrievalConfig(context_k=0)
    with pytest.raises(ValueError, match="needs a reranker"):
        Retriever(store, FakeEmbedder(), None, RetrievalConfig(mode="hybrid_rerank"))
