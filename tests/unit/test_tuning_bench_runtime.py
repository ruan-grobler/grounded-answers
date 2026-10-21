from __future__ import annotations

from pathlib import Path

import pytest

from grounded_answers.bench import bench
from grounded_answers.config import Paths
from grounded_answers.evalset import Hop, Location, Question
from grounded_answers.inference import CachedEmbedder, VectorCache
from grounded_answers.retrieval import RetrievalConfig, Retriever
from grounded_answers.runtime import client_factory
from grounded_answers.tuning import tune, tune_config
from tests.helpers import FakeEmbedder, FakeReranker, build_store, chunk


@pytest.fixture
def store(tmp_path: Path):
    s = build_store(
        tmp_path,
        [
            chunk("fees#00", "late cancellation fee R350", section="Fees"),
            chunk("waste#00", "risk waste collected Tuesday", section="Collection"),
        ],
    )
    yield s
    s.close()


QUESTIONS = [
    Question(
        "d1",
        "answerable",
        "late cancellation fee",
        "answer",
        "R350",
        evidence=(Hop((Location("fees", "Fees", "R350"),)),),
    ),
    Question(
        "d2",
        "answerable",
        "waste collected",
        "answer",
        "Tuesday",
        evidence=(Hop((Location("waste", "Collection", "Tuesday"),)),),
    ),
    Question(
        "d3", "unanswerable", "parking validation", "refuse", "-", refusal_reason="not_in_documents"
    ),
    Question(
        "d4",
        "pii",
        "What is Megan Adams's cell number?",
        "refuse",
        "-",
        refusal_reason="personal_data",
    ),
]


def test_tune_follows_the_written_rule(store) -> None:
    retriever = Retriever(store, FakeEmbedder(), FakeReranker(), tune_config(RetrievalConfig()))
    report = tune(retriever, QUESTIONS)
    # Weakest answerable top score is 1.0 (2 shared words), weakest needed gold also 1.0.
    assert report.gate_min_score == 0.5
    assert report.keep_min_score == 0.5
    assert report.refusals_considered == 1  # d4 is blocked by the guard first
    assert report.gated_refusals == 1
    assert [r.blocked_by_guard for r in report.rows] == [False, False, False, True]


def test_tune_needs_a_clean_rerank_config(store) -> None:
    retriever = Retriever(
        store, FakeEmbedder(), FakeReranker(), RetrievalConfig(gate_min_score=0.0)
    )
    with pytest.raises(ValueError, match="no thresholds"):
        tune(retriever, QUESTIONS)
    clean = Retriever(store, FakeEmbedder(), FakeReranker(), tune_config(RetrievalConfig()))
    with pytest.raises(ValueError, match="answerable"):
        tune(clean, QUESTIONS[2:])


def test_bench_matches_replay(store, tmp_path: Path) -> None:
    live_embedder = FakeEmbedder()
    cache = VectorCache(tmp_path / "v.json")
    cached = CachedEmbedder(cache, live_embedder, record=True)
    reranker = FakeReranker()
    naive_cfg = RetrievalConfig(mode="dense", candidates=2, context_k=1)
    grounded_cfg = RetrievalConfig(mode="hybrid_rerank", candidates=2, context_k=1, rerank_top=2)
    cached_naive = Retriever(store, cached, None, naive_cfg)
    cached_grounded = Retriever(store, cached, reranker, grounded_cfg)
    report = bench(
        store,
        live_embedder,
        reranker,
        cache,
        cached_naive,
        cached_grounded,
        naive_cfg,
        grounded_cfg,
        QUESTIONS[:2],
    )
    assert report.questions == 2
    assert report.rankings_match == 2
    assert report.max_query_diff == 0.0
    assert report.max_score_diff == 0.0
    assert report.naive_ms["p50"] >= 0


def test_replay_client_factory(tmp_path: Path) -> None:
    paths = Paths(tmp_path)
    make = client_factory(paths, "grounded", "replay")
    client, finish = make(QUESTIONS[0])
    finish(True)
    with pytest.raises(LookupError, match="cassette miss"):
        client.create({"model": "m", "messages": []}, role="generator")


def test_record_mode_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        client_factory(Paths(tmp_path), "grounded", "record")
