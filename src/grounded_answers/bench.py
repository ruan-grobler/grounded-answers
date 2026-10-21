"""Measure retrieval latency with the live local models, and check the committed caches.

Replay mode serves model outputs from index/, so its timings say nothing about
real latency. `ga bench` runs the real models on every eval question and
reports per stage timings, plus the largest difference between a live output
and the committed value, which is the evidence that the caches are genuine.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from grounded_answers.evalset import Question
from grounded_answers.inference import Embedder, Reranker, VectorCache
from grounded_answers.retrieval import RetrievalConfig, Retriever
from grounded_answers.store import Store


@dataclass(frozen=True)
class BenchReport:
    questions: int
    naive_ms: dict[str, float]
    grounded_ms: dict[str, float]
    max_passage_diff: float
    max_query_diff: float
    max_score_diff: float
    rankings_match: int
    threads_note: str


def _p(values: list[float], pct: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]


def _stats(values: list[float]) -> dict[str, float]:
    return {"p50": _p(values, 50), "p95": _p(values, 95), "mean": statistics.fmean(values)}


def bench(
    store: Store,
    live_embedder: Embedder,
    live_reranker: Reranker,
    cache: VectorCache,
    cached_naive: Retriever,
    cached_grounded: Retriever,
    naive_config: RetrievalConfig,
    grounded_config: RetrievalConfig,
    questions: list[Question],
) -> BenchReport:
    # 1. Every committed passage vector against a fresh live embedding.
    chunks = store.chunks
    fresh = live_embedder.embed([c.embed_text for c in chunks], "passage")
    passage_diff = 0.0
    for chunk, vector in zip(chunks, fresh, strict=True):
        cached = cache.get("passage", chunk.embed_text)
        if cached is not None:
            passage_diff = max(passage_diff, float(np.max(np.abs(cached - vector))))

    live_naive = Retriever(store, live_embedder, None, naive_config)
    live_grounded = Retriever(store, live_embedder, live_reranker, grounded_config)
    # Warm up once so model loading is not counted as latency.
    live_grounded.retrieve(questions[0].question)

    naive_ms: list[float] = []
    grounded_ms: list[float] = []
    query_diff = 0.0
    score_diff = 0.0
    matches = 0
    for question in questions:
        start = time.perf_counter()
        naive = live_naive.retrieve(question.question)
        naive_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        grounded = live_grounded.retrieve(question.question)
        grounded_ms.append((time.perf_counter() - start) * 1000)

        live_q = live_embedder.embed([question.question], "query")[0]
        cached_q = cache.get("query", question.question)
        if cached_q is not None:
            query_diff = max(query_diff, float(np.max(np.abs(cached_q - live_q))))
        replay_g = cached_grounded.retrieve(question.question)
        replay_n = cached_naive.retrieve(question.question)
        for a, b in zip(grounded.ranked, replay_g.ranked, strict=False):
            score_diff = max(score_diff, abs(a.score - b.score))
        same = [c.chunk.chunk_id for c in grounded.ranked] == [
            c.chunk.chunk_id for c in replay_g.ranked
        ] and [c.chunk.chunk_id for c in naive.ranked] == [
            c.chunk.chunk_id for c in replay_n.ranked
        ]
        matches += same
    return BenchReport(
        questions=len(questions),
        naive_ms=_stats(naive_ms),
        grounded_ms=_stats(grounded_ms),
        max_passage_diff=passage_diff,
        max_query_diff=query_diff,
        max_score_diff=score_diff,
        rankings_match=matches,
        threads_note="CPU only, ONNX runtime default threads",
    )
