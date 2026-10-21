"""Compare retrieval settings on the DEV split only (docs/method.md, "Choosing on dev").

Runs dense, hybrid and five cross-encoder rerankers, each with and without query
decomposition, over the dev questions that have evidence, and prints coverage,
hit rate, MRR and live CPU latency. It downloads the rerankers it needs (about
1.6 GB in total, free, no account) into .cache/models.

Run: .venv/bin/python scripts/select_on_dev.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from grounded_answers.config import Paths
from grounded_answers.evalset import Question, load_questions
from grounded_answers.inference import (
    CachedEmbedder,
    CachedReranker,
    FastEmbedEmbedder,
    FastEmbedReranker,
    Reranker,
    ScoreCache,
    VectorCache,
)
from grounded_answers.retrieval import Mode, RetrievalConfig, Retriever
from grounded_answers.store import Store

RERANKERS = (
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
)
KEEP = -6.5  # the keep threshold chosen by `ga tune`


def score(retriever: Retriever, questions: list[Question]) -> str:
    hits = dict.fromkeys((1, 3, 5, 10), 0)
    reciprocal: list[float] = []
    covered = 0
    for q in questions:
        result = retriever.retrieve(q.question)
        ranks = [
            next((i for i, c in enumerate(result.ranked, 1) if hop.matches(c.chunk)), None)
            for hop in q.evidence
        ]
        for k in hits:
            hits[k] += all(r is not None and r <= k for r in ranks)
        found = [r for r in ranks if r is not None]
        reciprocal.append(1 / min(found) if found else 0.0)
        covered += all(any(hop.matches(c.chunk) for c in result.context) for hop in q.evidence)
    n = len(questions)
    at = " ".join(f"@{k} {hits[k] / n:>4.0%}" for k in hits)
    mrr = statistics.fmean(reciprocal)
    return f"prompt has all evidence {covered / n:>4.0%}  {at}  MRR {mrr:.2f}"


def latency(retriever: Retriever, questions: list[Question]) -> str:
    retriever.retrieve(questions[0].question)
    times = []
    for q in questions:
        start = time.perf_counter()
        retriever.retrieve(q.question)
        times.append((time.perf_counter() - start) * 1000)
    cuts = statistics.quantiles(times, n=100, method="inclusive")
    return f"p50 {cuts[49]:>5.0f} ms  p95 {cuts[94]:>6.0f} ms"


def main() -> None:
    paths = Paths(Path(__file__).resolve().parents[1])
    work = paths.root / ".cache" / "experiments"
    work.mkdir(parents=True, exist_ok=True)
    store = Store(paths.index_db)
    live_embedder = FastEmbedEmbedder(paths.models_dir)
    embedder = CachedEmbedder(VectorCache(work / "vectors.json"), live_embedder, record=True)
    dev = [q for q in load_questions(paths.dev_questions) if q.evidence]
    print(f"Dev questions with evidence: {len(dev)}\n")
    modes: tuple[Mode, ...] = ("dense", "hybrid")
    for mode in modes:
        for decompose in (False, True):
            config = RetrievalConfig(mode=mode, decompose=decompose)
            line = score(Retriever(store, embedder, None, config), dev)
            print(f"{mode:<34} decompose {decompose!s:<5}  {line}")
    for name in RERANKERS:
        live: Reranker = FastEmbedReranker(paths.models_dir, allow_download=True, model_name=name)
        cache = ScoreCache(work / f"rerank-{name.replace('/', '_')}.json", name)
        cached = CachedReranker(cache, live, record=True)
        for decompose in (False, True):
            config = RetrievalConfig(mode="hybrid_rerank", keep_min_score=KEEP, decompose=decompose)
            quality = score(Retriever(store, embedder, cached, config), dev)
            speed = latency(Retriever(store, live_embedder, live, config), dev)
            print(f"{name:<34} decompose {decompose!s:<5}  {quality}  {speed}")
        cache.save()
    embedder.cache.save()


if __name__ == "__main__":
    main()
