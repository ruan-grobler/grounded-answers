"""Retrieval: dense, BM25, hybrid (reciprocal rank fusion) and hybrid plus cross-encoder rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from grounded_answers.documents import Chunk
from grounded_answers.inference import Embedder, Reranker
from grounded_answers.store import Store

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "hybrid_rerank"
    candidates: int = 20  # taken from each retriever before fusion
    rrf_k: int = 60
    rerank_top: int = 20
    context_k: int = 5
    keep_min_score: float | None = None  # rerank logit a chunk needs to reach the prompt
    gate_min_score: float | None = None  # refuse without calling the model below this
    blocked_flags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown retrieval mode {self.mode!r}")
        if self.context_k < 1 or self.candidates < self.context_k:
            raise ValueError("need 1 <= context_k <= candidates")


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank: float | None = None


@dataclass
class RetrievalResult:
    mode: Mode
    ranked: list[Candidate]
    context: list[Candidate]
    top_score: float | None
    gated: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalConfig,
    ) -> None:
        if config.mode == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        self._blocked = {
            c.chunk_id for c in store.chunks if config.blocked_flags.intersection(c.flags)
        }

    def _dense(self, question: str, timings: dict[str, float]) -> list[str]:
        start = time.perf_counter()
        vector = self.embedder.embed([question], "query")[0]
        timings["embed_query"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        hits = self.store.knn(vector, self.config.candidates + len(self._blocked))
        timings["vector_search"] = (time.perf_counter() - start) * 1000
        return [cid for cid, _ in hits if cid not in self._blocked][: self.config.candidates]

    def _bm25(self, question: str, timings: dict[str, float]) -> list[str]:
        start = time.perf_counter()
        hits = self.store.bm25(question, self.config.candidates + len(self._blocked))
        timings["bm25_search"] = (time.perf_counter() - start) * 1000
        return [cid for cid, _ in hits if cid not in self._blocked][: self.config.candidates]

    def retrieve(self, question: str) -> RetrievalResult:
        cfg = self.config
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        dense = self._dense(question, timings) if cfg.mode != "bm25" else []
        bm25 = self._bm25(question, timings) if cfg.mode != "dense" else []
        dense_rank = {cid: i for i, cid in enumerate(dense, start=1)}
        bm25_rank = {cid: i for i, cid in enumerate(bm25, start=1)}

        if cfg.mode == "dense":
            order = [(cid, 1.0 / (cfg.rrf_k + dense_rank[cid])) for cid in dense]
        elif cfg.mode == "bm25":
            order = [(cid, 1.0 / (cfg.rrf_k + bm25_rank[cid])) for cid in bm25]
        else:
            fused: dict[str, float] = {}
            for ranks in (dense_rank, bm25_rank):
                for cid, rank in ranks.items():
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
            # Ties broken by best single rank, then by chunk id, so the order is stable.
            order = sorted(
                fused.items(),
                key=lambda kv: (
                    -kv[1],
                    min(dense_rank.get(kv[0], 10**6), bm25_rank.get(kv[0], 10**6)),
                    kv[0],
                ),
            )

        candidates = [
            Candidate(
                chunk=self.store.get(cid),
                score=score,
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
            )
            for cid, score in order
        ]

        if cfg.mode == "hybrid_rerank":
            assert self.reranker is not None
            head = candidates[: cfg.rerank_top]
            start = time.perf_counter()
            scores = self.reranker.score(question, [c.chunk.embed_text for c in head])
            timings["rerank"] = (time.perf_counter() - start) * 1000
            reranked = [
                Candidate(c.chunk, s, c.dense_rank, c.bm25_rank, rerank=s)
                for c, s in zip(head, scores, strict=True)
            ]
            position = {c.chunk.chunk_id: i for i, c in enumerate(head)}
            reranked.sort(key=lambda c: (-c.score, position[c.chunk.chunk_id]))
            candidates = reranked

        top = candidates[0].score if candidates else None
        gated = False
        if cfg.mode == "hybrid_rerank" and cfg.gate_min_score is not None:
            gated = top is None or top < cfg.gate_min_score
        context: list[Candidate] = []
        if not gated:
            for candidate in candidates:
                if len(context) == cfg.context_k:
                    break
                if (
                    cfg.mode == "hybrid_rerank"
                    and cfg.keep_min_score is not None
                    and candidate.score < cfg.keep_min_score
                ):
                    break
                context.append(candidate)
        timings["retrieval_total"] = (time.perf_counter() - total_start) * 1000
        return RetrievalResult(
            mode=cfg.mode,
            ranked=candidates,
            context=context,
            top_score=top,
            gated=gated,
            timings_ms=timings,
        )
