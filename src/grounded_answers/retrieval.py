"""Retrieval: dense, BM25, hybrid (reciprocal rank fusion) and hybrid plus cross-encoder rerank.

Multi-part questions ("Who is the infection control lead, and what is their
extension?") need evidence from two places, and a cross-encoder scoring each
chunk against the whole question under-rates a chunk that answers only one
part. With `decompose` on, the question is split into its parts, each part is
retrieved and reranked on its own, a chunk keeps its best score over the
parts, and the prompt gets the best chunk for every part before the rest.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from grounded_answers.documents import Chunk
from grounded_answers.inference import Embedder, Reranker
from grounded_answers.store import Store

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")

# Split points: a question mark or full stop followed by more text, or "and" /
# ", and" followed by a question word. "debit and credit cards" is not split.
_SPLIT = re.compile(
    r"(?<=[?.])\s+(?=\S)"
    r"|,?\s+and\s+(?=(?:what|who|whom|whose|which|where|when|why|how|is|are|can|do|does|"
    r"did|should|must|will)\b)",
    re.IGNORECASE,
)


def sub_queries(question: str) -> list[str]:
    """The whole question first, then its parts if it has more than one."""
    parts = [p.strip(" ,") for p in _SPLIT.split(question.strip())]
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) < 2:
        return [question]
    out = [question]
    for part in parts:
        if part not in out:
            out.append(part)
    return out


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "hybrid_rerank"
    candidates: int = 20  # taken from each retriever, per query, before fusion
    rrf_k: int = 60
    rerank_top: int = 20  # fused candidates per query sent to the cross-encoder
    context_k: int = 5
    keep_min_score: float | None = None  # rerank logit a chunk needs to reach the prompt
    gate_min_score: float | None = None  # refuse without calling the model below this
    decompose: bool = False
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
    best_query: int = 0  # index into RetrievalResult.queries of the part it scored best on


@dataclass
class RetrievalResult:
    mode: Mode
    ranked: list[Candidate]
    context: list[Candidate]
    top_score: float | None
    gated: bool
    queries: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _add(timings: dict[str, float], key: str, start: float) -> None:
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start) * 1000


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

    def _dense(self, query: str, timings: dict[str, float]) -> list[str]:
        start = time.perf_counter()
        vector = self.embedder.embed([query], "query")[0]
        _add(timings, "embed_query", start)
        start = time.perf_counter()
        hits = self.store.knn(vector, self.config.candidates + len(self._blocked))
        _add(timings, "vector_search", start)
        return [cid for cid, _ in hits if cid not in self._blocked][: self.config.candidates]

    def _bm25(self, query: str, timings: dict[str, float]) -> list[str]:
        start = time.perf_counter()
        hits = self.store.bm25(query, self.config.candidates + len(self._blocked))
        _add(timings, "bm25_search", start)
        return [cid for cid, _ in hits if cid not in self._blocked][: self.config.candidates]

    def _fuse(
        self, dense: list[str], bm25: list[str]
    ) -> tuple[list[tuple[str, float]], dict[str, int], dict[str, int]]:
        cfg = self.config
        dense_rank = {cid: i for i, cid in enumerate(dense, start=1)}
        bm25_rank = {cid: i for i, cid in enumerate(bm25, start=1)}
        if cfg.mode == "dense":
            return [(c, 1.0 / (cfg.rrf_k + dense_rank[c])) for c in dense], dense_rank, bm25_rank
        if cfg.mode == "bm25":
            return [(c, 1.0 / (cfg.rrf_k + bm25_rank[c])) for c in bm25], dense_rank, bm25_rank
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
        return order, dense_rank, bm25_rank

    def retrieve(self, question: str) -> RetrievalResult:
        cfg = self.config
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        queries = sub_queries(question) if cfg.decompose else [question]

        per_query: list[list[Candidate]] = []
        for query in queries:
            dense = self._dense(query, timings) if cfg.mode != "bm25" else []
            bm25 = self._bm25(query, timings) if cfg.mode != "dense" else []
            order, dense_rank, bm25_rank = self._fuse(dense, bm25)
            per_query.append(
                [
                    Candidate(
                        chunk=self.store.get(cid),
                        score=score,
                        dense_rank=dense_rank.get(cid),
                        bm25_rank=bm25_rank.get(cid),
                    )
                    for cid, score in order
                ]
            )

        if cfg.mode == "hybrid_rerank":
            ranked, best_per_query = self._rerank(queries, per_query, timings)
        else:
            # Without a reranker, several parts are merged by summing their RRF scores.
            merged: dict[str, Candidate] = {}
            for candidates in per_query:
                for c in candidates:
                    old = merged.get(c.chunk.chunk_id)
                    merged[c.chunk.chunk_id] = (
                        c
                        if old is None
                        else Candidate(c.chunk, old.score + c.score, old.dense_rank, old.bm25_rank)
                    )
            first = {c.chunk.chunk_id: i for i, c in enumerate(per_query[0])}
            ranked = sorted(
                merged.values(),
                key=lambda c: (-c.score, first.get(c.chunk.chunk_id, 10**6), c.chunk.chunk_id),
            )
            best_per_query = []

        top = ranked[0].score if ranked else None
        gated = False
        if cfg.mode == "hybrid_rerank" and cfg.gate_min_score is not None:
            gated = top is None or top < cfg.gate_min_score
        context = [] if gated else self._select(ranked, best_per_query)
        timings["retrieval_total"] = (time.perf_counter() - total_start) * 1000
        return RetrievalResult(
            mode=cfg.mode,
            ranked=ranked,
            context=context,
            top_score=top,
            gated=gated,
            queries=queries,
            timings_ms=timings,
        )

    def _rerank(
        self,
        queries: list[str],
        per_query: list[list[Candidate]],
        timings: dict[str, float],
    ) -> tuple[list[Candidate], list[Candidate]]:
        assert self.reranker is not None
        cfg = self.config
        # The union of each part's top fused candidates, in first seen order.
        pool: dict[str, Candidate] = {}
        for candidates in per_query:
            for c in candidates[: cfg.rerank_top]:
                pool.setdefault(c.chunk.chunk_id, c)
        heads = list(pool.values())
        texts = [c.chunk.embed_text for c in heads]
        best: dict[str, tuple[float, int]] = {}
        best_per_query: list[Candidate] = []
        start = time.perf_counter()
        for qi, query in enumerate(queries):
            scores = self.reranker.score(query, texts)
            top_for_query: tuple[float, int] | None = None
            for hi, score in enumerate(scores):
                cid = heads[hi].chunk.chunk_id
                if cid not in best or score > best[cid][0]:
                    best[cid] = (score, qi)
                if top_for_query is None or score > top_for_query[0]:
                    top_for_query = (score, hi)
            if top_for_query is not None and qi > 0:
                h = heads[top_for_query[1]]
                best_per_query.append(
                    Candidate(
                        h.chunk, top_for_query[0], h.dense_rank, h.bm25_rank, top_for_query[0], qi
                    )
                )
        _add(timings, "rerank", start)
        position = {c.chunk.chunk_id: i for i, c in enumerate(heads)}
        ranked = [
            Candidate(
                c.chunk,
                best[c.chunk.chunk_id][0],
                c.dense_rank,
                c.bm25_rank,
                best[c.chunk.chunk_id][0],
                best[c.chunk.chunk_id][1],
            )
            for c in heads
        ]
        ranked.sort(key=lambda c: (-c.score, position[c.chunk.chunk_id]))
        return ranked, best_per_query

    def _select(self, ranked: list[Candidate], best_per_query: list[Candidate]) -> list[Candidate]:
        """The best chunk for every part of the question first, then the highest scores."""
        cfg = self.config
        threshold = cfg.keep_min_score if cfg.mode == "hybrid_rerank" else None
        chosen: list[Candidate] = []
        seen: set[str] = set()
        for candidate in [*ranked[:1], *best_per_query, *ranked]:
            if len(chosen) == cfg.context_k:
                break
            cid = candidate.chunk.chunk_id
            if cid in seen:
                continue
            if threshold is not None and candidate.score < threshold:
                continue
            chosen.append(next(c for c in ranked if c.chunk.chunk_id == cid))
            seen.add(cid)
        return chosen
