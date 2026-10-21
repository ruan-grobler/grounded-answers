"""Deterministic fakes for the models, so unit tests never load ONNX or touch the network."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from grounded_answers.documents import Chunk
from grounded_answers.inference import EMBED_DIM, Kind, Vector
from grounded_answers.llm import LLMResponse, Role, Usage
from grounded_answers.store import STOPWORDS, Store

ROOT = Path(__file__).resolve().parents[1]
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class FakeEmbedder:
    """Hashed bag of words: texts that share words get similar vectors."""

    model_name = "fake-embedder"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str], kind: Kind) -> list[Vector]:
        self.calls += len(texts)
        out = []
        for text in texts:
            vector = np.zeros(EMBED_DIM, dtype=np.float32)
            for word in words(text):
                index = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM
                vector[index] += 1.0
            norm = float(np.linalg.norm(vector))
            out.append(vector / norm if norm else vector)
        return out


class FakeReranker:
    """Score = 2 x shared words minus 3, so unrelated passages score below zero."""

    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += len(passages)
        q = set(words(query))
        return [2.0 * len(q & set(words(p))) - 3.0 for p in passages]


def chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str | None = None,
    section: str = "Section",
    title: str = "Doc",
    flags: tuple[str, ...] = (),
    page: int | None = None,
) -> Chunk:
    doc = doc_id or chunk_id.split("#")[0]
    return Chunk(chunk_id, doc, title, section, 0, text, page, flags)


def build_store(tmp_path: Path, chunks: list[Chunk]) -> Store:
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.embed_text for c in chunks], "passage")
    return Store.build(tmp_path / "index.db", chunks, vectors, {"fingerprint": "test"})


class FakeLLM:
    """Returns queued bodies (dicts become JSON text) and records every request."""

    def __init__(
        self, *bodies: Any, stop_reason: str = "end_turn", model: str = "claude-opus-5-5"
    ) -> None:
        self.bodies = list(bodies)
        self.requests: list[tuple[Role, dict[str, Any]]] = []
        self.stop_reason = stop_reason
        self.model = model

    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse:
        self.requests.append((role, request))
        body = self.bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        text = body if isinstance(body, str) else json.dumps(body)
        return LLMResponse(
            content=[{"type": "text", "text": text}],
            stop_reason=self.stop_reason,
            usage=Usage(input_tokens=1000, output_tokens=100),
            model=self.model,
            latency_ms=1234.0,
        )


def answer_body(
    answer: str = "R350 [S1]",
    claims: list[dict[str, Any]] | None = None,
    *,
    refused: bool = False,
    reason: str = "none",
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "answer": answer,
        "claims": claims
        if claims is not None
        else [
            {
                "statement": "The fee is R350.",
                "source_ids": ["S1"],
                "quote": "R350 late cancellation fee",
            }
        ],
        "confidence": confidence,
        "refused": refused,
        "refusal_reason": reason,
    }
