from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grounded_answers import inference
from grounded_answers.inference import (
    EMBED_DIM,
    CachedEmbedder,
    CachedReranker,
    CacheMiss,
    FastEmbedEmbedder,
    FastEmbedReranker,
    ModelUnavailable,
    ScoreCache,
    VectorCache,
    load_live_models,
    score_key,
    vector_key,
)
from tests.helpers import FakeEmbedder, FakeReranker


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).random(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_vector_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    cache = VectorCache(path)
    cache.put("passage", "hello", _vec(1))
    cache.put("query", "hello", _vec(2))
    assert len(cache) == 2
    cache.save()
    again = VectorCache(path)
    passage, query = again.get("passage", "hello"), again.get("query", "hello")
    assert passage is not None and query is not None
    assert np.array_equal(passage, _vec(1))
    assert np.array_equal(query, _vec(2))
    assert again.get("passage", "missing") is None
    key = vector_key(inference.EMBED_MODEL, "passage", "hello")
    assert again.entry(key)["preview"] == "hello"
    assert again.keys() == sorted(again.keys())


def test_vector_cache_rejects_other_models_and_bad_shapes(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps({"format": "x", "model": "other", "entries": {}}), encoding="utf-8")
    with pytest.raises(CacheMiss, match="expected"):
        VectorCache(path)
    with pytest.raises(ValueError, match="dimension"):
        VectorCache(tmp_path / "new.json").put("passage", "x", np.zeros(3, dtype=np.float32))


def test_save_is_a_no_op_when_clean(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path / "v.json")
    cache.save()
    assert not (tmp_path / "v.json").exists()


def test_prune(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path / "v.json")
    cache.put("passage", "keep", _vec(1))
    cache.put("passage", "drop", _vec(2))
    keep = {vector_key(cache.model, "passage", "keep")}
    assert cache.prune(keep) == 1
    assert cache.get("passage", "drop") is None


def test_score_cache_roundtrip_and_prune(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    cache = ScoreCache(path)
    cache.put("q", "passage text", 1.5)
    cache.save()
    again = ScoreCache(path)
    assert again.get("q", "passage text") == 1.5
    assert again.get("q", "other") is None
    key = score_key(again.model, "q", "passage text")
    assert again.entry(key)["query"] == "q"
    assert again.keys() == [key]
    assert again.prune(set()) == 1
    path.write_text(json.dumps({"format": "x", "model": "m", "entries": {}}), encoding="utf-8")
    with pytest.raises(CacheMiss):
        ScoreCache(path)


def test_cached_embedder_serves_cache_then_live(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path / "v.json")
    live = FakeEmbedder()
    embedder = CachedEmbedder(cache, live, record=True)
    first = embedder.embed(["a b", "c d"], "passage")
    assert live.calls == 2
    assert embedder.live_calls == 2
    second = embedder.embed(["a b", "c d"], "passage")
    assert live.calls == 2
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))
    assert len(embedder.used) == 2


def test_cached_embedder_without_live_raises(tmp_path: Path) -> None:
    embedder = CachedEmbedder(VectorCache(tmp_path / "v.json"), None, record=False)
    with pytest.raises(CacheMiss, match="make index"):
        embedder.embed(["unknown"], "query")


def test_cached_embedder_without_record_does_not_write(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path / "v.json")
    CachedEmbedder(cache, FakeEmbedder(), record=False).embed(["x"], "query")
    assert len(cache) == 0


def test_cached_reranker(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path / "s.json")
    live = FakeReranker()
    reranker = CachedReranker(cache, live, record=True)
    assert reranker.score("late fee", ["late fee R350", "gloves"]) == [1.0, -3.0]
    assert reranker.score("late fee", ["late fee R350", "gloves"]) == [1.0, -3.0]
    assert live.calls == 2
    with pytest.raises(CacheMiss, match="rerank"):
        CachedReranker(ScoreCache(tmp_path / "t.json"), None, record=False).score("q", ["p"])


def test_live_models_respect_mode(tmp_path: Path) -> None:
    assert load_live_models(tmp_path, "replay") == (None, None)
    assert load_live_models(tmp_path, "auto") == (None, None)
    with pytest.raises(ModelUnavailable, match="make models"):
        load_live_models(tmp_path, "live")


def test_fastembed_wrappers_report_missing_models(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailable):
        FastEmbedEmbedder(tmp_path)
    with pytest.raises(ModelUnavailable):
        FastEmbedReranker(tmp_path)


class _FakeModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen = texts
        return [[1.0] * EMBED_DIM for _ in texts]

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [float(len(d)) for d in docs]


def test_fastembed_wrappers_add_the_query_instruction() -> None:
    embedder = FastEmbedEmbedder.__new__(FastEmbedEmbedder)
    fake = _FakeModel()
    embedder._model = fake
    embedder.model_name = "m"
    vectors = embedder.embed(["what is the fee"], "query")
    assert fake.seen == [inference.QUERY_INSTRUCTION + "what is the fee"]
    assert vectors[0].dtype == np.float32
    embedder.embed(["passage"], "passage")
    assert fake.seen == ["passage"]
    reranker = FastEmbedReranker.__new__(FastEmbedReranker)
    reranker._model = fake
    assert reranker.score("q", ["ab", "abcd"]) == [2.0, 4.0]
