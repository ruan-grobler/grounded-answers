"""Local model inference (embeddings and reranking) with a committed, content addressed cache.

Both models run on CPU through fastembed (ONNX), free and offline once
downloaded. Every vector and every rerank score the eval needs is also stored
in `index/`, keyed by a SHA-256 of model + text. That gives two properties:

- CI and `make demo` need no model download: they replay the committed values.
- Results are identical on every machine, because a known input is always
  served from the cache. Live models only run for inputs the cache has not seen.

`ga bench` re-runs the live models on every cached input and reports the
largest difference, which is how the committed numbers can be checked.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from grounded_answers.jsonutil import read_json, sha256_hex, write_json

# Pinned models. Change these and the caches miss, by design.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# BGE v1.5 retrieval instruction, prepended to queries (not passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

VECTOR_FORMAT = "grounded-answers/vectors@1"
SCORE_FORMAT = "grounded-answers/rerank-scores@1"

Vector = npt.NDArray[np.float32]
Kind = Literal["passage", "query"]
ModelMode = Literal["replay", "auto", "live"]


class CacheMiss(LookupError):
    """A model output was needed that is not in the committed cache, and live models are off."""


class ModelUnavailable(RuntimeError):
    """The local model files are not downloaded (run `make models`)."""


# ------------------------------------------------------------------ embedders


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str], kind: Kind) -> list[Vector]: ...


class FastEmbedEmbedder:
    """BGE small via fastembed. Vectors come back L2 normalised (cosine = dot product)."""

    def __init__(self, models_dir: Path, *, allow_download: bool = False) -> None:
        self.model_name = EMBED_MODEL
        try:
            from fastembed import TextEmbedding

            self._model: Any = TextEmbedding(
                model_name=EMBED_MODEL,
                cache_dir=str(models_dir),
                local_files_only=not allow_download,
            )
        except Exception as exc:  # fastembed raises several types for a missing model
            raise ModelUnavailable(
                f"{EMBED_MODEL} is not available in {models_dir}. Run `make models` to download it "
                "(about 70 MB, free, no account needed)."
            ) from exc

    def embed(self, texts: Sequence[str], kind: Kind) -> list[Vector]:
        inputs = [QUERY_INSTRUCTION + t if kind == "query" else t for t in texts]
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(inputs)]


def _encode(vector: Vector) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def _decode(blob: str) -> Vector:
    return np.frombuffer(base64.b64decode(blob), dtype="<f4").astype(np.float32)


def vector_key(model: str, kind: Kind, text: str) -> str:
    return sha256_hex(f"{model}\n{kind}\n{text}")


class VectorCache:
    """JSON file of base64 float32 vectors. Sorted keys, so it diffs cleanly in git."""

    def __init__(self, path: Path, model: str = EMBED_MODEL, dim: int = EMBED_DIM) -> None:
        self.path = path
        self.model = model
        self.dim = dim
        self._entries: dict[str, dict[str, str]] = {}
        self.dirty = False
        if path.is_file():
            data = read_json(path)
            if data.get("format") != VECTOR_FORMAT or data.get("model") != model:
                raise CacheMiss(
                    f"{path} was written for {data.get('model')!r} in format "
                    f"{data.get('format')!r}, expected {model!r} in {VECTOR_FORMAT!r}"
                )
            self._entries = data["entries"]

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, key: str) -> dict[str, str]:
        return self._entries[key]

    def get(self, kind: Kind, text: str) -> Vector | None:
        entry = self._entries.get(vector_key(self.model, kind, text))
        return None if entry is None else _decode(entry["vector"])

    def put(self, kind: Kind, text: str, vector: Vector) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"expected a {self.dim} dimension vector, got {vector.shape}")
        self._entries[vector_key(self.model, kind, text)] = {
            "kind": kind,
            "preview": " ".join(text.split())[:80],
            "vector": _encode(vector),
        }
        self.dirty = True

    def prune(self, keep: set[str]) -> int:
        stale = [k for k in self._entries if k not in keep]
        for key in stale:
            del self._entries[key]
        self.dirty = self.dirty or bool(stale)
        return len(stale)

    def save(self) -> None:
        if not self.dirty:
            return
        write_json(
            self.path,
            {
                "format": VECTOR_FORMAT,
                "model": self.model,
                "dim": self.dim,
                "entries": self._entries,
            },
        )
        self.dirty = False


# ------------------------------------------------------------------ rerankers


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """MS MARCO MiniLM cross-encoder. Returns raw logits: above 0 means relevant."""

    def __init__(
        self, models_dir: Path, *, allow_download: bool = False, model_name: str = RERANK_MODEL
    ) -> None:
        self.model_name = model_name
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model: Any = TextCrossEncoder(
                model_name=model_name,
                cache_dir=str(models_dir),
                local_files_only=not allow_download,
            )
        except Exception as exc:
            raise ModelUnavailable(
                f"{model_name} is not available in {models_dir}. Run `make models` to download it "
                "(free, no account needed)."
            ) from exc

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(passages))]


def score_key(model: str, query: str, passage: str) -> str:
    return sha256_hex(f"{model}\n{query}\n{passage}")


class ScoreCache:
    def __init__(self, path: Path, model: str = RERANK_MODEL) -> None:
        self.path = path
        self.model = model
        self._entries: dict[str, dict[str, Any]] = {}
        self.dirty = False
        if path.is_file():
            data = read_json(path)
            if data.get("format") != SCORE_FORMAT or data.get("model") != model:
                raise CacheMiss(f"{path} was written for another reranker or format")
            self._entries = data["entries"]

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, key: str) -> dict[str, Any]:
        return self._entries[key]

    def get(self, query: str, passage: str) -> float | None:
        entry = self._entries.get(score_key(self.model, query, passage))
        return None if entry is None else float(entry["score"])

    def put(self, query: str, passage: str, score: float) -> None:
        self._entries[score_key(self.model, query, passage)] = {
            "query": query[:80],
            "passage": " ".join(passage.split())[:60],
            "score": score,
        }
        self.dirty = True

    def prune(self, keep: set[str]) -> int:
        stale = [k for k in self._entries if k not in keep]
        for key in stale:
            del self._entries[key]
        self.dirty = self.dirty or bool(stale)
        return len(stale)

    def save(self) -> None:
        if not self.dirty:
            return
        write_json(
            self.path,
            {"format": SCORE_FORMAT, "model": self.model, "entries": self._entries},
        )
        self.dirty = False


# --------------------------------------------------------- cache-first wrappers


class CachedEmbedder:
    """Serve from the cache. On a miss, run the live model if there is one, else raise."""

    def __init__(self, cache: VectorCache, live: Embedder | None = None, *, record: bool) -> None:
        self.cache = cache
        self.live = live
        self.record = record
        self.model_name = cache.model
        self.used: set[str] = set()
        self.live_calls = 0

    def embed(self, texts: Sequence[str], kind: Kind) -> list[Vector]:
        out: list[Vector | None] = [self.cache.get(kind, t) for t in texts]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            if self.live is None:
                sample = texts[missing[0]][:70]
                raise CacheMiss(
                    f"no cached {kind} vector for {len(missing)} text(s), for example "
                    f"{sample!r}. Run `make models` once, then `make index` to embed new text."
                )
            fresh = self.live.embed([texts[i] for i in missing], kind)
            self.live_calls += len(missing)
            for i, vector in zip(missing, fresh, strict=True):
                out[i] = vector
                if self.record:
                    self.cache.put(kind, texts[i], vector)
        for text in texts:
            self.used.add(vector_key(self.cache.model, kind, text))
        return [v for v in out if v is not None]


class CachedReranker:
    def __init__(self, cache: ScoreCache, live: Reranker | None = None, *, record: bool) -> None:
        self.cache = cache
        self.live = live
        self.record = record
        self.model_name = cache.model
        self.used: set[str] = set()
        self.live_calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        out: list[float | None] = [self.cache.get(query, p) for p in passages]
        missing = [i for i, s in enumerate(out) if s is None]
        if missing:
            if self.live is None:
                raise CacheMiss(
                    f"no cached rerank score for {len(missing)} passage(s) of query "
                    f"{query[:70]!r}. Run `make models` once, then `make index`."
                )
            fresh = self.live.score(query, [passages[i] for i in missing])
            self.live_calls += len(missing)
            for i, score in zip(missing, fresh, strict=True):
                out[i] = score
                if self.record:
                    self.cache.put(query, passages[i], score)
        for passage in passages:
            self.used.add(score_key(self.cache.model, query, passage))
        return [s for s in out if s is not None]


def models_downloaded(models_dir: Path) -> bool:
    """Both pinned models are in the local Hugging Face cache folder (no network check)."""
    if not models_dir.is_dir():
        return False
    names = [p.name.lower() for p in models_dir.iterdir()]
    wanted = ("bge-small-en-v1.5", RERANK_MODEL.split("/")[-1].lower())
    return all(any(w in n for n in names) for w in wanted)


def load_live_models(
    models_dir: Path, mode: ModelMode, *, allow_download: bool = False
) -> tuple[Embedder | None, Reranker | None]:
    """`replay` never loads models, `auto` only if already downloaded, `live` must load them."""
    if mode == "replay":
        return None, None
    if mode == "auto" and not models_downloaded(models_dir):
        return None, None
    try:
        return (
            FastEmbedEmbedder(models_dir, allow_download=allow_download),
            FastEmbedReranker(models_dir, allow_download=allow_download),
        )
    except ModelUnavailable:
        if mode == "live":
            raise
        return None, None
