"""One SQLite file holds both indexes: FTS5 for BM25 and sqlite-vec for dense vectors.

The index is rebuilt into a temporary file and swapped in with an atomic rename,
so a failed build never leaves a half written index behind. Rebuilding is cheap
because every vector comes from the content addressed cache.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import sqlite_vec

from grounded_answers.documents import Chunk
from grounded_answers.inference import EMBED_DIM, Vector

STORE_VERSION = "1"

# Small English stopword list for the BM25 query. FTS5's porter tokenizer stems
# both sides, so "cancelling" in a question matches "cancellation" in a policy.
STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been before being but by can could
    did do does doing for from get gets got had has have having he her here hers him his how
    i if in into is it its just me might more most must my no nor not of off on once only or
    other our ours out over own same she should so some such than that the their theirs them
    then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours okay ok""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


class StoreError(RuntimeError):
    """The index is missing, stale or could not be opened."""


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        raise StoreError(
            "this Python's sqlite3 cannot load extensions, which sqlite-vec needs. "
            "Use a Python built with loadable sqlite extensions (python.org, Homebrew, "
            "or actions/setup-python all are)."
        ) from exc
    return db


def query_terms(question: str) -> list[str]:
    seen: list[str] = []
    for token in _TOKEN.findall(question.lower()):
        if len(token) < 2 and not token.isdigit():
            continue
        if token not in STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def fts_query(question: str) -> str | None:
    terms = query_terms(question)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


class Store:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise StoreError(f"no index at {path}. Run `ga index` first.")
        self.path = path
        self.db = _connect(path)
        self.meta = dict(self.db.execute("SELECT key, value FROM meta").fetchall())
        if self.meta.get("store_version") != STORE_VERSION:
            raise StoreError(f"{path} is from another version. Run `ga index` to rebuild it.")
        self._chunks: dict[str, Chunk] = {}
        self._by_rowid: dict[int, str] = {}
        rows = self.db.execute(
            "SELECT id, chunk_id, doc_id, doc_title, section, ordinal, text, page, flags "
            "FROM chunks ORDER BY id"
        )
        for rowid, chunk_id, doc_id, title, section, ordinal, text, page, flags in rows:
            self._chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=title,
                section=section,
                ordinal=ordinal,
                text=text,
                page=page,
                flags=tuple(json.loads(flags)),
            )
            self._by_rowid[rowid] = chunk_id

    def close(self) -> None:
        self.db.close()

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    @classmethod
    def build(
        cls,
        path: Path,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        meta: dict[str, str],
    ) -> Store:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".index.", suffix=".db")
        os.close(fd)
        tmp = Path(tmp_name)
        db: sqlite3.Connection | None = None
        try:
            db = _connect(tmp)
            db.executescript(
                f"""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    flags TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    header, text, content='', tokenize='porter unicode61'
                );
                CREATE VIRTUAL TABLE chunks_vec USING vec0(
                    embedding float[{EMBED_DIM}] distance_metric=cosine
                );
                """
            )
            for rowid, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
                db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rowid,
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.doc_title,
                        chunk.section,
                        chunk.ordinal,
                        chunk.text,
                        chunk.page,
                        json.dumps(list(chunk.flags)),
                    ),
                )
                db.execute(
                    "INSERT INTO chunks_fts (rowid, header, text) VALUES (?, ?, ?)",
                    (rowid, chunk.header, chunk.text),
                )
                db.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, np.asarray(vector, dtype=np.float32).tobytes()),
                )
            all_meta = {**meta, "store_version": STORE_VERSION, "chunks": str(len(chunks))}
            db.executemany("INSERT INTO meta VALUES (?, ?)", sorted(all_meta.items()))
            db.commit()
            db.close()
            db = None
            os.replace(tmp, path)
        except BaseException:
            if db is not None:
                db.close()
            for leftover in (tmp, tmp.with_name(tmp.name + "-journal")):
                leftover.unlink(missing_ok=True)
            raise
        return cls(path)

    def bm25(self, question: str, limit: int) -> list[tuple[str, float]]:
        """BM25 over the section path and text. Higher score is better."""
        match = fts_query(question)
        if match is None:
            return []
        rows = self.db.execute(
            "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY rank, rowid LIMIT ?",
            (match, limit),
        ).fetchall()
        # SQLite returns BM25 as a negative number (more negative = better match).
        return [(self._by_rowid[rowid], -float(rank)) for rowid, rank in rows]

    def knn(self, vector: Vector, limit: int) -> list[tuple[str, float]]:
        """Nearest chunks by cosine similarity. Higher score is better."""
        rows = self.db.execute(
            "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (np.asarray(vector, dtype=np.float32).tobytes(), limit),
        ).fetchall()
        # Exact (brute force) search; ties broken by rowid so the order is stable.
        rows.sort(key=lambda r: (r[1], r[0]))
        return [(self._by_rowid[rowid], 1.0 - float(distance)) for rowid, distance in rows]

    def filtered(self, results: Iterable[tuple[str, float]], blocked: set[str]) -> list[str]:
        return [cid for cid, _ in results if not blocked.intersection(self.get(cid).flags)]
