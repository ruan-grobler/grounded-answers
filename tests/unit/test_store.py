from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from grounded_answers.documents import Chunk
from grounded_answers.store import Store, StoreError, fts_query, query_terms
from tests.helpers import FakeEmbedder, build_store, chunk


def _chunks() -> list[Chunk]:
    return [
        chunk("fees#00", "A late cancellation is charged a R350 fee.", section="Cancellations"),
        chunk("fees#01", "Scale and polish costs R890.", section="Prices"),
        chunk(
            "staff#00", "Megan Adams is on extension 107.", section="Work contacts", flags=("x",)
        ),
    ]


def test_query_terms_drop_stopwords_and_single_letters() -> None:
    assert query_terms("What's the fee if a patient is cancelling late?") == [
        "fee",
        "patient",
        "cancelling",
        "late",
    ]
    assert query_terms("Is it 5 or 6?") == ["5", "6"]
    assert fts_query("the and of") is None
    assert fts_query("late fee") == '"late" OR "fee"'


def test_build_and_search(tmp_path: Path) -> None:
    store = build_store(tmp_path, _chunks())
    assert len(store.chunks) == 3
    assert store.get("staff#00").flags == ("x",)
    assert store.meta["chunks"] == "3"
    bm25 = store.bm25("How much is a late cancellation?", 5)
    assert bm25[0][0] == "fees#00"
    assert bm25[0][1] > 0
    vector = FakeEmbedder().embed(["extension Megan"], "query")[0]
    knn = store.knn(vector, 3)
    assert knn[0][0] == "staff#00"
    assert knn[0][1] == pytest.approx(max(s for _, s in knn))
    assert store.filtered(knn, {"x"}) == [cid for cid, _ in knn if cid != "staff#00"]
    assert store.bm25("the of and", 5) == []
    store.close()


def test_porter_stemming_matches_word_forms(tmp_path: Path) -> None:
    store = build_store(tmp_path, _chunks())
    assert store.bm25("cancelled", 5)[0][0] == "fees#00"


def test_missing_and_stale_index(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="ga index"):
        Store(tmp_path / "none.db")
    store = build_store(tmp_path, _chunks())
    store.db.execute("UPDATE meta SET value = '0' WHERE key = 'store_version'")
    store.db.commit()
    store.close()
    with pytest.raises(StoreError, match="another version"):
        Store(tmp_path / "index.db")


def test_failed_build_leaves_no_file(tmp_path: Path) -> None:
    chunks = _chunks()
    bad = [np.zeros(3, dtype=np.float32)] * len(chunks)  # wrong dimension
    with pytest.raises(sqlite3.Error):
        Store.build(tmp_path / "index.db", chunks, bad, {})
    assert not (tmp_path / "index.db").exists()
    assert not list(tmp_path.glob(".index.*"))
    with pytest.raises(ValueError, match="one vector per chunk"):
        Store.build(tmp_path / "index.db", chunks, bad[:1], {})


def test_rebuild_replaces_atomically(tmp_path: Path) -> None:
    build_store(tmp_path, _chunks()).close()
    store = build_store(tmp_path, _chunks()[:1])
    assert [c.chunk_id for c in store.chunks] == ["fees#00"]
