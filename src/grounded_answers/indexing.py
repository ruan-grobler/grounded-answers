"""Build the search index from the corpus: ingest, chunk, flag, embed, store.

Idempotent: the index records a fingerprint of the corpus files, the chunking
settings and the model, and `build_index` does nothing when it matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grounded_answers.chunking import ChunkingConfig, chunk_document
from grounded_answers.documents import Chunk, Document
from grounded_answers.inference import EMBED_MODEL, CachedEmbedder
from grounded_answers.ingest import load_corpus
from grounded_answers.jsonutil import canonical_json, sha256_hex
from grounded_answers.scan import flag_chunks
from grounded_answers.store import STORE_VERSION, Store, StoreError


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    flagged: dict[str, list[str]]
    fingerprint: str
    rebuilt: bool


def prepare_chunks(
    corpus_dir: Path, root: Path, config: ChunkingConfig
) -> tuple[list[Document], list[Chunk]]:
    documents = load_corpus(corpus_dir, root)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return documents, flag_chunks(chunks)


def fingerprint(documents: list[Document], chunks: list[Chunk], config: ChunkingConfig) -> str:
    """Changes when a file, the chunking, the text the models see, the flags or the model change."""
    return sha256_hex(
        canonical_json(
            {
                "docs": [[d.doc_id, d.sha256] for d in documents],
                "chunking": [config.max_words, config.overlap_words],
                "chunks": [[c.chunk_id, c.text, c.embed_text, list(c.flags)] for c in chunks],
                "embed_model": EMBED_MODEL,
                "store_version": STORE_VERSION,
            }
        )
    )


def build_index(
    corpus_dir: Path,
    root: Path,
    db_path: Path,
    embedder: CachedEmbedder,
    config: ChunkingConfig | None = None,
    *,
    force: bool = False,
) -> IndexReport:
    config = config or ChunkingConfig()
    documents, chunks = prepare_chunks(corpus_dir, root, config)
    print_flags = {c.chunk_id: list(c.flags) for c in chunks if c.flags}
    fp = fingerprint(documents, chunks, config)
    if not force and db_path.is_file():
        try:
            store = Store(db_path)
            current = store.meta.get("fingerprint")
            store.close()
        except StoreError:
            current = None
        if current == fp:
            # Still touch every passage vector, so cache pruning knows they are in use.
            embedder.embed([c.embed_text for c in chunks], "passage")
            return IndexReport(len(documents), len(chunks), print_flags, fp, rebuilt=False)
    vectors = embedder.embed([c.embed_text for c in chunks], "passage")
    store = Store.build(
        db_path,
        chunks,
        vectors,
        {
            "fingerprint": fp,
            "embed_model": EMBED_MODEL,
            "max_words": str(config.max_words),
            "overlap_words": str(config.overlap_words),
            "documents": str(len(documents)),
        },
    )
    store.close()
    return IndexReport(len(documents), len(chunks), print_flags, fp, rebuilt=True)
