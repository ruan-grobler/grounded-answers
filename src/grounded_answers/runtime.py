"""Wire it together: caches, local models, the index, retrievers, pipelines and LLM clients."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from grounded_answers.cassette import (
    Cassette,
    RecordingClient,
    ReplayClient,
    cassette_path,
)
from grounded_answers.config import (
    PIPELINES,
    ModelSettings,
    Paths,
    PipelineConfig,
    PipelineName,
    load_pipelines,
)
from grounded_answers.evalset import Question
from grounded_answers.indexing import IndexReport, build_index
from grounded_answers.inference import (
    CachedEmbedder,
    CachedReranker,
    ModelMode,
    ScoreCache,
    VectorCache,
    load_live_models,
)
from grounded_answers.llm import AnthropicClient, LLMClient
from grounded_answers.pipeline import Pipeline
from grounded_answers.retrieval import Mode, RetrievalConfig, Retriever
from grounded_answers.store import Store

LLMMode = Literal["replay", "record", "live"]


@dataclass
class Runtime:
    paths: Paths
    store: Store
    embedder: CachedEmbedder
    reranker: CachedReranker
    configs: dict[PipelineName, PipelineConfig]
    pipelines: dict[PipelineName, Pipeline]
    settings: ModelSettings
    live_models: bool
    index_report: IndexReport

    def retriever(self, config: RetrievalConfig) -> Retriever:
        return Retriever(self.store, self.embedder, self.reranker, config)

    def ablation_retriever(self, mode: Mode) -> Retriever:
        """The same candidate settings as the grounded pipeline, no gate, no flag filtering."""
        base = self.configs["grounded"].retrieval
        return self.retriever(
            RetrievalConfig(
                mode=mode,
                candidates=base.candidates,
                rrf_k=base.rrf_k,
                rerank_top=base.rerank_top,
                context_k=base.context_k,
            )
        )

    def save_caches(self) -> None:
        self.embedder.cache.save()
        self.reranker.cache.save()

    def close(self) -> None:
        self.store.close()


def open_runtime(
    root: Path,
    *,
    model_mode: ModelMode = "auto",
    record_models: bool = False,
    allow_download: bool = False,
    force_index: bool = False,
    settings: ModelSettings | None = None,
) -> Runtime:
    paths = Paths(root)
    embed_live, rerank_live = load_live_models(
        paths.models_dir, model_mode, allow_download=allow_download
    )
    embedder = CachedEmbedder(VectorCache(paths.vector_cache), embed_live, record=record_models)
    reranker = CachedReranker(ScoreCache(paths.score_cache), rerank_live, record=record_models)
    report = build_index(paths.corpus, root, paths.index_db, embedder, force=force_index)
    store = Store(paths.index_db)
    configs = load_pipelines(paths.pipelines)
    settings = settings or ModelSettings.from_env()
    pipelines = {
        name: Pipeline(
            configs[name], Retriever(store, embedder, reranker, configs[name].retrieval), settings
        )
        for name in PIPELINES
    }
    return Runtime(
        paths=paths,
        store=store,
        embedder=embedder,
        reranker=reranker,
        configs=configs,
        pipelines=pipelines,
        settings=settings,
        live_models=embed_live is not None,
        index_report=report,
    )


Finish = Callable[[bool], None]


def client_factory(
    paths: Paths, pipeline: str, mode: LLMMode
) -> Callable[[Question], tuple[LLMClient, Finish]]:
    """Per question LLM clients: replay from cassettes, record new ones, or call live."""
    live = AnthropicClient() if mode in ("record", "live") else None

    def make(question: Question) -> tuple[LLMClient, Finish]:
        path = cassette_path(paths.cassettes, pipeline, question.id)
        if mode == "replay":
            if path.is_file():
                cassette = Cassette.load(path)
            else:
                cassette = Cassette(pipeline, question.id, question.question, "hand-crafted", "")
            return ReplayClient(cassette), lambda ok: None
        assert live is not None
        if mode == "live":
            return live, lambda ok: None
        today = dt.date.today().isoformat()
        cassette = Cassette(
            pipeline,
            question.id,
            question.question,
            "recorded",
            f"Recorded from the Claude API on {today} by `make record`.",
        )
        recorder = RecordingClient(live, cassette)

        def finish(ok: bool) -> None:
            # Only complete, error free questions become cassettes, so a failed
            # recording never replaces a good fixture with half a conversation.
            if ok and cassette.interactions:
                cassette.save(path)
            elif ok and path.exists():
                path.unlink()

        return recorder, finish

    return make
