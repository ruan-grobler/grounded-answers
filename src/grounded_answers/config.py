"""Paths, model settings and the two pipeline configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from grounded_answers.documents import ALL_FLAGS
from grounded_answers.retrieval import MODES, Mode, RetrievalConfig

PipelineName = Literal["naive", "grounded"]
PIPELINES: tuple[PipelineName, ...] = ("naive", "grounded")

DEFAULT_GENERATOR_MODEL = "claude-opus-5-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"


class ConfigError(ValueError):
    """config/pipelines.yaml or an environment variable is invalid."""


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    @property
    def questions(self) -> Path:
        return self.root / "eval" / "questions.yaml"

    @property
    def dev_questions(self) -> Path:
        return self.root / "eval" / "dev.yaml"

    @property
    def pipelines(self) -> Path:
        return self.root / "config" / "pipelines.yaml"

    @property
    def vector_cache(self) -> Path:
        return self.root / "index" / "embeddings.json"

    @property
    def score_cache(self) -> Path:
        return self.root / "index" / "rerank-scores.json"

    @property
    def models_dir(self) -> Path:
        return Path(os.environ.get("GA_MODELS_DIR", self.root / ".cache" / "models"))

    @property
    def index_db(self) -> Path:
        return self.root / ".cache" / "index.db"

    @property
    def cassettes(self) -> Path:
        return self.root / "fixtures" / "cassettes"

    @property
    def scripts(self) -> Path:
        return self.root / "fixtures" / "handcrafted" / "answers.yaml"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def find_root(start: Path | None = None) -> Path:
    """The nearest folder at or above `start` that has corpus/ and eval/."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "corpus").is_dir() and (folder / "eval").is_dir():
            return folder
    raise ConfigError(f"could not find the project root (corpus/ and eval/) above {here}")


@dataclass(frozen=True)
class PipelineConfig:
    name: PipelineName
    description: str
    retrieval: RetrievalConfig
    guardrails: bool
    prompt: str


def _pipeline(name: PipelineName, raw: dict[str, Any]) -> PipelineConfig:
    mode = raw.get("retrieval")
    if mode not in MODES:
        raise ConfigError(f"{name}: retrieval must be one of {', '.join(MODES)}")
    guardrails = bool(raw.get("guardrails", False))
    prompt = str(raw.get("prompt", name))
    if prompt not in ("naive", "grounded"):
        raise ConfigError(f"{name}: prompt must be naive or grounded")

    def number(key: str) -> float | None:
        value = raw.get(key)
        return None if value is None else float(value)

    retrieval_mode: Mode = mode
    return PipelineConfig(
        name=name,
        description=str(raw.get("description", "")).strip(),
        retrieval=RetrievalConfig(
            mode=retrieval_mode,
            candidates=int(raw.get("candidates", 20)),
            rrf_k=int(raw.get("rrf_k", 60)),
            rerank_top=int(raw.get("rerank_top", 20)),
            context_k=int(raw.get("context_k", 5)),
            keep_min_score=number("keep_min_score"),
            gate_min_score=number("gate_min_score"),
            blocked_flags=frozenset(ALL_FLAGS) if guardrails else frozenset(),
        ),
        guardrails=guardrails,
        prompt=prompt,
    )


def load_pipelines(path: Path) -> dict[PipelineName, PipelineConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [p for p in PIPELINES if p not in raw]
    if missing:
        raise ConfigError(f"{path} is missing pipelines: {', '.join(missing)}")
    return {name: _pipeline(name, raw[name]) for name in PIPELINES}


@dataclass(frozen=True)
class ModelSettings:
    generator_model: str = DEFAULT_GENERATOR_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    generator_effort: str = "low"
    judge_effort: str = "medium"
    fallbacks: bool = True
    budget_usd: float = 2.00

    @classmethod
    def from_env(cls) -> ModelSettings:
        fallbacks = os.environ.get("GA_FALLBACKS", "default").strip().lower()
        if fallbacks not in ("default", "off"):
            raise ConfigError("GA_FALLBACKS must be default or off")
        try:
            budget = float(os.environ.get("GA_BUDGET_USD", "2.00"))
        except ValueError as exc:
            raise ConfigError("GA_BUDGET_USD must be a number of US dollars") from exc
        return cls(
            generator_model=os.environ.get("GA_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL),
            judge_model=os.environ.get("GA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            fallbacks=fallbacks == "default",
            budget_usd=budget,
        )
