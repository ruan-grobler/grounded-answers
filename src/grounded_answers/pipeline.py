"""The two pipelines end to end: question in, structured answer with citations out.

naive:    dense top 5 -> plain prompt -> answer
grounded: personal data gate -> hybrid + rerank -> confidence gate
          -> hardened prompt -> output guard -> answer
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from grounded_answers.config import ModelSettings, PipelineConfig
from grounded_answers.documents import Chunk
from grounded_answers.generator import Answer, GenerationError, build_request, parse_answer
from grounded_answers.guardrails import OutputCheck, check_output, classify_query
from grounded_answers.llm import LLMClient, LLMError, Usage
from grounded_answers.pricing import cost_usd
from grounded_answers.prompts import label
from grounded_answers.retrieval import RetrievalResult, Retriever

Stage = Literal["query_guard", "retrieval_gate", "model", "output_guard", "error"]

REFUSE_PERSONAL = (
    "I can't share personal information about staff or patients. For HR matters, "
    "please speak to the practice manager."
)
REFUSE_NOT_FOUND = (
    "I couldn't find this in the practice documents, so I won't guess. "
    "Please check with the practice manager."
)
REFUSE_UNVERIFIED = (
    "I found related documents but could not verify an answer against them, so I won't guess. "
    "Please check with the practice manager."
)


@dataclass
class PipelineResult:
    pipeline: str
    question: str
    final_text: str
    refused: bool
    refusal_reason: str
    stage: Stage
    retrieval: RetrievalResult | None = None
    context: list[Chunk] = field(default_factory=list)
    answer: Answer | None = None
    output_check: OutputCheck | None = None
    guard_rule: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    cost_usd: float = 0.0
    llm_calls: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def labels(self) -> dict[str, Chunk]:
        return {label(i): c for i, c in enumerate(self.context)}

    @property
    def cited_chunks(self) -> list[Chunk]:
        """Chunks the answer cites, in order, without duplicates."""
        if self.answer is None or self.refused:
            return []
        labels = self.labels
        seen: list[Chunk] = []
        for claim in self.answer.claims:
            for source in claim.source_ids:
                chunk = labels.get(source)
                if chunk is not None and chunk not in seen:
                    seen.append(chunk)
        return seen


class Pipeline:
    def __init__(
        self, config: PipelineConfig, retriever: Retriever, settings: ModelSettings
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.settings = settings

    @property
    def name(self) -> str:
        return self.config.name

    def answer(self, question: str, llm: LLMClient) -> PipelineResult:
        start = time.perf_counter()
        result = self._answer(question, llm)
        result.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return result

    def _answer(self, question: str, llm: LLMClient) -> PipelineResult:
        guarded = self.config.guardrails
        if guarded:
            verdict = classify_query(question)
            if not verdict.allowed:
                return PipelineResult(
                    pipeline=self.name,
                    question=question,
                    final_text=REFUSE_PERSONAL,
                    refused=True,
                    refusal_reason="personal_data",
                    stage="query_guard",
                    guard_rule=verdict.rule,
                )

        retrieval = self.retriever.retrieve(question)
        timings = dict(retrieval.timings_ms)
        if retrieval.gated:
            return PipelineResult(
                pipeline=self.name,
                question=question,
                final_text=REFUSE_NOT_FOUND,
                refused=True,
                refusal_reason="not_in_documents",
                stage="retrieval_gate",
                retrieval=retrieval,
                timings_ms=timings,
            )

        context = [c.chunk for c in retrieval.context]
        request = build_request(self.settings, self.config.prompt, question, context)
        result = PipelineResult(
            pipeline=self.name,
            question=question,
            final_text="",
            refused=False,
            refusal_reason="none",
            stage="model",
            retrieval=retrieval,
            context=context,
            timings_ms=timings,
        )
        try:
            response = llm.create(request, role="generator")
        except LLMError as exc:
            result.stage, result.error = "error", str(exc)
            result.final_text, result.refused, result.refusal_reason = "", True, "error"
            return result
        result.llm_calls = 1
        result.usage = response.usage
        result.model = response.model or self.settings.generator_model
        result.cost_usd = cost_usd(result.model, response.usage) or 0.0
        if response.latency_ms is not None:
            result.timings_ms["generation"] = response.latency_ms
        try:
            answer = parse_answer(response)
        except GenerationError as exc:
            result.stage, result.error = "error", str(exc)
            result.final_text, result.refused, result.refusal_reason = "", True, "error"
            return result

        result.answer = answer
        result.final_text = answer.answer
        result.refused = answer.refused
        result.refusal_reason = answer.refusal_reason if answer.refused else "none"
        result.output_check = check_output(answer, result.labels)
        if guarded and result.output_check.blocked:
            result.stage = "output_guard"
            result.final_text = REFUSE_UNVERIFIED
            result.refused = True
            result.refusal_reason = "unverified"
        return result
