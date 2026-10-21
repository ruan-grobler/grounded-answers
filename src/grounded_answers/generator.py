"""The generator adapter: build a structured output request for Claude and parse the answer.

Output is forced into ANSWER_SCHEMA with `output_config.format`, then validated
again in code. Anything malformed, cut off or declined is a `GenerationError`,
never a silent pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator

from grounded_answers.config import ModelSettings
from grounded_answers.documents import Chunk
from grounded_answers.llm import LLMResponse
from grounded_answers.prompts import SYSTEM_PROMPTS, user_message

RefusalReason = Literal["none", "not_in_documents", "personal_data", "unsafe_request"]
Confidence = Literal["high", "medium", "low"]

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["statement", "source_ids", "quote"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "refused": {"type": "boolean"},
        "refusal_reason": {
            "type": "string",
            "enum": ["none", "not_in_documents", "personal_data", "unsafe_request"],
        },
    },
    "required": ["answer", "claims", "confidence", "refused", "refusal_reason"],
    "additionalProperties": False,
}
_VALIDATOR = Draft202012Validator(ANSWER_SCHEMA)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class GenerationError(ValueError):
    """The model's output could not be used: declined, cut off, not JSON, or the wrong shape."""


@dataclass(frozen=True)
class Claim:
    statement: str
    source_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class Answer:
    answer: str
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    confidence: Confidence = "low"
    refused: bool = False
    refusal_reason: RefusalReason = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [
                {"statement": c.statement, "source_ids": list(c.source_ids), "quote": c.quote}
                for c in self.claims
            ],
            "confidence": self.confidence,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Answer:
        errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: list(e.path))
        if errors:
            where = "/".join(str(p) for p in errors[0].path) or "(root)"
            raise GenerationError(f"answer does not match the schema at {where}: {errors[0].message}")
        return cls(
            answer=data["answer"],
            claims=tuple(
                Claim(c["statement"], tuple(c["source_ids"]), c["quote"]) for c in data["claims"]
            ),
            confidence=data["confidence"],
            refused=data["refused"],
            refusal_reason=data["refusal_reason"],
        )


def build_request(
    settings: ModelSettings, prompt: str, question: str, context: list[Chunk]
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": settings.generator_model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPTS[prompt],
        "messages": [{"role": "user", "content": user_message(prompt, question, context)}],
        "output_config": {
            "effort": settings.generator_effort,
            "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
        },
    }
    if settings.fallbacks:
        # If a safety classifier declines, the API retries on Anthropic's recommended
        # fallback model inside the same call instead of returning a refusal.
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def parse_answer(response: LLMResponse) -> Answer:
    if response.stop_reason == "refusal":
        raise GenerationError("the model declined the request (stop_reason refusal)")
    if response.stop_reason == "max_tokens":
        raise GenerationError("the answer was cut off at max_tokens")
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"the answer is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise GenerationError("the answer is not a JSON object")
    return Answer.from_dict(data)
