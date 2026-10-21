"""The one seam between the pipeline and Claude.

The generator and the judge both call `LLMClient.create(request, role=...)`,
where `request` is the exact keyword arguments for the Messages API. Replay,
record and live modes are different `LLMClient` implementations behind it, so
the code under test is the same code that runs live.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Role = Literal["generator", "judge"]


class LLMError(RuntimeError):
    """A model call failed after the SDK's own retries, or was declined."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        return cls(
            int(data.get("input_tokens") or 0),
            int(data.get("output_tokens") or 0),
            int(data.get("cache_creation_input_tokens") or 0),
            int(data.get("cache_read_input_tokens") or 0),
        )


@dataclass(frozen=True)
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    model: str
    latency_ms: float | None = None  # wall clock, only known for recorded or live calls

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMResponse:
        latency = data.get("latency_ms")
        return cls(
            content=[_keep(b) for b in data.get("content") or []],
            stop_reason=data.get("stop_reason") or "end_turn",
            usage=Usage.from_dict(data.get("usage") or {}),
            model=data.get("model", ""),
            latency_ms=None if latency is None else float(latency),
        )


def _keep(block: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields we read. Thinking blocks are dropped: this harness never replays them."""
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": block.get("text", "")}
    return {"type": str(kind)}


class LLMClient(Protocol):
    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse: ...


class AnthropicClient:
    """Live Claude API client. Only built in `record` or `live` mode.

    The SDK retries connection errors, 408, 409, 429 and 5xx with exponential
    backoff. `max_retries` raises that from the SDK default of 2 to 4. Requests
    that carry `betas` (server side refusal fallbacks) go to the beta endpoint.
    """

    def __init__(self, *, max_retries: int = 4, timeout_s: float = 120.0) -> None:
        import anthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError(
                "record and live modes call the Claude API: export ANTHROPIC_API_KEY first "
                "(replay mode, the default, needs no key)"
            )
        self._sdk = anthropic
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)

    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse:
        sdk = self._sdk
        start = time.perf_counter()
        try:
            if "betas" in request:
                message: Any = self._client.beta.messages.create(**request)
            else:
                message = self._client.messages.create(**request)
        except sdk.AuthenticationError as exc:
            raise LLMError("the Claude API rejected the credentials (check the key)") from exc
        except sdk.BadRequestError as exc:
            raise LLMError(f"{role} request was rejected as invalid: {exc.message}") from exc
        except sdk.RateLimitError as exc:
            raise LLMError(f"{role} request still rate limited after retries") from exc
        except sdk.APIStatusError as exc:
            raise LLMError(f"{role} request failed with HTTP {exc.status_code}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMError(f"{role} request could not reach the Claude API") from exc
        latency = (time.perf_counter() - start) * 1000
        data = message.to_dict()
        data["latency_ms"] = round(latency, 1)
        return LLMResponse.from_dict(data)
