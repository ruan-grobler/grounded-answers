from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from grounded_answers.cassette import (
    Cassette,
    CassetteMiss,
    RecordingClient,
    ReplayClient,
    cassette_path,
    fingerprint,
)
from grounded_answers.llm import AnthropicClient, LLMError, LLMResponse, Usage
from tests.helpers import FakeLLM, answer_body


def _request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": "claude-opus-5-5",
        "max_tokens": 10,
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "low"},
    }
    request.update(overrides)
    return request


def test_usage_and_response_roundtrip() -> None:
    usage = Usage(1, 2, 3, 4) + Usage(1, 1, 1, 1)
    assert usage.to_dict() == {
        "input_tokens": 2,
        "output_tokens": 3,
        "cache_creation_input_tokens": 4,
        "cache_read_input_tokens": 5,
    }
    response = LLMResponse(
        [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": " ok "}],
        "end_turn",
        usage,
        "m",
        12.5,
    )
    again = LLMResponse.from_dict(response.to_dict())
    assert again.text == "ok"
    assert again.content[0] == {"type": "thinking"}
    assert again.latency_ms == 12.5
    assert LLMResponse.from_dict({"content": []}).stop_reason == "end_turn"


def test_record_then_replay(tmp_path: Path) -> None:
    cassette = Cassette("grounded", "a01", "Q?", "recorded", "note")
    recorder = RecordingClient(FakeLLM(answer_body()), cassette)
    recorder.create(_request(), role="generator")
    path = cassette_path(tmp_path, "grounded", "a01")
    cassette.save(path)
    replay = ReplayClient(Cassette.load(path))
    assert replay.unused == 1
    assert replay.create(_request(), role="generator").text.startswith("{")
    assert replay.unused == 0
    with pytest.raises(CassetteMiss, match="no recorded generator call is left"):
        replay.create(_request(), role="generator")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"model": "claude-sonnet-5-5"}, "model changed"),
        ({"system": "other"}, "system prompt changed"),
        ({"output_config": {"effort": "high"}}, "output schema or effort"),
        ({"messages": [{"role": "user", "content": "bye"}]}, "retrieved context changed"),
        ({"max_tokens": 99}, "request parameters"),
    ],
)
def test_replay_explains_misses(change: dict[str, Any], reason: str) -> None:
    cassette = Cassette("grounded", "a01", "Q?", "hand-crafted", "")
    RecordingClient(FakeLLM(answer_body()), cassette).create(_request(), role="generator")
    with pytest.raises(CassetteMiss, match=reason):
        ReplayClient(cassette).create(_request(**change), role="generator")


def test_fingerprint_is_stable() -> None:
    assert fingerprint(_request()) == fingerprint(_request())
    assert fingerprint(_request())["hash"] != fingerprint(_request(max_tokens=11))["hash"]


def test_cassette_format_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss, match="make record"):
        Cassette.load(tmp_path / "none.json")
    with pytest.raises(CassetteMiss, match="unsupported"):
        Cassette.from_dict({"format": "old"})


def test_anthropic_client_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError, match="export ANTHROPIC_API_KEY"):
        AnthropicClient()


class _Message:
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 6},
            "content": [{"type": "text", "text": "{}"}],
        }


def _fake_sdk(raise_with: type[Exception] | None = None) -> types.ModuleType:
    sdk = types.ModuleType("anthropic")

    class APIError(Exception):
        message = "bad"
        status_code = 500

    for name in (
        "AuthenticationError",
        "BadRequestError",
        "RateLimitError",
        "APIStatusError",
        "APIConnectionError",
    ):
        setattr(sdk, name, type(name, (APIError,), {}))
    calls: list[str] = []

    class _Endpoint:
        def __init__(self, label: str) -> None:
            self.label = label

        def create(self, **kwargs: Any) -> _Message:
            calls.append(self.label)
            if raise_with is not None:
                raise getattr(sdk, raise_with.__name__)()
            return _Message()

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _Endpoint("messages")
            self.beta = types.SimpleNamespace(messages=_Endpoint("beta"))

    sdk.Anthropic = Anthropic  # type: ignore[attr-defined]
    sdk.calls = calls  # type: ignore[attr-defined]
    return sdk


def test_anthropic_client_routes_betas_to_the_beta_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _fake_sdk()
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    client = AnthropicClient()
    response = client.create(_request(), role="generator")
    assert response.usage.output_tokens == 6
    assert response.latency_ms is not None
    client.create(_request(betas=["x"], fallbacks="default"), role="judge")
    assert sdk.calls == ["messages", "beta"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        ("AuthenticationError", "credentials"),
        ("BadRequestError", "rejected as invalid"),
        ("RateLimitError", "rate limited"),
        ("APIStatusError", "HTTP 500"),
        ("APIConnectionError", "could not reach"),
    ],
)
def test_anthropic_client_maps_errors(
    monkeypatch: pytest.MonkeyPatch, error: str, message: str
) -> None:
    sdk = _fake_sdk(type(error, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    with pytest.raises(LLMError, match=message):
        AnthropicClient().create(_request(), role="generator")
