"""Record and replay model calls, keyed by a hash of the exact request.

A cassette holds every model call for one question through one pipeline.
Replay looks a response up by the SHA-256 of the canonical request JSON. If a
prompt, the retrieved context, the output schema or the model changes, replay
fails loudly with a `CassetteMiss` instead of grading a stale answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from grounded_answers.jsonutil import canonical_json, read_json, sha256_hex, write_json
from grounded_answers.llm import LLMClient, LLMResponse, Role

CASSETTE_FORMAT = "grounded-answers/cassette@1"
Provenance = Literal["hand-crafted", "recorded"]


class CassetteMiss(LookupError):
    """Replay was asked for a request the cassette does not contain."""


def _short(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))[:12]


def fingerprint(request: dict[str, Any]) -> dict[str, Any]:
    """Enough about a request to match it exactly, and to explain a mismatch."""
    messages = request.get("messages", [])
    return {
        "hash": sha256_hex(canonical_json(request)),
        "model": request.get("model"),
        "system_sha": _short(request.get("system")),
        "output_sha": _short(request.get("output_config")),
        "messages_sha": _short(messages),
        "params_sha": _short(
            {k: v for k, v in request.items() if k not in {"system", "messages", "output_config"}}
        ),
    }


@dataclass
class Interaction:
    role: Role
    request: dict[str, Any]  # the fingerprint, not the full request
    response: LLMResponse

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "request": self.request, "response": self.response.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interaction:
        return cls(data["role"], data["request"], LLMResponse.from_dict(data["response"]))


@dataclass
class Cassette:
    pipeline: str
    question_id: str
    question: str
    provenance: Provenance
    note: str
    interactions: list[Interaction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CASSETTE_FORMAT,
            "provenance": self.provenance,
            "note": self.note,
            "pipeline": self.pipeline,
            "question_id": self.question_id,
            "question": self.question,
            "interactions": [i.to_dict() for i in self.interactions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        if data.get("format") != CASSETTE_FORMAT:
            raise CassetteMiss(f"unsupported cassette format {data.get('format')!r}")
        return cls(
            pipeline=data["pipeline"],
            question_id=data["question_id"],
            question=data["question"],
            provenance=data["provenance"],
            note=data["note"],
            interactions=[Interaction.from_dict(i) for i in data["interactions"]],
        )

    @classmethod
    def load(cls, path: Path) -> Cassette:
        if not path.is_file():
            raise CassetteMiss(f"no cassette at {path}. Record one with `make record`.")
        return cls.from_dict(read_json(path))

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())


def cassette_path(cassettes_dir: Path, pipeline: str, question_id: str) -> Path:
    return cassettes_dir / pipeline / f"{question_id}.json"


class ReplayClient:
    """Serves responses from a cassette. Never touches the network."""

    def __init__(self, cassette: Cassette) -> None:
        self.cassette = cassette
        self._unused = list(cassette.interactions)

    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse:
        fp = fingerprint(request)
        for index, interaction in enumerate(self._unused):
            if interaction.role == role and interaction.request["hash"] == fp["hash"]:
                del self._unused[index]
                return interaction.response
        raise CassetteMiss(self._explain(fp, role))

    @property
    def unused(self) -> int:
        return len(self._unused)

    def _explain(self, fp: dict[str, Any], role: Role) -> str:
        label = f"{self.cassette.pipeline}/{self.cassette.question_id}"
        hint = "If the change was intended, rebuild the fixtures (`make craft`) or re-record."
        candidates = [i for i in self._unused if i.role == role]
        if not candidates:
            return f"cassette miss in {label}: no recorded {role} call is left. {hint}"
        old = candidates[0].request
        if old.get("model") != fp["model"]:
            reason = f"the model changed from {old.get('model')} to {fp['model']}"
        elif old.get("system_sha") != fp["system_sha"]:
            reason = "the system prompt changed"
        elif old.get("output_sha") != fp["output_sha"]:
            reason = "the output schema or effort changed"
        elif old.get("messages_sha") != fp["messages_sha"]:
            reason = "the question or the retrieved context changed"
        else:
            reason = "request parameters changed (max_tokens, betas or fallbacks)"
        return f"cassette miss in {label} for a {role} request: {reason}. {hint}"


class RecordingClient:
    """Wraps a real (or scripted) client and keeps every call for the cassette."""

    def __init__(self, inner: LLMClient, cassette: Cassette) -> None:
        self.inner = inner
        self.cassette = cassette

    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse:
        response = self.inner.create(request, role=role)
        self.cassette.interactions.append(Interaction(role, fingerprint(request), response))
        return response
