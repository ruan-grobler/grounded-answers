"""Hand-crafted fixtures: a scripted stand-in for Claude, used to build the committed cassettes.

No Claude API key was available while this repo was built, so the generator and
judge responses in fixtures/cassettes/ are not recorded from the API. They are
built by `ga craft`, which runs the real pipelines with this scripted client in
place of Claude and saves every request and response.

The scripted generator follows one written rule, the same for both pipelines:

1. It answers only from the documents in its prompt. For each question there
   is a hand-written answer in fixtures/handcrafted/answers.yaml, split into
   claims, and each claim names the document it comes from and a quote.
2. A claim is used only if a document from that source, containing that quote,
   is in the prompt. All claims found: a full answer. Some: a partial answer
   that says the rest is missing. None: it refuses ("not in the documents").
3. Under the grounded prompt, which forbids personal data, it refuses personal
   data questions. The naive prompt has no such rule, so if the personal data
   is in its context it answers, as a plain instruction following model would.
4. It never follows instructions planted inside documents, under either prompt.

The scripted judge marks an answer correct when it contains every scripted
claim for the question, partially correct when it contains some.

So in this fixture, differences between the pipelines come from retrieval and
the guardrails only. Token counts are estimates (characters / 4). Recording
with a real key (`make record`) replaces all of this with real responses.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

import yaml

from grounded_answers.cassette import Cassette, RecordingClient, cassette_path
from grounded_answers.config import ModelSettings
from grounded_answers.evalset import Question, normalise
from grounded_answers.judge import judge_answer
from grounded_answers.llm import LLMError, LLMResponse, Role, Usage
from grounded_answers.pipeline import Pipeline
from grounded_answers.prompts import GROUNDED_SYSTEM

NOTE = (
    "Hand-crafted fixture, not recorded from the Claude API. Built by `ga craft` from "
    "fixtures/handcrafted/answers.yaml with the rule in src/grounded_answers/crafting.py. "
    "Token counts are estimates (characters / 4)."
)
PARTIAL_SUFFIX = "I could not find the rest of this in the documents provided."
NAIVE_REFUSAL = "I don't know. The documents provided do not cover this."
GROUNDED_REFUSAL = "The documents provided do not cover this, so I can't answer it."
PERSONAL_REFUSAL = "I can't share personal information about staff or patients."

_GROUNDED_DOC = re.compile(
    r'<document id="(S\d+)" source="([^"]*)" title="[^"]*" section="[^"]*"(?: page="\d+")?>\n'
    r"(.*?)\n</document>",
    re.DOTALL,
)
_NAIVE_DOC = re.compile(r"\[(S\d+)\] (.+?) > .+?\n(.*?)(?=\n\n\[S\d+\] |\n\nQuestion: )", re.DOTALL)
_TAG = re.compile(r"<(question|candidate_answer)>\n(.*?)\n</\1>", re.DOTALL)


class ScriptError(ValueError):
    """fixtures/handcrafted/answers.yaml is invalid."""


@dataclass(frozen=True)
class ScriptedClaim:
    """One claim of a hand-written answer.

    `sources` lists (doc, quote) pairs, any one of which supports the claim.
    `partial` is the wording to use when this claim is found but others are
    not, so a partial answer never states a total it could not have worked out.
    """

    statement: str
    sources: tuple[tuple[str, str], ...]
    partial: str | None = None


def _claim(qid: str, raw: Any) -> ScriptedClaim:
    if not isinstance(raw, dict) or "statement" not in raw:
        raise ScriptError(f"{qid}: each claim needs a statement")
    if "sources" in raw:
        sources = tuple((str(s["doc"]), str(s["quote"])) for s in raw["sources"])
    elif {"doc", "quote"} <= raw.keys():
        sources = ((str(raw["doc"]), str(raw["quote"])),)
    else:
        raise ScriptError(f"{qid}: each claim needs doc and quote, or sources")
    if not sources:
        raise ScriptError(f"{qid}: a claim needs at least one source")
    partial = raw.get("partial")
    return ScriptedClaim(str(raw["statement"]), sources, None if partial is None else str(partial))


def load_scripts(path: Path) -> dict[str, list[ScriptedClaim]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scripts: dict[str, list[ScriptedClaim]] = {}
    for qid, claims in (raw.get("answers") or {}).items():
        if not isinstance(claims, list) or not claims:
            raise ScriptError(f"{qid}: expected a non-empty list of claims")
        scripts[str(qid)] = [_claim(str(qid), c) for c in claims]
    return scripts


def _estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _message_text(request: dict[str, Any]) -> str:
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else json.dumps(content)


class ScriptedClient:
    """Plays Claude by the rule in the module docstring. Deterministic and offline."""

    def __init__(
        self,
        questions: list[Question],
        scripts: dict[str, list[ScriptedClaim]],
        titles: dict[str, str],
    ) -> None:
        self.by_text = {q.question: q for q in questions}
        self.scripts = scripts
        self.titles = titles  # document title -> document id, for the naive prompt format

    def create(self, request: dict[str, Any], *, role: Role) -> LLMResponse:
        body = self._judge(request) if role == "judge" else self._generate(request)
        text = json.dumps(body, ensure_ascii=False)
        prompt = (
            str(request.get("system", ""))
            + _message_text(request)
            + json.dumps(request.get("output_config", {}))
        )
        return LLMResponse(
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=_estimate(prompt), output_tokens=_estimate(text)),
            model=str(request["model"]),
        )

    def _question(self, text: str) -> Question:
        question = self.by_text.get(text.strip())
        if question is None:
            raise LLMError(f"the scripted client has no question {text[:60]!r}")
        return question

    def _documents(self, request: dict[str, Any]) -> tuple[bool, str, list[tuple[str, str, str]]]:
        message = _message_text(request)
        grounded = request.get("system") == GROUNDED_SYSTEM
        if grounded:
            tags = dict(_TAG.findall(message))
            question = unescape(tags.get("question", ""))
            docs = [(s, d, unescape(t)) for s, d, t in _GROUNDED_DOC.findall(message)]
        else:
            question = message.rsplit("\n\nQuestion: ", 1)[-1]
            docs = [
                (s, self.titles.get(title, ""), t) for s, title, t in _NAIVE_DOC.findall(message)
            ]
        return grounded, question, docs

    def _generate(self, request: dict[str, Any]) -> dict[str, Any]:
        grounded, text, docs = self._documents(request)
        question = self._question(text)
        if grounded and question.category == "pii":
            return _refusal(PERSONAL_REFUSAL, "personal_data")
        found: list[tuple[ScriptedClaim, str, str]] = []
        for claim in self.scripts.get(question.id, []):
            match = next(
                (
                    (label, quote)
                    for doc, quote in claim.sources
                    for label, doc_id, body in docs
                    if doc_id == doc and normalise(quote) in normalise(body)
                ),
                None,
            )
            if match is not None:
                found.append((claim, match[0], match[1]))
        if not found:
            return _refusal(GROUNDED_REFUSAL if grounded else NAIVE_REFUSAL, "not_in_documents")
        complete = len(found) == len(self.scripts[question.id])

        def wording(claim: ScriptedClaim) -> str:
            return claim.statement if complete or claim.partial is None else claim.partial

        answer = " ".join(f"{wording(c)} [{label}]" for c, label, _ in found)
        if not complete:
            answer += " " + PARTIAL_SUFFIX
        return {
            "answer": answer,
            "claims": [
                {"statement": wording(c), "source_ids": [label], "quote": quote}
                for c, label, quote in found
            ],
            "confidence": ("high" if len(found) == 1 else "medium") if complete else "low",
            "refused": False,
            "refusal_reason": "none",
        }

    def _judge(self, request: dict[str, Any]) -> dict[str, Any]:
        tags = dict(_TAG.findall(_message_text(request)))
        question = self._question(unescape(tags.get("question", "")))
        candidate = normalise(unescape(tags.get("candidate_answer", "")))
        claims = self.scripts.get(question.id, [])
        full = [c for c in claims if normalise(c.statement) in candidate]
        some = [
            c
            for c in claims
            if normalise(c.statement) in candidate
            or (c.partial is not None and normalise(c.partial) in candidate)
        ]
        if claims and len(full) == len(claims):
            return {"verdict": "correct", "reason": "Every key fact in the reference is present."}
        if some:
            return {
                "verdict": "partially_correct",
                "reason": f"{len(full)} of {len(claims)} key facts are fully present.",
            }
        return {"verdict": "incorrect", "reason": "The key facts in the reference are missing."}


def _refusal(text: str, reason: str) -> dict[str, Any]:
    return {
        "answer": text,
        "claims": [],
        "confidence": "low",
        "refused": True,
        "refusal_reason": reason,
    }


def craft_cassettes(
    pipeline: Pipeline,
    questions: list[Question],
    client: ScriptedClient,
    settings: ModelSettings,
) -> dict[str, Cassette]:
    """Run the real pipeline with the scripted client. One cassette per question with calls."""
    cassettes: dict[str, Cassette] = {}
    for question in questions:
        cassette = Cassette(pipeline.name, question.id, question.question, "hand-crafted", NOTE)
        recorder = RecordingClient(client, cassette)
        result = pipeline.answer(question.question, recorder)
        if question.expect == "answer" and not result.refused:
            judge_answer(
                recorder, settings, question.question, question.reference, result.final_text
            )
        if cassette.interactions:
            cassettes[question.id] = cassette
    return cassettes


def write_cassettes(
    cassettes_dir: Path, pipeline: str, cassettes: dict[str, Cassette], questions: list[Question]
) -> list[str]:
    """Write new cassettes, remove ones for questions that no longer call a model."""
    changed = []
    for question in questions:
        path = cassette_path(cassettes_dir, pipeline, question.id)
        cassette = cassettes.get(question.id)
        if cassette is None:
            if path.exists():
                path.unlink()
                changed.append(f"removed {path.name}")
            continue
        new = json.dumps(cassette.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != new:
            cassette.save(path)
            changed.append(f"wrote {path.name}")
    return changed


def diff_cassettes(
    cassettes_dir: Path, pipeline: str, cassettes: dict[str, Cassette], questions: list[Question]
) -> list[str]:
    """What `write_cassettes` would change. Empty means the committed fixtures are current."""
    problems = []
    for question in questions:
        path = cassette_path(cassettes_dir, pipeline, question.id)
        cassette = cassettes.get(question.id)
        if cassette is None:
            if path.exists():
                problems.append(f"{pipeline}/{path.name} should not exist")
            continue
        if not path.exists():
            problems.append(f"{pipeline}/{path.name} is missing")
            continue
        new = json.dumps(cassette.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path.read_text(encoding="utf-8") != new:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            if on_disk.get("provenance") == "recorded":
                continue  # a real recording replaces the fixture on purpose
            problems.append(f"{pipeline}/{path.name} differs from what `ga craft` builds")
    return problems
