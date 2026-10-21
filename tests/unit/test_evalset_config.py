from __future__ import annotations

from pathlib import Path

import pytest

from grounded_answers.config import ConfigError, ModelSettings, Paths, find_root, load_pipelines
from grounded_answers.evalset import EvalSetError, check_evidence, load_questions, normalise
from tests.helpers import ROOT, chunk


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "q.yaml"
    path.write_text(body, encoding="utf-8")
    return path


GOOD = """
questions:
  - id: q1
    category: multi_hop
    question: What is the fee?
    expect: answer
    reference: R350.
    evidence:
      - {doc: fees, section: Cancellations, fact: "R350"}
      - any_of:
          - {doc: fees, section: Other, fact: "x"}
          - {doc: prices, section: List, fact: "R650"}
    must_include: ["R350", ["24 hours", "a day"]]
    must_not_include: ["R500"]
  - id: q2
    category: pii
    question: Whose cell?
    expect: refuse
    refusal_reason: personal_data
    reference: Refuse.
"""


def test_load_questions(tmp_path: Path) -> None:
    q1, q2 = load_questions(_write(tmp_path, GOOD))
    assert q1.must_include == (("R350",), ("24 hours", "a day"))
    assert len(q1.evidence[1].locations) == 2
    assert q1.is_gold(chunk("prices#00", "R650", section="List"))
    assert not q1.is_gold(chunk("prices#00", "R650", section="Other"))
    assert q2.refusal_reason == "personal_data"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("questions: []", "no questions"),
        ("questions:\n  - {id: x}", "missing category"),
        (
            "questions:\n  - {id: x, category: odd, question: q, expect: answer, reference: r}",
            "unknown category",
        ),
        (
            "questions:\n  - {id: x, category: pii, question: q, expect: maybe, reference: r}",
            "expect must be",
        ),
        (
            "questions:\n  - {id: x, category: pii, question: q, expect: answer, reference: r}",
            "needs evidence",
        ),
        (
            "questions:\n  - {id: x, category: pii, question: q, expect: refuse, reference: r}",
            "refusal_reason",
        ),
        (
            "questions:\n  - {id: x, category: pii, question: q, expect: refuse, reference: r, "
            "refusal_reason: personal_data, evidence: [{doc: a, section: b, fact: c}]}",
            "should not list evidence",
        ),
        (
            "questions:\n  - {id: x, category: answerable, question: q, expect: answer, reference: r, "
            "evidence: [{doc: a}]}",
            "doc, section and fact",
        ),
        (
            "questions:\n  - {id: x, category: answerable, question: q, expect: answer, reference: r, "
            "evidence: [{any_of: []}]}",
            "at least one",
        ),
    ],
)
def test_load_errors(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(EvalSetError, match=message):
        load_questions(_write(tmp_path, body))


def test_duplicates(tmp_path: Path) -> None:
    one = "  - {id: x, category: pii, question: q, expect: refuse, refusal_reason: personal_data, reference: r}\n"
    with pytest.raises(EvalSetError, match="duplicate question ids"):
        load_questions(_write(tmp_path, "questions:\n" + one + one))
    two = one.replace("id: x", "id: y")
    with pytest.raises(EvalSetError, match="duplicate question texts"):
        load_questions(_write(tmp_path, "questions:\n" + one + two))


def test_normalise() -> None:
    assert normalise("R4,900  and" + chr(0x2019) + "s   R1,150,000") == "r4900 and's r1150000"
    assert normalise("Café “quoted”") == 'café "quoted"'


def test_check_evidence(tmp_path: Path) -> None:
    questions = load_questions(_write(tmp_path, GOOD))
    chunks = [chunk("fees#00", "Charged R350.", section="Cancellations")]
    problems = check_evidence(questions, chunks)
    assert "no section fees > Other" in problems[0]
    chunks = [chunk("fees#00", "Charged R999.", section="Cancellations")]
    assert any("not in Cancellations" in p for p in check_evidence(questions, chunks))


def test_paths_and_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert find_root(ROOT / "src") == ROOT
    with pytest.raises(ConfigError):
        find_root(tmp_path)
    paths = Paths(ROOT)
    assert paths.questions.name == "questions.yaml"
    assert paths.dev_questions.name == "dev.yaml"
    assert paths.index_db.parent.name == ".cache"
    assert paths.scripts.name == "answers.yaml"
    assert paths.reports.name == "reports"
    monkeypatch.setenv("GA_MODELS_DIR", str(tmp_path))
    assert paths.models_dir == tmp_path


def test_load_pipelines() -> None:
    configs = load_pipelines(ROOT / "config" / "pipelines.yaml")
    assert configs["naive"].retrieval.mode == "dense"
    assert configs["naive"].retrieval.blocked_flags == frozenset()
    grounded = configs["grounded"]
    assert grounded.retrieval.mode == "hybrid_rerank"
    assert grounded.guardrails is True
    assert grounded.retrieval.gate_min_score is not None
    assert "personal_data" in grounded.retrieval.blocked_flags


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("naive: {retrieval: dense}", "missing pipelines"),
        ("naive: {retrieval: magic}\ngrounded: {retrieval: dense}", "retrieval must be"),
        ("naive: {retrieval: dense, prompt: odd}\ngrounded: {retrieval: dense}", "prompt must be"),
    ],
)
def test_bad_pipelines(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_pipelines(_write(tmp_path, body))


def test_model_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GA_FALLBACKS", "off")
    monkeypatch.setenv("GA_BUDGET_USD", "0.50")
    monkeypatch.setenv("GA_JUDGE_MODEL", "claude-haiku-5-5")
    settings = ModelSettings.from_env()
    assert settings.fallbacks is False
    assert settings.budget_usd == 0.5
    assert settings.judge_model == "claude-haiku-5-5"
    monkeypatch.setenv("GA_FALLBACKS", "maybe")
    with pytest.raises(ConfigError, match="GA_FALLBACKS"):
        ModelSettings.from_env()
    monkeypatch.setenv("GA_FALLBACKS", "default")
    monkeypatch.setenv("GA_BUDGET_USD", "lots")
    with pytest.raises(ConfigError, match="GA_BUDGET_USD"):
        ModelSettings.from_env()
