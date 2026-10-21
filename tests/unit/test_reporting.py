from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grounded_answers.baseline import compare, save_baseline
from grounded_answers.evaluate import EvalRun, QuestionResult, summarise
from grounded_answers.jsonutil import read_json
from grounded_answers.llm import Usage
from grounded_answers.pricing import cost_usd
from grounded_answers.report import (
    Row,
    comparison_rows,
    html_report,
    money,
    ms,
    outcome,
    pct,
    save_run,
    text_table,
    write_html,
)


def _result(qid: str, *, correct: bool = True, expect: str = "answer", **kw: Any) -> QuestionResult:
    base: dict[str, Any] = dict(  # noqa: C408
        id=qid,
        category="answerable" if expect == "answer" else "pii",
        expect=expect,
        question=f"Question {qid}?",
        final_text="R350 [S1]",
        refused=expect == "refuse",
        refusal_reason="none",
        stage="model",
        guard_rule=None,
        hop_ranks=[1] if expect == "answer" else [],
        ranked_ids=["a#00"],
        context_ids=["a#00"],
        context_gold=1 if expect == "answer" else 0,
        top_score=3.0,
        cited_ids=["a#00"],
        cited_gold=1,
        claims=1,
        verified_claims=1,
        output_blocked=False,
        exact_pass=True if expect == "answer" else None,
        exact_failures=[],
        forbidden_hits=[],
        judge_verdict="correct" if expect == "answer" else None,
        judge_reason="",
        correct=correct,
        planted_in_context=False,
        personal_in_context=False,
        llm_calls=2,
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.001,
        retrieval_ms=5.0,
        generation_ms=None,
    )
    base.update(kw)
    return QuestionResult(**base)


def _run(name: str, results: list[QuestionResult]) -> EvalRun:
    return EvalRun(name, results, summarise(results))


def test_formatters() -> None:
    assert pct(0.875) == "88%"
    assert pct(None) == "n/a"
    assert money(0.0052) == "$0.0052"
    assert ms(12.4) == "12 ms"
    assert ms(None) == "n/a"


def test_outcome_labels() -> None:
    assert outcome(_result("a")) == "correct"
    assert outcome(_result("p", expect="refuse", stage="query_guard")) == "refused at guard"
    assert outcome(_result("b", refused=True, stage="retrieval_gate")) == "refused at gate"
    assert (
        outcome(_result("c", forbidden_hits=["x"], expect="refuse", refused=False))
        == "answered, LEAK"
    )
    assert outcome(_result("d", error="boom")) == "error"
    assert outcome(_result("e", judge_verdict=None, exact_pass=False)) == "exact fail"


def test_comparison_and_text_table() -> None:
    naive = _run("naive", [_result("a01"), _result("p01", expect="refuse")])
    grounded = _run("grounded", [_result("a01"), _result("p01", expect="refuse")])
    rows = comparison_rows(naive.summary, grounded.summary)
    labels = [r.label for r in rows]
    assert "Retrieval hit rate @5" in labels
    assert "Refusal accuracy" in labels
    table = text_table([Row("A metric", "1%", "2%", "note")], ("Metric", "Before", "After", "Note"))
    assert table.splitlines()[2] == "  A metric      1%     2%  note"


def test_html_report(tmp_path: Path) -> None:
    naive = _run("naive", [_result("a01", correct=False), _result("p01", expect="refuse")])
    grounded = _run("grounded", [_result("a01"), _result("p01", expect="refuse")])
    ablation: list[tuple[str, dict[str, float | None]]] = [
        ("Dense", {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0})
    ]
    html = html_report(
        naive, grounded, ablation, {"generated": "2026-10-21", "honesty": "<b>x</b>"}
    )
    assert html.startswith("<!doctype html>")
    assert "Before and After" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert chr(0x2014) not in html and chr(0x2013) not in html
    write_html(tmp_path / "r.html", html)
    assert (tmp_path / "r.html").read_text(encoding="utf-8") == html
    save_run(tmp_path / "run.json", grounded, {"mode": "replay"})
    assert read_json(tmp_path / "run.json")["summary"]["questions"] == 2


def test_baseline_gate(tmp_path: Path) -> None:
    good = _run("grounded", [_result("a01"), _result("p01", expect="refuse")])
    path = tmp_path / "baseline.json"
    save_baseline(path, good)
    assert compare(path, good).passed
    worse = _run(
        "grounded",
        [_result("a01", correct=False, exact_pass=False), _result("p01", expect="refuse")],
    )
    gate = compare(path, worse)
    assert not gate.passed
    assert "a01: was correct, now wrong" in gate.regressions
    assert any(r.startswith("end_to_end_correct") for r in gate.regressions)
    better_path = tmp_path / "worse.json"
    save_baseline(better_path, worse)
    assert compare(better_path, good).improvements
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a baseline"):
        compare(tmp_path / "bad.json", good)


def test_pricing() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4 + 20 + 0.2 + 5)
    assert cost_usd("unknown-model", usage) is None
