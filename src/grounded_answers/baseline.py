"""The regression gate: compare a run against a saved known good baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grounded_answers.evaluate import EvalRun
from grounded_answers.jsonutil import read_json, write_json

BASELINE_FORMAT = "grounded-answers/baseline@1"

# (section, metric, direction): +1 means higher is better, -1 means lower is better.
GATED_METRICS: tuple[tuple[str, str, int], ...] = (
    ("", "end_to_end_correct", 1),
    ("retrieval", "hit@5", 1),
    ("retrieval", "context_precision", 1),
    ("refusals", "refusal_accuracy", 1),
    ("answers", "correct", 1),
    ("safety", "pii_leaks", -1),
    ("safety", "planted_text_reached_model", -1),
    ("safety", "injection_followed", -1),
)


@dataclass
class GateResult:
    passed: bool
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)


def _metric(summary: dict[str, Any], section: str, name: str) -> float:
    value = summary.get(name) if not section else summary.get(section, {}).get(name)
    return float(value or 0.0)


def save_baseline(path: Path, run: EvalRun) -> None:
    write_json(
        path,
        {
            "format": BASELINE_FORMAT,
            "pipeline": run.pipeline,
            "metrics": {
                f"{s}.{m}" if s else m: _metric(run.summary, s, m) for s, m, _ in GATED_METRICS
            },
            "correct": {r.id: r.correct for r in run.results},
        },
    )


def compare(path: Path, run: EvalRun) -> GateResult:
    data = read_json(path)
    if data.get("format") != BASELINE_FORMAT:
        raise ValueError(f"{path} is not a baseline file")
    result = GateResult(passed=True)
    for section, name, direction in GATED_METRICS:
        key = f"{section}.{name}" if section else name
        old = float(data["metrics"].get(key, 0.0))
        new = _metric(run.summary, section, name)
        if (new - old) * direction < -1e-9:
            result.regressions.append(f"{key}: {old:.3f} -> {new:.3f}")
        elif (new - old) * direction > 1e-9:
            result.improvements.append(f"{key}: {old:.3f} -> {new:.3f}")
    for row in run.results:
        was = data["correct"].get(row.id)
        if was and not row.correct:
            result.regressions.append(f"{row.id}: was correct, now wrong")
    result.passed = not result.regressions
    return result
