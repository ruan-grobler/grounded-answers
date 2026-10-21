from __future__ import annotations

from pathlib import Path

import pytest

from grounded_answers.cli import main
from tests.helpers import ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def no_local_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the model folder somewhere empty, so `auto` replays like CI does."""
    monkeypatch.setenv("GA_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["--root", str(ROOT), *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def test_check(capsys) -> None:
    code, out = run(capsys, "check")
    assert code == 0
    assert "consistent" in out


def test_craft_check(capsys) -> None:
    code, out = run(capsys, "craft", "--check")
    assert code == 0
    assert "Cassettes match" in out


def test_eval_with_gate(capsys, tmp_path: Path) -> None:
    code, out = run(
        capsys,
        "eval",
        "--pipeline",
        "grounded",
        "--baseline",
        "baselines/grounded.json",
        "--out",
        str(tmp_path),
    )
    assert code == 0, out
    assert "Regression gate: PASS" in out
    assert (tmp_path / "grounded-test.json").is_file()


def test_replay_fails_loudly_without_cassettes(capsys, tmp_path: Path) -> None:
    code, out = run(
        capsys,
        "eval",
        "--pipeline",
        "naive",
        "--split",
        "dev",
        "--out",
        str(tmp_path),
        "--save-baseline",
        str(tmp_path / "b.json"),
    )
    assert code == 2  # the dev split has no cassettes, so the first model call is a miss
    assert "cassette miss" in out
    assert not (tmp_path / "b.json").exists()


def test_search_and_ask(capsys) -> None:
    question = "What is the after-hours call-out fee?"
    code, out = run(capsys, "search", question, "-k", "3")
    assert code == 0
    assert "after-hours-emergencies" in out or "fee-schedule-2026" in out
    code, out = run(capsys, "ask", question)
    assert code == 0
    assert "R950" in out
    assert "source:" in out
    code, out = run(capsys, "ask", "What is Dr Naidoo's personal cell number?")
    assert code == 0
    assert "stage: query_guard" in out


def test_ask_unknown_question_in_replay(capsys) -> None:
    code, out = run(capsys, "ask", "What is the meaning of life?")
    assert code == 2
    assert "Replay mode only knows the eval questions" in out


def test_live_modes_need_a_key(capsys) -> None:
    code, out = run(capsys, "ask", "What is the late fee?", "--mode", "live")
    assert code == 2
    assert "ANTHROPIC_API_KEY" in out


def test_tune_without_writing(capsys) -> None:
    code, out = run(capsys, "tune")
    assert code == 0
    assert "gate_min_score: -5.5" in out


def test_index_replay(capsys) -> None:
    code, out = run(capsys, "index", "--models", "replay")
    assert code == 0
    assert "0 new from the live models" in out


def test_bench_needs_models(capsys) -> None:
    code, out = run(capsys, "bench")
    assert code == 2
    assert "make models" in out


def test_demo(capsys, tmp_path: Path) -> None:
    code, out = run(capsys, "demo", "--out", str(tmp_path))
    assert code == 0, out
    assert "Regression gate against baselines/grounded.json: PASS" in out
    assert "Skipped: local models are not downloaded" in out
    assert (tmp_path / "report.html").is_file()
