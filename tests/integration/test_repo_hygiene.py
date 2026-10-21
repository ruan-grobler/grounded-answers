"""Rules from the project spec that are cheap to check on every run."""

from __future__ import annotations

import re
import subprocess

import pytest

from tests.helpers import ROOT

pytestmark = pytest.mark.integration

TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".json", ".example", ""}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def text_files() -> list[str]:
    return [
        f
        for f in tracked_files()
        if (ROOT / f).suffix in TEXT_SUFFIXES and (ROOT / f).is_file() and "cassettes" not in f
    ]


DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")


def test_no_em_or_en_dashes() -> None:
    offenders = [f for f in text_files() if DASHES.search((ROOT / f).read_text(encoding="utf-8"))]
    assert offenders == []


def test_every_corpus_document_says_it_is_fictional() -> None:
    for path in sorted((ROOT / "corpus").glob("*.md")):
        assert "Fictional document." in path.read_text(encoding="utf-8"), path.name
    for path in sorted((ROOT / "scripts" / "pdf_sources").glob("*.txt")):
        assert "banner: Fictional document." in path.read_text(encoding="utf-8"), path.name


def test_only_example_emails_and_fake_numbers_in_the_data() -> None:
    folders = ("corpus", "eval", "fixtures/handcrafted", "scripts/pdf_sources")
    for folder in folders:
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in {".md", ".yaml", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            for email in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text):
                assert email.endswith(".example.com"), (path.name, email)
            for phone in re.findall(r"\+27[\d ]{9,13}\d", text):
                assert phone.startswith("+27 00 000 0"), (path.name, phone)


def test_env_example_has_no_values() -> None:
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY"):
            assert line.strip() == "ANTHROPIC_API_KEY="


def test_no_key_like_strings_are_tracked() -> None:
    pattern = re.compile(
        r"sk-ant-[A-Za-z0-9_\-]{20,}|\bgh[pousr]_[A-Za-z0-9]{36,}|\bAKIA[0-9A-Z]{16}\b"
    )
    for f in text_files():
        assert not pattern.search((ROOT / f).read_text(encoding="utf-8")), f
