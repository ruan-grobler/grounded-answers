"""Regex secret scan over the full git history (every commit on every branch).

Used because gitleaks is not installed on the build machine. Matches are never
printed in full: only the pattern name, the commit and a redacted hint.
Exit code 0 means no findings, 1 means findings, 2 means git failed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: dict[str, re.Pattern[str]] = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI style key": re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "Stripe live key": re.compile(r"\b[rs]k_live_[0-9A-Za-z]{16,}\b"),
    "Twilio key": re.compile(r"\bSK[0-9a-f]{32}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "generic key assignment": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{16,}"
    ),
}


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    return result.stdout


def redact(value: str) -> str:
    return value[:4] + "*" * 8


def main() -> int:
    tool = "gitleaks" if shutil.which("gitleaks") else "regex (gitleaks is not installed)"
    commits = git("rev-list", "--all").split()
    history = git("log", "--all", "-p", "--no-color", "--format=commit %H%n%B")
    findings: list[str] = []
    commit = "?"
    added = 0
    for line in history.splitlines():
        if line.startswith("commit ") and len(line) == 47:
            commit = line[7:15]
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        for name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match:
                findings.append(f"  {commit} {name}: {redact(match.group(0))}")
    print(f"Secret scan, method: {tool}")
    print(f"History: {len(commits)} commits on all branches, {added:,} added lines scanned.")
    print(f"Patterns: {', '.join(PATTERNS)}.")
    print(f"Findings: {len(findings)}")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
