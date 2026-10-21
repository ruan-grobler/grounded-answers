"""Reports: the text comparison printed by `make demo`, a JSON run file, and an HTML report."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from grounded_answers.evaluate import EvalRun, QuestionResult
from grounded_answers.jsonutil import write_atomic, write_json

CATEGORY_NAMES = {
    "answerable": "answerable",
    "multi_hop": "multi hop",
    "unanswerable": "unanswerable",
    "pii": "personal data",
    "injection": "injection",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


@dataclass(frozen=True)
class Row:
    label: str
    before: str
    after: str
    note: str = ""


def comparison_rows(naive: dict[str, Any], grounded: dict[str, Any]) -> list[Row]:
    def r(section: str, key: str) -> tuple[Any, Any]:
        a = naive.get(key) if not section else naive[section].get(key)
        b = grounded.get(key) if not section else grounded[section].get(key)
        return a, b

    rows = []
    for k in (1, 3, 5, 10):
        a, b = r("retrieval", f"hit@{k}")
        rows.append(Row(f"Retrieval hit rate @{k}", pct(a), pct(b), "all evidence in top k"))
    a, b = r("retrieval", "mrr")
    rows.append(Row("Mean reciprocal rank", f"{a:.2f}", f"{b:.2f}"))
    a, b = r("retrieval", "context_has_all_evidence")
    rows.append(Row("Prompt had all the evidence", pct(a), pct(b)))
    a, b = r("retrieval", "context_precision")
    rows.append(Row("Context precision", pct(a), pct(b), "share of prompt chunks that are gold"))
    a, b = r("retrieval", "mean_context_chunks")
    rows.append(Row("Chunks per prompt", f"{a:.1f}", f"{b:.1f}"))
    a, b = r("citations", "citation_precision")
    rows.append(Row("Citation precision", pct(a), pct(b), "cited chunks that are gold"))
    a, b = r("citations", "quotes_verified")
    rows.append(Row("Quotes verified in source", pct(a), pct(b)))
    a, b = r("answers", "exact_pass")
    rows.append(Row("Answer correct, exact checks", pct(a), pct(b), "35 answerable"))
    a, b = r("answers", "judge_correct")
    rows.append(Row("Answer correct, LLM judge", pct(a), pct(b), "35 answerable"))
    a, b = r("refusals", "refusal_accuracy")
    rows.append(Row("Refusal accuracy", pct(a), pct(b), "right call on all 50"))
    a, b = r("refusals", "must_refuse_refused")
    rows.append(Row("Must refuse, refused", pct(a), pct(b), "15 questions"))
    a, b = r("refusals", "false_refusals")
    rows.append(Row("False refusals", pct(a), pct(b), "of 35 answerable"))
    a, b = r("safety", "pii_leaks")
    rows.append(Row("Personal data leaked", str(a), str(b), "of 7 PII questions"))
    a, b = r("safety", "planted_text_reached_model")
    rows.append(Row("Planted instruction reached the model", str(a), str(b), "questions"))
    a, b = r("safety", "injection_followed")
    rows.append(Row("Injection followed", str(a), str(b), "of 5"))
    a, b = r("", "end_to_end_correct")
    rows.append(Row("End to end correct", pct(a), pct(b), "all 50"))
    a, b = r("cost", "llm_calls")
    rows.append(Row("Model calls", str(a), str(b), "generator plus judge"))
    a, b = r("cost", "per_question_usd")
    rows.append(Row("Cost per question (estimated)", money(a), money(b), "generator plus judge"))
    return rows


def text_table(rows: list[Row], headers: tuple[str, str, str, str]) -> str:
    widths = [
        max(len(headers[0]), *(len(r.label) for r in rows)),
        max(len(headers[1]), *(len(r.before) for r in rows)),
        max(len(headers[2]), *(len(r.after) for r in rows)),
    ]
    lines = [
        f"  {headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:>{widths[2]}}  {headers[3]}",
        f"  {'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * len(headers[3])}",
    ]
    for r in rows:
        lines.append(
            f"  {r.label:<{widths[0]}}  {r.before:>{widths[1]}}  {r.after:>{widths[2]}}  {r.note}"
        )
    return "\n".join(lines)


def outcome(r: QuestionResult) -> str:
    if r.error:
        return "error"
    if r.expect == "refuse":
        verdict = "refused" if r.refused else "answered"
    elif r.refused:
        verdict = "refused"
    else:
        verdict = r.judge_verdict or ("exact pass" if r.exact_pass else "exact fail")
    where = {"query_guard": " at guard", "retrieval_gate": " at gate", "output_guard": " at output"}
    suffix = where.get(r.stage, "")
    if r.forbidden_hits:
        suffix += ", LEAK"
    return verdict.replace("_", " ") + suffix


def save_run(path: Path, run: EvalRun, meta: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "pipeline": run.pipeline,
            "meta": meta,
            "summary": run.summary,
            "results": [r.to_dict() for r in run.results],
            "skipped_for_budget": run.skipped_for_budget,
        },
    )


# --------------------------------------------------------------------- html

_CSS = """
:root { --ink:#16202a; --muted:#5b6774; --line:#e3e8ee; --bg:#f6f8fa; --card:#fff;
  --good:#11734b; --good-bg:#e6f4ec; --bad:#b42318; --bad-bg:#fdecea; --accent:#1f5fbf; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 30px 0 10px; }
.sub { color: var(--muted); margin: 0 0 14px; }
.badges { display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 22px; }
.badge { border:1px solid var(--line); background:var(--card); border-radius:999px; padding:3px 10px;
  font-size:12px; color:var(--muted); }
.badge b { color: var(--ink); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 14px; }
.card .k { color:var(--muted); font-size:12px; text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 19px; font-weight: 650; margin-top: 6px; white-space: nowrap; }
.card .v .from { font-size: 14px; }
.card .v .from { color: var(--muted); font-weight: 500; }
.card .v .arrow { color: var(--muted); font-weight: 400; margin: 0 6px; }
table { width:100%; border-collapse: collapse; background: var(--card); border:1px solid var(--line);
  border-radius: 10px; overflow: hidden; }
th, td { text-align:left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: #fbfcfd; font-size: 12px; color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.better { color: var(--good); font-weight: 600; }
.worse { color: var(--bad); font-weight: 600; }
.pill { display:inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; white-space: nowrap; }
.ok { background: var(--good-bg); color: var(--good); }
.no { background: var(--bad-bg); color: var(--bad); }
.q { max-width: 420px; }
.note { color: var(--muted); font-size: 12px; }
.honesty { background: #fff8e6; border: 1px solid #f1dfa6; border-radius: 10px; padding: 12px 14px;
  margin-top: 18px; font-size: 13px; }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
"""


def _card(label: str, before: str, after: str) -> str:
    return (
        f'<div class="card"><div class="k">{escape(label)}</div><div class="v">'
        f'<span class="from">{escape(before)}</span><span class="arrow">&rarr;</span>'
        f"{escape(after)}</div></div>"
    )


def _num(text: str) -> float | None:
    try:
        return float(text.rstrip("%").lstrip("$").split()[0])
    except (ValueError, IndexError):
        return None


LOWER_IS_BETTER = (
    "False refusals",
    "Personal data leaked",
    "Planted instruction reached the model",
    "Injection followed",
    "Cost per question (estimated)",
    "Model calls",
    "Chunks per prompt",
)


def _delta_class(row: Row) -> str:
    a, b = _num(row.before), _num(row.after)
    if a is None or b is None or a == b:
        return ""
    better = b < a if row.label in LOWER_IS_BETTER else b > a
    return "better" if better else "worse"


def html_report(
    naive: EvalRun,
    grounded: EvalRun,
    ablation: list[tuple[str, dict[str, float | None]]],
    meta: dict[str, str],
) -> str:
    ns, gs = naive.summary, grounded.summary
    cards = "".join(
        [
            _card(
                "End to end correct", pct(ns["end_to_end_correct"]), pct(gs["end_to_end_correct"])
            ),
            _card("Hit rate @5", pct(ns["retrieval"]["hit@5"]), pct(gs["retrieval"]["hit@5"])),
            _card(
                "Refusal accuracy",
                pct(ns["refusals"]["refusal_accuracy"]),
                pct(gs["refusals"]["refusal_accuracy"]),
            ),
            _card(
                "Personal data leaked",
                str(ns["safety"]["pii_leaks"]),
                str(gs["safety"]["pii_leaks"]),
            ),
            _card(
                "Injection reached model",
                str(ns["safety"]["planted_text_reached_model"]),
                str(gs["safety"]["planted_text_reached_model"]),
            ),
            _card(
                "Cost per question",
                money(ns["cost"]["per_question_usd"]),
                money(gs["cost"]["per_question_usd"]),
            ),
        ]
    )
    rows = "".join(
        f"<tr><td>{escape(r.label)}</td><td class='num'>{escape(r.before)}</td>"
        f"<td class='num {_delta_class(r)}'>{escape(r.after)}</td>"
        f"<td class='note'>{escape(r.note)}</td></tr>"
        for r in comparison_rows(ns, gs)
    )
    ablation_rows = "".join(
        f"<tr><td>{escape(name)}</td>"
        + "".join(
            f"<td class='num'>{pct(m[k]) if k != 'mrr' else f'{m[k]:.2f}'}</td>"
            for k in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr")
        )
        + "</tr>"
        for name, m in ablation
    )
    by_cat = "".join(
        f"<tr><td>{escape(CATEGORY_NAMES.get(cat, cat))}</td><td class='num'>{v['questions']}</td>"
        f"<td class='num'>{ns['by_category'][cat]['correct']}</td>"
        f"<td class='num'>{v['correct']}</td></tr>"
        for cat, v in gs["by_category"].items()
    )
    naive_by_id = {r.id: r for r in naive.results}
    question_rows = []
    for g in grounded.results:
        n = naive_by_id[g.id]
        ranks = ", ".join("-" if x is None else str(x) for x in g.hop_ranks) or "n/a"
        question_rows.append(
            f"<tr><td>{escape(g.id)}</td><td>{escape(CATEGORY_NAMES.get(g.category, g.category))}</td>"
            f"<td class='q'>{escape(g.question)}</td>"
            f"<td><span class='pill {'ok' if n.correct else 'no'}'>{escape(outcome(n))}</span></td>"
            f"<td><span class='pill {'ok' if g.correct else 'no'}'>{escape(outcome(g))}</span></td>"
            f"<td class='num'>{escape(ranks)}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grounded Answers eval report</title><style>{_CSS}</style></head>
<body><main>
<h1>Grounded Answers: eval report</h1>
<p class="sub">Seapoint Smile Dental (a fictional practice). {gs["questions"]} test questions. Before is naive dense-only RAG, After is hybrid retrieval plus rerank with guardrails.</p>
<div class="badges">
<span class="badge">Generated <b>{escape(meta.get("generated", ""))}</b></span>
<span class="badge">Retrieval <b>{escape(meta.get("retrieval", ""))}</b></span>
<span class="badge">Generator and judge <b>{escape(meta.get("llm", ""))}</b></span>
<span class="badge">Corpus <b>{escape(meta.get("corpus", ""))}</b></span>
</div>
<div class="cards">{cards}</div>
<div class="grid2">
<div><h2>Before and After</h2>
<table><tr><th>Metric</th><th class="num">Before</th><th class="num">After</th><th>Note</th></tr>{rows}</table></div>
<div><h2>Retrieval ablation (35 questions with evidence)</h2>
<table><tr><th>Method</th><th class="num">@1</th><th class="num">@3</th><th class="num">@5</th><th class="num">@10</th><th class="num">MRR</th></tr>{ablation_rows}</table>
<p class="note">Hit rate counts a question only when every piece of evidence it needs is in the top k. No guardrail filtering in this table.</p>
<h2>Correct by category</h2>
<table><tr><th>Category</th><th class="num">Questions</th><th class="num">Before</th><th class="num">After</th></tr>{by_cat}</table>
<div class="honesty"><b>What is real here.</b> {escape(meta.get("honesty", ""))}</div>
</div></div>
<h2>Every question</h2>
<table><tr><th>Id</th><th>Category</th><th>Question</th><th>Before</th><th>After</th><th class="num">Evidence rank</th></tr>{"".join(question_rows)}</table>
</main></body></html>
"""


def write_html(path: Path, html: str) -> None:
    write_atomic(path, html.encode("utf-8"))


def today() -> str:
    return dt.date.today().isoformat()
