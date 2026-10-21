"""`make demo`: replay the 50 questions through both pipelines and print the Before and After."""

from __future__ import annotations

import collections
import statistics
from collections.abc import Callable
from pathlib import Path

from grounded_answers.baseline import compare
from grounded_answers.bench import bench
from grounded_answers.evalset import Question, load_questions
from grounded_answers.evaluate import K_VALUES, EvalRun, run_eval
from grounded_answers.inference import EMBED_MODEL, RERANK_MODEL
from grounded_answers.report import (
    CATEGORY_NAMES,
    Row,
    comparison_rows,
    html_report,
    ms,
    outcome,
    pct,
    save_run,
    text_table,
    today,
    write_html,
)
from grounded_answers.retrieval import MODES
from grounded_answers.runtime import Runtime, client_factory, open_runtime

Echo = Callable[[str], None]

MODE_NAMES = {
    "dense": "Dense only (BGE small, cosine)",
    "bm25": "BM25 only (SQLite FTS5)",
    "hybrid": "Hybrid (BM25 + dense, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}

HONESTY = (
    "Retrieval, reranking, the guardrails and every metric are real code, and the embedding "
    "and rerank scores are real outputs of the pinned local models. The generator and judge "
    "responses are a hand-crafted fixture, not recorded from the Claude API: a scripted "
    "stand-in answers only when the evidence is in its prompt. Costs are estimates from "
    "prompt sizes. All data is synthetic."
)


def retrieval_ablation(
    rt: Runtime, questions: list[Question]
) -> list[tuple[str, dict[str, float | None]]]:
    rows = []
    evidence = [q for q in questions if q.evidence]
    for mode in MODES:
        retriever = rt.ablation_retriever(mode)
        hits = collections.Counter[int]()
        reciprocal = []
        for question in evidence:
            ranked = retriever.retrieve(question.question).ranked
            ranks = [
                next((i for i, c in enumerate(ranked, 1) if hop.matches(c.chunk)), None)
                for hop in question.evidence
            ]
            for k in K_VALUES:
                hits[k] += all(r is not None and r <= k for r in ranks)
            found = [r for r in ranks if r is not None]
            reciprocal.append(1 / min(found) if found else 0.0)
        metrics: dict[str, float | None] = {f"hit@{k}": hits[k] / len(evidence) for k in K_VALUES}
        metrics["mrr"] = statistics.fmean(reciprocal)
        rows.append((MODE_NAMES[mode], metrics))
    return rows


def _show(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_demo(root: Path, out: Path, echo: Echo) -> int:
    rt = open_runtime(root, model_mode="auto")
    paths = rt.paths
    questions = load_questions(paths.questions)
    report = rt.index_report
    docs = collections.Counter(c.doc_id for c in rt.store.chunks)
    pdfs = sum(1 for p in paths.corpus.glob("*.pdf"))
    categories = collections.Counter(q.category for q in questions)

    echo("Grounded Answers demo")
    echo("=" * 72)
    echo("Fictional business: Seapoint Smile Dental, Cape Town. All data is synthetic.")
    echo(
        f"Corpus: {len(docs)} documents ({len(docs) - pdfs} Markdown, {pdfs} PDF), "
        f"{report.chunks} chunks."
    )
    echo(f"Embeddings: {EMBED_MODEL}. Reranker: {RERANK_MODEL}. Vector store: sqlite-vec.")
    echo(
        "Local models: "
        + (
            "live (downloaded), cache first"
            if rt.live_models
            else "replayed from the committed cache in index/"
        )
    )
    echo("Generator and judge: replayed from fixtures/cassettes/ (hand-crafted, see the README).")
    echo("Flagged at ingest, never retrieved by the grounded pipeline:")
    for chunk_id, flags in report.flagged.items():
        echo(f"  {chunk_id:<36} {', '.join(flags)}")
    echo(
        "Eval set: "
        + ", ".join(f"{n} {c.replace('_', ' ')}" for c, n in categories.items())
        + f" ({len(questions)} questions)."
    )

    echo("")
    echo("Retrieval ablation, 35 questions with evidence (all evidence must be in the top k)")
    echo("-" * 72)
    ablation = retrieval_ablation(rt, questions)
    ab_rows = [
        Row(
            name,
            " ".join(f"{pct(m[f'hit@{k}']):>5}" for k in K_VALUES),
            f"{m['mrr']:.2f}" if m["mrr"] is not None else "n/a",
        )
        for name, m in ablation
    ]
    echo(text_table(ab_rows, ("Method", "  @1    @3    @5   @10", "MRR", "")))

    runs: dict[str, EvalRun] = {}
    for name in ("naive", "grounded"):
        runs[name] = run_eval(
            rt.pipelines[name], questions, client_factory(paths, name, "replay"), rt.settings
        )
    naive, grounded = runs["naive"], runs["grounded"]

    echo("")
    echo("Before (naive dense-only RAG) and After (hybrid + rerank + guardrails)")
    echo("-" * 72)
    echo(
        text_table(
            comparison_rows(naive.summary, grounded.summary), ("Metric", "Before", "After", "Note")
        )
    )

    echo("")
    echo("Correct by category")
    echo("-" * 72)
    cat_rows = [
        Row(
            f"{CATEGORY_NAMES.get(cat, cat)} ({v['questions']})",
            str(naive.summary["by_category"][cat]["correct"]),
            str(v["correct"]),
        )
        for cat, v in grounded.summary["by_category"].items()
    ]
    echo(text_table(cat_rows, ("Category (questions)", "Before", "After", "")))

    echo("")
    echo("Questions that changed")
    echo("-" * 72)
    naive_by_id = {r.id: r for r in naive.results}
    for g in grounded.results:
        n = naive_by_id[g.id]
        if n.correct != g.correct or outcome(n) != outcome(g):
            if g.correct and not n.correct:
                mark = "fixed"
            elif n.correct and not g.correct:
                mark = "WORSE"
            else:
                mark = "same"
            echo(f"  {g.id} {mark:<6} before: {outcome(n):<26} after: {outcome(g)}")
            echo(f"         {g.question}")

    echo("")
    echo("Retrieval latency per question")
    echo("-" * 72)
    if rt.embedder.live is not None and rt.reranker.live is not None:
        b = bench(
            rt.store,
            rt.embedder.live,
            rt.reranker.live,
            rt.embedder.cache,
            rt.pipelines["naive"].retriever,
            rt.pipelines["grounded"].retriever,
            rt.configs["naive"].retrieval,
            rt.configs["grounded"].retrieval,
            questions,
        )
        echo(
            f"  Live local models on CPU, {b.questions} questions "
            "(query embedding, search, rerank):"
        )
        echo(f"  Before p50 {ms(b.naive_ms['p50'])}, p95 {ms(b.naive_ms['p95'])}")
        echo(f"  After  p50 {ms(b.grounded_ms['p50'])}, p95 {ms(b.grounded_ms['p95'])}")
        echo(
            "  Cache check: largest live vs committed difference "
            f"{max(b.max_passage_diff, b.max_query_diff):.1e} "
            f"(vectors), {b.max_score_diff:.1e} (rerank scores); rankings identical on "
            f"{b.rankings_match} of {b.questions} questions."
        )
    else:
        echo("  Skipped: local models are not downloaded, so the scores were replayed from index/.")
        echo("  Run `make models` once, then `make demo` again, for real CPU timings.")
    echo("  Generation latency: not measured, because the generator responses are a fixture.")

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated": today(),
        "retrieval": "real local models" if rt.live_models else "replayed real model outputs",
        "llm": "hand-crafted fixture",
        "corpus": f"{len(docs)} docs, {report.chunks} chunks",
        "honesty": HONESTY,
    }
    save_run(out / "naive.json", naive, meta)
    save_run(out / "grounded.json", grounded, meta)
    write_html(out / "report.html", html_report(naive, grounded, ablation, meta))

    echo("")
    baseline = paths.baselines / "grounded.json"
    code = 0
    if baseline.is_file():
        gate = compare(baseline, grounded)
        echo(
            f"Regression gate against {_show(baseline, root)}: {'PASS' if gate.passed else 'FAIL'}"
        )
        for line in gate.regressions:
            echo(f"  regression: {line}")
        code = 0 if gate.passed else 1
    errors = naive.summary["errors"] + grounded.summary["errors"]
    if errors:
        echo(f"{errors} question(s) errored. See the JSON reports.")
        code = 2
    echo(f"Reports: {_show(out / 'report.html', root)}, naive.json, grounded.json")
    rt.close()
    return code
