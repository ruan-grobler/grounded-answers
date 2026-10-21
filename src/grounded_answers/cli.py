"""The `ga` command line."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from grounded_answers.baseline import compare, save_baseline
from grounded_answers.cassette import CassetteMiss
from grounded_answers.config import PIPELINES, ConfigError, find_root
from grounded_answers.crafting import (
    ScriptedClaim,
    ScriptedClient,
    craft_cassettes,
    diff_cassettes,
    load_scripts,
    write_cassettes,
)
from grounded_answers.evalset import EvalSetError, check_evidence, load_questions, normalise
from grounded_answers.evaluate import run_eval
from grounded_answers.inference import CacheMiss, ModelUnavailable
from grounded_answers.llm import LLMError
from grounded_answers.report import pct, save_run, today
from grounded_answers.retrieval import MODES
from grounded_answers.runtime import Runtime, client_factory, open_runtime
from grounded_answers.store import StoreError
from grounded_answers.tuning import tune, tune_config

VECTOR_TOLERANCE = 1e-3
SCORE_TOLERANCE = 1e-2


def echo(text: str = "") -> None:
    print(text, flush=True)


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root).resolve() if args.root else find_root()


# ------------------------------------------------------------------ commands


def cmd_index(args: argparse.Namespace) -> int:
    root = _root(args)
    rt = open_runtime(
        root,
        model_mode=args.models,
        record_models=True,
        allow_download=args.download,
        force_index=args.force,
    )
    report = rt.index_report
    echo(
        f"{report.documents} documents, {report.chunks} chunks, "
        f"index {'rebuilt' if report.rebuilt else 'already current'}."
    )
    for chunk_id, flags in report.flagged.items():
        echo(f"  flagged {chunk_id}: {', '.join(flags)}")
    # Warm the caches for every eval question, in every retrieval mode both pipelines and the
    # ablation use, so replay (CI, `make demo`) never needs a model.
    questions = load_questions(rt.paths.questions) + load_questions(rt.paths.dev_questions)
    retrievers = [p.retriever for p in rt.pipelines.values()]
    retrievers += [rt.ablation_retriever(m) for m in MODES]
    retrievers.append(rt.retriever(tune_config(rt.configs["grounded"].retrieval)))
    for question in questions:
        for retriever in retrievers:
            retriever.retrieve(question.question)
    if args.prune:
        dropped = rt.embedder.cache.prune(rt.embedder.used) + rt.reranker.cache.prune(
            rt.reranker.used
        )
        echo(f"Pruned {dropped} unused cache entries.")
    rt.save_caches()
    echo(
        f"Caches: {len(rt.embedder.cache)} vectors, {len(rt.reranker.cache)} rerank scores "
        f"({rt.embedder.live_calls + rt.reranker.live_calls} new from the live models)."
    )
    rt.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    rt = open_runtime(_root(args), model_mode=args.models)
    result = rt.pipelines[args.pipeline].retriever.retrieve(args.question)
    context = {c.chunk.chunk_id for c in result.context}
    echo(
        f"{args.pipeline}: top score {result.top_score:.3f}, gated {result.gated}"
        if result.top_score is not None
        else "no results"
    )
    for i, cand in enumerate(result.ranked[: args.k], 1):
        mark = "*" if cand.chunk.chunk_id in context else " "
        echo(f"{mark}{i:>2}. {cand.score:8.3f}  {cand.chunk.chunk_id:<36} {cand.chunk.section}")
    echo("* = sent to the model")
    rt.close()
    return 0


def _question_for(rt: Runtime, text: str) -> object:
    for q in load_questions(rt.paths.questions) + load_questions(rt.paths.dev_questions):
        if normalise(q.question) == normalise(text):
            return q
    return None


def cmd_ask(args: argparse.Namespace) -> int:
    from grounded_answers.cassette import Cassette, ReplayClient, cassette_path
    from grounded_answers.evalset import Question
    from grounded_answers.llm import AnthropicClient, LLMClient

    rt = open_runtime(_root(args), model_mode=args.models)
    pipeline = rt.pipelines[args.pipeline]
    client: LLMClient
    if args.mode == "live":
        client = AnthropicClient()
    else:
        question = _question_for(rt, args.question)
        if not isinstance(question, Question):
            echo(
                "Replay mode only knows the eval questions. "
                "Use --mode live with ANTHROPIC_API_KEY set,"
            )
            echo("or `ga search` to see what retrieval finds for any question.")
            return 2
        path = cassette_path(rt.paths.cassettes, args.pipeline, question.id)
        cassette = (
            Cassette.load(path)
            if path.is_file()
            else Cassette(args.pipeline, question.id, question.question, "hand-crafted", "")
        )
        client = ReplayClient(cassette)
    result = pipeline.answer(args.question, client)
    echo(result.final_text or "(no answer)")
    echo("")
    echo(f"refused: {result.refused} ({result.refusal_reason}), stage: {result.stage}")
    if result.answer and not result.refused:
        echo(f"confidence: {result.answer.confidence}")
        for claim in result.answer.claims:
            sources = ", ".join(
                result.labels[s].location for s in claim.source_ids if s in result.labels
            )
            echo(f"  - {claim.statement}")
            echo(f"    source: {sources}")
            echo(f'    quote: "{claim.quote}"')
    if result.output_check and result.output_check.reasons:
        echo(f"output guard: {', '.join(result.output_check.reasons)}")
    if result.error:
        echo(f"error: {result.error}")
    rt.close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    root = _root(args)
    rt = open_runtime(root, model_mode=args.models)
    path = rt.paths.dev_questions if args.split == "dev" else rt.paths.questions
    questions = load_questions(path)
    budget = rt.settings.budget_usd if args.mode in ("record", "live") else None
    run = run_eval(
        rt.pipelines[args.pipeline],
        questions,
        client_factory(rt.paths, args.pipeline, args.mode),
        rt.settings,
        use_judge=not args.no_judge,
        budget_usd=budget,
    )
    s = run.summary
    echo(f"{args.pipeline} on {len(run.results)} {args.split} questions ({args.mode} mode)")
    echo(
        f"  end to end correct {pct(s['end_to_end_correct'])}, "
        f"hit@5 {pct(s['retrieval']['hit@5'])}, "
        f"refusal accuracy {pct(s['refusals']['refusal_accuracy'])}, "
        f"pii leaks {s['safety']['pii_leaks']}, errors {s['errors']}"
    )
    if run.skipped_for_budget:
        echo(
            f"  stopped at the ${budget:.2f} budget: "
            f"{len(run.skipped_for_budget)} questions not run"
        )
    out = Path(args.out) if args.out else rt.paths.reports / args.pipeline
    save_run(
        out / f"{args.pipeline}-{args.split}.json", run, {"generated": today(), "mode": args.mode}
    )
    failed = [r for r in run.results if r.error]
    for r in failed:
        echo(f"  error {r.id}: {r.error}")
    if failed:
        # Dead letter: every question that could not be answered, with its error and stage.
        lines = [
            json.dumps({"id": r.id, "question": r.question, "stage": r.stage, "error": r.error})
            for r in failed
        ]
        dead = out / "dead-letter.jsonl"
        dead.write_text("\n".join(lines) + "\n", encoding="utf-8")
        echo(f"  {len(failed)} question(s) written to {dead}")
    code = 2 if s["errors"] else 0
    if args.save_baseline:
        if s["errors"]:
            echo("Not saving a baseline from a run with errors.")
            return 2
        save_baseline(Path(args.save_baseline), run)
        echo(f"Saved baseline {args.save_baseline}")
    if args.baseline:
        gate = compare(Path(args.baseline), run)
        echo(f"Regression gate: {'PASS' if gate.passed else 'FAIL'}")
        for line in gate.regressions:
            echo(f"  regression: {line}")
        for line in gate.improvements:
            echo(f"  improved:   {line}")
        if not gate.passed:
            code = max(code, 1)
    rt.close()
    return code


def cmd_demo(args: argparse.Namespace) -> int:
    from grounded_answers.demo import run_demo

    root = _root(args)
    return run_demo(root, Path(args.out) if args.out else root / "reports" / "demo", echo)


def cmd_craft(args: argparse.Namespace) -> int:
    rt = open_runtime(_root(args), model_mode=args.models)
    questions = load_questions(rt.paths.questions)
    scripts = load_scripts(rt.paths.scripts)
    titles = {c.doc_title: c.doc_id for c in rt.store.chunks}
    problems = check_scripts(rt, scripts)
    if problems:
        for p in problems:
            echo(f"script problem: {p}")
        return 1
    client = ScriptedClient(questions, scripts, titles)
    failures: list[str] = []
    for name in PIPELINES:
        cassettes = craft_cassettes(rt.pipelines[name], questions, client, rt.settings)
        if args.check:
            failures += diff_cassettes(rt.paths.cassettes, name, cassettes, questions)
        else:
            changed = write_cassettes(rt.paths.cassettes, name, cassettes, questions)
            echo(f"{name}: {len(cassettes)} cassettes, {len(changed)} changed")
    if args.check:
        if failures:
            for f in failures:
                echo(f"stale: {f}")
            echo("Run `make craft` and commit the result.")
            return 1
        echo("Cassettes match what `ga craft` builds from the scripts.")
    rt.close()
    return 0


def check_scripts(rt: Runtime, scripts: dict[str, list[ScriptedClaim]]) -> list[str]:
    """Every scripted quote must be in the document it names."""
    text_by_doc: dict[str, str] = {}
    for chunk in rt.store.chunks:
        text_by_doc[chunk.doc_id] = text_by_doc.get(chunk.doc_id, "") + "\n" + normalise(chunk.text)
    problems = []
    for qid, claims in scripts.items():
        for claim in claims:
            for doc, quote in claim.sources:
                if normalise(quote) not in text_by_doc.get(doc, ""):
                    problems.append(f"{qid}: quote not found in {doc}: {quote[:60]!r}")
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    rt = open_runtime(_root(args), model_mode="replay")
    problems = []
    for path in (rt.paths.questions, rt.paths.dev_questions):
        problems += check_evidence(load_questions(path), rt.store.chunks)
    problems += check_scripts(rt, load_scripts(rt.paths.scripts))
    for p in problems:
        echo(p)
    echo(
        "Labels and scripts are consistent with the corpus."
        if not problems
        else f"{len(problems)} problem(s)"
    )
    rt.close()
    return 1 if problems else 0


def cmd_tune(args: argparse.Namespace) -> int:
    rt = open_runtime(_root(args), model_mode=args.models)
    dev = load_questions(rt.paths.dev_questions)
    retriever = rt.retriever(tune_config(rt.configs["grounded"].retrieval))
    report = tune(retriever, dev)
    echo("Dev split, top rerank score and best score per evidence hop:")
    for row in report.rows:
        hops = ", ".join("none" if s is None else f"{s:.2f}" for s in row.hop_scores)
        guard = " (blocked by the personal data guard)" if row.blocked_by_guard else ""
        top = "none" if row.top_score is None else f"{row.top_score:.2f}"
        echo(f"  {row.id} {row.expect:<7} top {top:>6}  evidence [{hops}]{guard}")
    echo(f"gate_min_score: {report.gate_min_score}")
    echo(f"keep_min_score: {report.keep_min_score}")
    echo(
        "Dev refusals the gate would stop before the model: "
        f"{report.gated_refusals} of {report.refusals_considered}"
    )
    if args.write:
        path = rt.paths.pipelines
        text = path.read_text(encoding="utf-8")
        text = re.sub(
            r"(gate_min_score:\s*)(-?[\d.]+|null)", rf"\g<1>{report.gate_min_score}", text
        )
        text = re.sub(
            r"(keep_min_score:\s*)(-?[\d.]+|null)", rf"\g<1>{report.keep_min_score}", text
        )
        path.write_text(text, encoding="utf-8")
        echo(f"Wrote the thresholds to {path.relative_to(rt.paths.root)}")
    rt.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from grounded_answers.bench import bench

    rt = open_runtime(_root(args), model_mode="live", allow_download=args.download)
    questions = load_questions(rt.paths.questions)
    assert rt.embedder.live is not None and rt.reranker.live is not None
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
    echo(f"{b.questions} questions, live models ({b.threads_note})")
    echo(f"naive    p50 {b.naive_ms['p50']:.1f} ms  p95 {b.naive_ms['p95']:.1f} ms")
    echo(f"grounded p50 {b.grounded_ms['p50']:.1f} ms  p95 {b.grounded_ms['p95']:.1f} ms")
    echo(
        f"max |live - cached|: passages {b.max_passage_diff:.2e}, "
        f"queries {b.max_query_diff:.2e}, rerank {b.max_score_diff:.2e}"
    )
    echo(f"rankings identical to replay: {b.rankings_match} of {b.questions}")
    rt.close()
    # Another CPU can differ in the last float digits, and a near tie can then swap two
    # chunks. A cache that is not from these models would be off by far more.
    genuine = max(b.max_passage_diff, b.max_query_diff) < VECTOR_TOLERANCE
    genuine = genuine and b.max_score_diff < SCORE_TOLERANCE
    echo("Caches match the live models." if genuine else "Caches do NOT match the live models.")
    return 0 if genuine else 1


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ga", description=__doc__)
    parser.add_argument("--root", help="project root (default: found from the current folder)")
    sub = parser.add_subparsers(dest="command", required=True)

    def models(p: argparse.ArgumentParser, default: str = "auto") -> None:
        p.add_argument(
            "--models",
            choices=["auto", "live", "replay"],
            default=default,
            help="local models: auto uses them if downloaded, replay uses only the committed cache",
        )

    p = sub.add_parser("index", help="build the index and fill the model caches")
    models(p, "live")
    p.add_argument("--download", action="store_true", help="allow downloading the local models")
    p.add_argument("--force", action="store_true", help="rebuild even if the corpus is unchanged")
    p.add_argument("--prune", action="store_true", help="drop cache entries nothing uses")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="show what retrieval finds for a question")
    models(p)
    p.add_argument("question")
    p.add_argument("--pipeline", choices=PIPELINES, default="grounded")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer one question")
    models(p)
    p.add_argument("question")
    p.add_argument("--pipeline", choices=PIPELINES, default="grounded")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the eval set through one pipeline")
    models(p)
    p.add_argument("--pipeline", choices=PIPELINES, default="grounded")
    p.add_argument("--split", choices=["test", "dev"], default="test")
    p.add_argument("--mode", choices=["replay", "record", "live"], default="replay")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--baseline", help="fail (exit 1) on any regression against this baseline")
    p.add_argument("--save-baseline", help="save this run as the new baseline")
    p.add_argument("--out", help="report folder (default reports/<pipeline>)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("demo", help="Before and After on all 50 questions, with reports")
    p.add_argument("--out", help="report folder (default reports/demo)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("craft", help="rebuild the hand-crafted cassettes from the scripts")
    models(p, "replay")
    p.add_argument("--check", action="store_true", help="fail if the committed cassettes are stale")
    p.set_defaults(func=cmd_craft)

    p = sub.add_parser("check", help="check eval labels and scripts against the corpus")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("tune", help="choose the rerank thresholds from the dev split")
    models(p)
    p.add_argument("--write", action="store_true", help="write them to config/pipelines.yaml")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("bench", help="time retrieval with the live models and check the caches")
    p.add_argument("--download", action="store_true", help="allow downloading the local models")
    p.set_defaults(func=cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (
        CacheMiss,
        CassetteMiss,
        ModelUnavailable,
        StoreError,
        ConfigError,
        EvalSetError,
        LLMError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
