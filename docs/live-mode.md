# Live mode (optional, costs money)

Everything in this repo runs without a key. Live mode swaps the hand-crafted
generator and judge fixtures for real Claude calls.

## Checklist

1. Create an Anthropic Console account and an API key at console.anthropic.com. Set a monthly spend limit on the key first.
2. `cp .env.example .env`, paste the key into `.env` (it is gitignored), then `set -a; source .env; set +a`.
3. Optional but recommended: `make models` downloads the two local models (about 150 MB, free, no account), so new questions can be embedded and reranked.
4. `make record` re-records both pipelines on the 50 test questions and writes real cassettes to `fixtures/cassettes/` with `"provenance": "recorded"`.
   - Estimated cost from the fixture's prompt sizes: about $0.26 per pipeline at list prices, so about $0.52 for both. The real figure will be higher because thinking tokens are not in the estimate.
   - `GA_BUDGET_USD` (default $2.00) stops each `ga eval` run from scheduling more questions once it has spent that much.
5. Run `make demo` again. It now replays the real recordings, and the generation latency line shows measured numbers.
6. Update the README numbers and the honesty note from the new `docs/demo-output.txt`, then save a new baseline: `ga eval --pipeline grounded --save-baseline baselines/grounded.json`.
7. `ga craft --check` skips recorded cassettes, so CI stays green with a mix of recorded and hand-crafted files.

## One question, live

```bash
ga ask "How much deposit does a patient pay before getting a porcelain crown?" --mode live
```

Any question works in live mode. In replay mode only the eval questions have cassettes.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Needed for `make record` and `--mode live` only |
| `GA_GENERATOR_MODEL` | `claude-opus-5-5` | Answers the question |
| `GA_JUDGE_MODEL` | `claude-sonnet-5-5` | Grades answers against the reference |
| `GA_FALLBACKS` | `default` | Server side refusal fallbacks (`server-side-fallback-2026-07-01`). `off` to disable |
| `GA_BUDGET_USD` | `2.00` | Hard stop per eval run |
| `GA_MODELS_DIR` | `.cache/models` | Where the local models live |

## What the adapter sends

- `output_config.format` with a JSON schema: answer, claims (statement, source ids, quote), confidence, refused, refusal reason. The response is validated again in code.
- `output_config.effort: low` for the generator and `medium` for the judge. Claude Opus 5.5 always thinks adaptively, so effort is the only control and the request sends no `thinking` field.
- `betas: ["server-side-fallback-2026-07-01"]` and `fallbacks: "default"`, so a safety classifier decline is retried server side instead of returned. A `refusal` stop reason that still comes back is an error, never an answer.
- The SDK retries connection errors, 408, 409, 429 and 5xx with backoff, raised here to 4 retries.

These request shapes are unit tested against a mocked SDK. They have not yet been sent to the live API from this repo, so the first `make record` is also the first live test of the adapter.
