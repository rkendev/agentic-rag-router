# agentic-rag-router

A retrieval router that decides, **per question**, whether to search a vector
corpus, query a SQL database, or search the web, then grades its own evidence
deterministically, answers with citations and a full tool trajectory, and
**refuses with zero citations when no evidence supports an answer**. The refusal
behaviour is the point: a router that confidently answers everything is easy; one
that knows when it *cannot* ground an answer, and proves it, is the hard part.

It runs a hand-written agentic loop over the Claude API, three tool substrates
(pgvector / Postgres / Tavily), and a deterministic evidence-grading rubric. No
LLM-as-judge anywhere: every routing, grading, and refusal decision is
reproducible.

## See it decide

Answerable question. The router writes SQL, runs it, grades the result
`sufficient`, and answers with a citation:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was the average trip distance across all taxi trips in the dataset?"}'
```

```json
{
  "answer": "The average trip distance across all taxi trips in the dataset is approximately **3.65 miles**. This is computed over the full ~3 million NYC yellow-taxi trips recorded in January 2024.",
  "citations": [
    { "tool": "sql_query", "source": "taxi_trips" }
  ],
  "trajectory": [
    {
      "tool": "sql_query",
      "input": { "sql": "SELECT AVG(trip_distance) AS avg_trip_distance FROM taxi_trips" },
      "latency_ms": 1123,
      "ok": true,
      "error_code": null,
      "grade": "sufficient"
    }
  ],
  "refusal_reason": null,
  "iterations": 2
}
```

Unanswerable question. The router *can* run a tool and even grade it
`sufficient`, but the question asks for an unknowable future value, so it refuses
with **zero citations** rather than dressing up a guess:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Exactly how many taxi trips will occur in New York City next Saturday?"}'
```

```json
{
  "answer": null,
  "citations": [],
  "trajectory": [
    {
      "tool": "sql_query",
      "input": { "sql": "SELECT COUNT(*) FROM taxi_trips" },
      "latency_ms": 249,
      "ok": true,
      "error_code": null,
      "grade": "sufficient"
    }
  ],
  "refusal_reason": "no_supporting_evidence",
  "iterations": 2
}
```

Both responses are real output from the live `POST /ask` service (answers
lightly trimmed for length).

### Reading a refusal trajectory

The refusal above is worth annotating, because it shows the key invariant: a
`sufficient` tool result does **not** oblige the model to answer:

```jsonc
{
  "trajectory": [
    {
      "tool": "sql_query",                 // the router did pick a route and run a tool
      "grade": "sufficient"                 // the COUNT(*) executed fine; historical trips ARE countable
    }
  ],
  "refusal_reason": "no_supporting_evidence", // …but next Saturday is unknowable; the model refused
  "citations": []                             // refusals always carry zero citations (enforced)
}
```

## Evaluation

Quality is gated against a 60-question, hand-labelled golden set (vector / SQL /
web / no-answer / hybrid classes, including adversarial near-misses that *look*
answerable by one substrate but are not). Scoring is deterministic, computed
purely from the response envelope; there is no LLM grader. Latest run
(temperature 0, full report in [`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md)):

| Metric | Result | Gate |
| --- | --- | --- |
| Routing accuracy (first tool ∈ acceptable, 48 answerable) | **1.00** (48/48) | ≥ 0.85 |
| Refusal correctness (12 no-answer questions refused, zero citations) | **1.00** (12/12) | = 1.00 |
| Over-refusals (answerable questions wrongly refused) | **0** | = 0 |
| Citation coverage (answered questions carrying ≥ 1 citation) | **1.00** (48/48) | n/a |

For context, the best constant single-tool policy ("always call X") scores only
**0.40** on the same answerable set, so routing is doing real work, not riding a
lucky default.

Refusal correctness is enforced by **two attributable layers**: a model
*sentinel* (the model replies `REFUSE: …`, surfaced as
`refusal_reason: "no_supporting_evidence"`) and a deterministic *grade backstop*
(`"insufficient_evidence"`) that suppresses any answer resting on no `sufficient`
evidence. Every refusal records which layer fired.

### Reproduce offline, no keys

Both `POST /ask` examples above are pinned as recorded cassettes, so you can
replay the exact routing and refusal decisions with no API key, no database,
and no network:

    make demo-replay

The Anthropic turns come from committed cassettes; the SQL and vector substrates
are in-memory fakes returning the same evidence shown above. Re-record after a
prompt or loop change with
`RUN_LIVE=1 ANTHROPIC_API_KEY=sk-... uv run pytest tests/replay/ --record-mode=once`.

## How it works

- **Three tool adapters.** `vector_search` (semantic search over ~11k arXiv
  CS-paper abstracts in pgvector), `sql_query` (a single read-only `SELECT`
  against a 3M-row NYC yellow-taxi table, authored by the model), and
  `web_search` (live Tavily). The tool **descriptions are the routing policy**:
  the model routes on them, so they are tuned, not documentation.
- **A hand-rolled agentic loop** (no framework). Iteration 0 forces a tool choice
  so the model commits to a route; the middle turns relax so it can stop; the
  final turn *forbids* tools so the model must answer or refuse from the evidence
  it already gathered instead of re-searching forever. Parallel tool calls in one
  turn are each answered. The model runs at **temperature 0**, so the route and
  the refusal are reproducible.
- **Deterministic evidence grading.** Each tool result is graded
  `sufficient` / `weak` / `none` from its envelope (vector on a cosine-similarity
  floor, web on source-URL presence, SQL on successful execution). Citations flow
  **only** from `sufficient` evidence; refusals carry none.
- **A thin FastAPI surface.** `POST /ask` returns the full envelope above:
  answer, citations, the per-step trajectory with grades, the machine-readable
  `refusal_reason`, and the model-turn count.

## Limitations

Read these before trusting the numbers:

- **The eval set doubled as the tuning target.** The 60 questions were authored
  and frozen *before* any router code existed, but the tool descriptions and
  system prompt were iterated against them. So the metrics measure fit to a known
  target; a held-out set is future work.
- **Refusal correctness rests primarily on the model honouring the sentinel
  protocol.** The deterministic grade backstop is genuine defense-in-depth, but
  in live runs it has never been the layer that fired; every no-answer refusal
  came from the model's sentinel. The backstop has fired only in unit tests.
- **The substrates are time-bounded snapshots.** The taxi table is NYC TLC
  yellow-taxi trips for January 2024; the arXiv corpus has a fixed cutoff
  (2026-06-11). "Current" web questions are the only live-data path.

## Quickstart

Bring up the data layer, load the substrates, and run the service:

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY and TAVILY_API_KEY
uv sync --all-extras

# 1. Postgres + pgvector (digest-pinned; host loopback 127.0.0.1:5436 only).
docker compose up -d postgres

# 2. Heavy ingest deps (pyarrow, sentence-transformers → torch), kept out of the
#    default sync so the test/lint path stays lean.
uv sync --group ingest

# 3. Schema, vector extension, and the SELECT-only router_ro role.
uv run python -m scripts.init_db

# 4. Load both substrates (idempotent / re-run safe).
uv run python -m scripts.ingest_taxi     # ~3M taxi trips
uv run python -m scripts.ingest_corpus   # ~11k arXiv abstracts + embeddings

# 5. Serve POST /ask (needs .env on the environment for the live substrates).
set -a && . ./.env && set +a
uv run uvicorn agentic_rag_router.api.app:app
```

Connection settings (`POSTGRES_*`, `ROUTER_RO_*`) come from `.env`; see
`.env.example` and [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for provenance.

Re-run the evaluation (live model + substrates; rewrites `eval/`):

```bash
set -a && . ./.env && set +a
uv run python scripts/run_eval.py
```

`make check` runs the offline gate (ruff + mypy + bandit + the unit/contract
suite, 387 tests, 100% line coverage on `src/`); a committed eval report is gated
in CI by a unit test pinned to the frozen golden set, so a router change without a
fresh eval run turns CI red.

## Underlying infrastructure

The router is built on a small hexagonal / DDD-lite library that long predates
it and is kept as the plumbing layer:

- A three-tier LLM adapter stack (Anthropic / OpenAI / Ollama) behind an
  `LLMPort` protocol, with a `FallbackModel` that cascades primary → secondary →
  tertiary on retryable errors. Used by the example scripts in `examples/`; the
  router itself drives the Claude API directly through its own client.
- A 32-case parametrized contract suite (`tests/contract/`) that every adapter
  must satisfy: the architectural drift detector.
- Strict layering (`domain → application → infrastructure`, one composition
  root). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the dependency rule and the
  layer map.

The router is the product; this adapter library is the substrate it stands on.

---

> Scaffolded from `roy-ai-template@v0.5.0`: the hexagonal layout, the
> contract-suite pattern, the pinned pre-commit/CI chain, and the editor-agnostic
> agent-tooling config under `.claude/` all come from that template.

MIT licensed. See [`LICENSE`](LICENSE). Full change history in
[`CHANGELOG.md`](CHANGELOG.md).
