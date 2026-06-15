# agentic-rag-router

**A retrieval system that knows when to say "I don't know", and proves it.**

Most RAG systems answer every question confidently, even when the evidence does
not support an answer, which is a real liability in production. This one routes
each question to the right source (document search, a SQL database, or the live
web), checks whether what it found actually supports an answer, and refuses with
zero citations when it does not. Every decision, which source to use, whether the
evidence is good enough, and whether to answer or refuse, is reproducible and
shown step by step in the response.

Concretely: per question, the router picks one of three tools (semantic search
over a research-paper corpus, a read-only SQL query against a three-million-row
table, or a live web search), grades the evidence against a fixed set of rules,
and answers with citations or refuses. The refusal behaviour is the point. A
router that confidently answers everything is easy; one that knows when it
*cannot* ground an answer, and proves it, is the hard part.

It runs a hand-written agentic loop over the Claude API with no orchestration
framework, and uses no LLM-as-judge anywhere: every routing, grading, and refusal
decision is deterministic and reproducible.

## See it decide

*Ask something the data can answer.* The router picks SQL, runs it, grades the
result good enough, and answers with a citation:

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

*Ask something unknowable.* The router can run a query and even grade the result
good enough, but the question asks for a future value no data can settle, so it
refuses with **zero citations** rather than dressing up a guess:

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

### Reading a refusal

The refusal above is worth a closer look, because it shows the key rule: a tool
result that ran fine does **not** force the model to answer.

```jsonc
{
  "trajectory": [
    {
      "tool": "sql_query",                 // the router did pick a route and run a tool
      "grade": "sufficient"                 // the COUNT(*) executed fine; past trips ARE countable
    }
  ],
  "refusal_reason": "no_supporting_evidence", // but next Saturday is unknowable, so the model refused
  "citations": []                             // refusals always carry zero citations (enforced)
}
```

## Evaluation

Quality is measured against a 60-question, hand-labelled test set covering five
question types (document, SQL, web, no-answer, and hybrid), including hard cases
that *look* answerable by one source but are not. Scoring is deterministic and
computed purely from the response, with no LLM grader. Latest run (temperature 0,
full report in [`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md)):

| Metric | Result | Target |
| --- | --- | --- |
| Routing accuracy (right source chosen, 48 answerable questions) | **1.00** (48/48) | ≥ 0.85 |
| Refusal correctness (12 unanswerable questions refused, zero citations) | **1.00** (12/12) | = 1.00 |
| Over-refusals (answerable questions wrongly refused) | **0** | = 0 |
| Citation coverage (answered questions carrying ≥ 1 citation) | **1.00** (48/48) | n/a |

For context, always calling a single fixed tool ("always search the web") scores
only **0.40** on the same questions, so the routing is doing real work, not
riding a lucky default.

**Held-out check.** The 60-question set above was also the tuning target (the
tool descriptions and prompt were iterated against it), so on its own it measures
fit to a known target. A separate 15-question set, written after the router was
final and never used to tune anything, scores the same: routing **1.00** (11/11),
refusal **1.00** (4/4), zero over-refusals. An 8-probe adversarial red-team
(prompt injection, off-topic questions, out-of-scope requests, and questions
phrased to bait an over-refusal) is reported alongside it. Full results and an
honest note on where the design has no automated defence yet are in
[`docs/HOLDOUT_AND_REDTEAM.md`](docs/HOLDOUT_AND_REDTEAM.md).

Refusal correctness rests on **two independent checks**. The first is a model
*sentinel*: when the evidence does not support an answer, the model replies
`REFUSE: …`, surfaced as `refusal_reason: "no_supporting_evidence"`. The second
is a deterministic *backstop* (`"insufficient_evidence"`) that suppresses any
answer not resting on at least one good-enough result, even if the model tried to
answer. Every refusal records which check fired.

### Reproduce offline, no keys

Both `POST /ask` examples above are saved as recorded responses, so you can
replay the exact routing and refusal decisions with no API key, no database, and
no network:

    make demo-replay

The Claude turns come from committed recordings; the SQL and document sources are
in-memory stand-ins returning the same evidence shown above. Re-record after a
prompt or loop change with
`RUN_LIVE=1 ANTHROPIC_API_KEY=sk-... uv run pytest tests/replay/ --record-mode=once`.

## How it works

- **Three tools, and their descriptions are the routing policy.**
  `vector_search` (semantic search over ~11k arXiv CS-paper abstracts in
  pgvector), `sql_query` (a single read-only `SELECT` against a 3-million-row NYC
  yellow-taxi table, written by the model), and `web_search` (live Tavily). The
  model decides which tool to use by reading each tool's description, so those
  descriptions are tuned like code, not written as documentation.
- **A hand-rolled agentic loop, no framework.** The first turn forces the model
  to commit to a tool so it cannot answer from memory; the middle turns relax so
  it can stop once it has evidence; the final turn forbids tools entirely, so the
  model must answer or refuse from what it already gathered instead of searching
  forever. Parallel tool calls in one turn are each answered. The model runs at
  **temperature 0**, so the route and the refusal are reproducible.
- **Deterministic evidence grading.** Each tool result is graded
  `sufficient` / `weak` / `none` by fixed rules (documents on a similarity
  threshold, web on whether a real source URL came back, SQL on whether the query
  executed). Citations come **only** from `sufficient` results; refusals carry
  none.
- **A thin FastAPI service.** `POST /ask` returns the full response shown above:
  the answer, its citations, a step-by-step record of every tool call and its
  grade, the machine-readable `refusal_reason`, and how many model turns it took.

## Data sources

Two real, public datasets back the retrieval tools, chosen so the SQL and
document paths exercise genuine scale instead of toy data:

- **SQL: NYC taxi trips.** The `taxi_trips` table is New York City Taxi &
  Limousine Commission yellow-taxi trip records for January 2024, loaded from the
  TLC's public parquet release: **2,964,624 rows** across 19 columns (fares,
  distances, timestamps, passenger counts, payment types, pickup/dropoff zones).
  At ~3 million rows, an aggregate query hits a realistic table, not a handful of
  seeded rows. The model reaches it through a read-only `router_ro` database role
  that can only run `SELECT`, so a generated query can never modify data.
- **Documents: arXiv CS abstracts.** The `corpus_docs` table holds **11,194
  arXiv paper abstracts** from four computer-science categories (cs.CL, cs.LG,
  cs.AI, cs.IR), pulled from the public arXiv API, deduplicated, and spanning
  October 2025 to June 2026. Each abstract is embedded locally with
  `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, pinned to an exact
  model revision), so the embeddings are deterministic and need no API key.
  Claude stays the only paid API in the project.

The arXiv corpus has a fixed cutoff of **2026-06-11**, the date of the newest
paper ingested. That cutoff is a contract, not trivia: the test set's web
questions are defined as facts dated *after* it, so the live web search is the
only path that can answer them. Full provenance, source URLs, the exact column
schema, and the parquet-to-table column mapping are in
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Limitations

Read these before trusting the numbers:

- **The gate set doubled as the tuning target.** The 60 questions were authored
  and frozen *before* any router code existed, but the tool descriptions and
  system prompt were iterated against them, so on their own those metrics measure
  fit to a known target. The held-out set in the Evaluation section addresses
  this: it scores the same 1.00 routing and refusal without ever being tuned on.
- **Refusal rests primarily on the model honouring the sentinel.** The
  deterministic backstop is genuine defence-in-depth, but in live runs it has
  never been the check that fired; every real refusal came from the model's
  sentinel. The backstop has only fired in unit tests.
- **The data sources are fixed snapshots.** The taxi table is NYC TLC yellow-taxi
  trips for January 2024; the arXiv corpus has a fixed cutoff (2026-06-11).
  Current-events questions, answered via web search, are the only live-data path.

## Quickstart

Bring up the data layer, load the sources, and run the service:

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY and TAVILY_API_KEY
uv sync --all-extras

# 1. Postgres + pgvector (digest-pinned; host loopback 127.0.0.1:5436 only).
docker compose up -d postgres

# 2. Heavy ingest deps (pyarrow, sentence-transformers, torch), kept out of the
#    default sync so the test/lint path stays lean.
uv sync --group ingest

# 3. Schema, vector extension, and the SELECT-only router_ro role.
uv run python -m scripts.init_db

# 4. Load both data sources (safe to re-run).
uv run python -m scripts.ingest_taxi     # ~3M taxi trips
uv run python -m scripts.ingest_corpus   # ~11k arXiv abstracts + embeddings

# 5. Serve POST /ask (needs .env on the environment for the live sources).
set -a && . ./.env && set +a
uv run uvicorn agentic_rag_router.api.app:app
```

Connection settings (`POSTGRES_*`, `ROUTER_RO_*`) come from `.env`; see
`.env.example` and [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for where the
data comes from.

Re-run the evaluation (live model + sources; rewrites `eval/`):

```bash
set -a && . ./.env && set +a
uv run python scripts/run_eval.py
```

`make check` runs the offline gate (ruff + mypy + bandit + the unit and contract
suite, 389 tests, 100% line coverage on `src/`). A committed eval report is gated
in CI by a test pinned to the frozen question set, so a router change without a
fresh eval run turns CI red.

## Underlying infrastructure

The router is built on a small hexagonal / DDD-lite library that predates it and
serves as the plumbing layer:

- A three-tier LLM adapter stack (Anthropic / OpenAI / Ollama) behind an
  `LLMPort` protocol, with a `FallbackModel` that cascades primary → secondary →
  tertiary on retryable errors. The router does not use this stack; it drives the
  Claude API directly through its own client. The adapter library backs the
  example scripts in `examples/`.
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
