# Evaluation Rubric — agentic-rag-router routing & refusal

This document is the **measurement contract** for the frozen golden set at
`data/eval/golden_questions.jsonl`. It defines what every label means, how a
routing decision is scored, and how the set is frozen. Tool descriptions,
adapters, and the router are written and tuned **against** this contract in
later tasks — the contract is fixed first so that tuning cannot quietly
redefine "correct".

## 1. Why this exists

The service routes each question to one of three substrates — `vector_search`
(pgvector over arXiv cs.* abstracts), `sql_query` (a read-only NYC taxi-trips
database), `web_search` (Tavily) — or **refuses** when no substrate can ground
an answer. The v0.1.0 gates are:

- **Routing accuracy ≥ 0.85** against a naive always-search baseline, reported
  with a per-class confusion table.
- **Refusal correctness = 1.0** — every `no_answer` question must be refused
  with zero citations.

A frozen, hand-labelled set is the only way to measure either honestly across
iterations.

## 2. Record schema

Each line of `data/eval/golden_questions.jsonl` is one JSON object:

| field | type | meaning |
| --- | --- | --- |
| `id` | str | Stable identifier, `G001`..`G060`, sequential. |
| `question` | str | The user question as routed. |
| `label` | str | One of the five classes in §3. |
| `acceptable_tools` | list[str] | Every tool whose answer counts as a correct route. Subset of `{vector_search, sql_query, web_search}`. Empty for `no_answer`. |
| `adversarial` | bool | `true` for near-miss `no_answer` questions that pattern-match a substrate (see §3). |
| `rationale` | str | One-line justification for the label and tool set. |

## 3. Class definitions

- **`vector_only`** — answerable from arXiv cs.* **abstract-level** knowledge
  (conceptual / explanatory). `acceptable_tools = ["vector_search"]`.
- **`sql_only`** — an aggregation, count, ranking, or filtered statistic
  computable from the assumed taxi schema in §4.
  `acceptable_tools = ["sql_query"]`.
- **`web_only`** — a fact that is live, post-cutoff, or otherwise outside both
  substrates (current prices, office-holders, software releases, sports
  results, real-time status). `acceptable_tools = ["web_search"]`.
- **`no_answer`** — no substrate can ground an answer; the correct behaviour is
  **refusal with zero citations**. `acceptable_tools = []`. This class includes
  two kinds of question:
  - **adversarial near-misses** (`adversarial = true`) — questions that *look*
    answerable by one substrate but are not (e.g. a taxi-shaped query whose
    column does not exist in the schema, a "from the abstract corpus" query
    asking for detail abstracts never contain, or a web-shaped query asking for
    an unknowable future value). These defend the refusal gate against
    pattern-matching and are the hardest cases in the set.
  - **plainly out-of-scope** (`adversarial = false`) — subjective, generative,
    or advice questions with no factual source.
- **`hybrid`** — deliberately ambiguous: more than one substrate can produce a
  correct answer. `acceptable_tools` lists **every** such tool (always ≥ 2).

## 4. Assumed NYC taxi-trips schema (bound in D2)

`sql_only` questions are written against the standard NYC TLC **yellow-taxi**
trip schema below. This is the assumed contract for labelling only; the actual
table, types, and row count are bound when the SQL substrate is wired in a
later task. The database is **SELECT-only**, > 500k rows.

| column | type | notes |
| --- | --- | --- |
| `vendor_id` | int | TPEP provider code. |
| `tpep_pickup_datetime` | timestamp | Trip start. |
| `tpep_dropoff_datetime` | timestamp | Trip end. |
| `passenger_count` | int | Reported passengers. |
| `trip_distance` | float | Miles, as metered. |
| `PULocationID` | int | TLC taxi-zone of pickup. |
| `DOLocationID` | int | TLC taxi-zone of dropoff. |
| `rate_code_id` | int | Rate class. |
| `store_and_fwd_flag` | char | Y/N store-and-forward. |
| `payment_type` | int | 1 = credit card, 2 = cash, 3 = no charge, 4 = dispute, … |
| `fare_amount` | float | Metered fare. |
| `extra` | float | Misc. extras/surcharges. |
| `mta_tax` | float | Fixed MTA tax. |
| `tip_amount` | float | Tip (card trips only, generally). |
| `tolls_amount` | float | Tolls paid. |
| `improvement_surcharge` | float | Fixed surcharge. |
| `total_amount` | float | Total charged to passenger. |
| `congestion_surcharge` | float | Congestion-zone surcharge. |
| `airport_fee` | float | Airport pickup fee. |

A taxi-shaped question that depends on a field **not** in this schema (driver
experience, medallion owner, satisfaction rating, cancellations, …) is
`no_answer`, not `sql_only`. See `G043`–`G046`.

## 5. Scoring

For each question the router emits a chosen tool (or a refusal). Scoring is
deterministic from the labels above.

### 5.1 Routing error (misroute)

- For `vector_only` / `sql_only` / `web_only` / `hybrid`: a **misroute** is
  choosing a tool **outside** `acceptable_tools`. For single-tool classes that
  means any tool other than the one listed; for `hybrid` it means a tool not in
  the listed set (choosing *any* listed tool is correct — see §6).
- For `no_answer`: **any tool-derived answer is a routing error.** Invoking a
  tool and returning a substantive answer fails the question regardless of which
  tool was chosen. The only correct outcome is refusal with zero citations.

**Routing accuracy** = (questions routed to a tool in `acceptable_tools`,
counting refusal as the correct "route" for `no_answer`) ÷ 60. This is the
number compared to the naive always-search baseline, and it is reported with a
per-class confusion table (rows = true label, columns = chosen route including
a `refuse` column).

### 5.2 Over-refusal (a distinct error class)

Refusing a question that **is** answerable (any class other than `no_answer`)
is **over-refusal**. It is *not* folded into routing accuracy as a generic
miss; it is counted and **reported separately**, because the two failures have
opposite fixes (a misroute means the tool descriptions overlap; an over-refusal
means the grading threshold is too strict). The refusal-correctness gate is
therefore two-sided:

- every `no_answer` question is refused (no false answers), **and**
- no answerable question is refused (no over-refusals).

`refusal correctness = 1.0` requires the `no_answer` side to be perfect; the
over-refusal count is reported alongside it and tracked across iterations.

## 6. Hybrid acceptable-answer-overlap rule

A `hybrid` question lists every tool whose answer counts as correct. Routing is
scored as correct if the chosen tool is **any** member of `acceptable_tools` —
there is no single "gold" tool and no partial credit. Hybrids are deliberately
ambiguous so that the router is not penalised for a reasonable substrate choice;
they exist to measure that the router does not *refuse* an answerable question
and does not pick a substrate that cannot answer it at all. A `hybrid` always
has at least two acceptable tools.

## 7. Freeze policy (immutable after merge)

Once the PR introducing this file and `data/eval/golden_questions.jsonl` is
merged to `main`, **both files are frozen and immutable**. The freeze is
enforced mechanically by `tests/test_eval_set_frozen.py`, which pins the
sha256 of each file and validates the JSONL schema.

Changing the golden set or this rubric is **not** an edit to these files. It is:

1. a **new versioned file** — `data/eval/golden_questions_v2.jsonl` (and, if the
   contract changes, a revised rubric section) — leaving the originals intact, and
2. a **logged decision in `CHANGELOG.md`** under `[Unreleased]` stating what
   changed and why, and
3. an update to the freeze test to pin the new file alongside (not instead of)
   the old one for the version in force.

This guarantees that any routing-accuracy number is always traceable to an exact,
unmodified evaluation set, and that "we improved the score" can never silently
mean "we changed the test."
