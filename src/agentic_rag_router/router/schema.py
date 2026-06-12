"""Anthropic tool definitions for the router --- the descriptions ARE the router.

`run_router` (`loop.py`) hands these three tool definitions to Claude Sonnet and
lets the model decide which substrate answers each question. The model routes
purely on the tool ``description`` fields, so those descriptions are not
documentation --- they are the routing policy, tuned against the frozen golden
set via ``scripts/route_probe.py``. Editing a description changes routing
behaviour; editing an adapter (the T003 ``tools/`` package) does not.

Three substrates:

- ``vector_search`` --- conceptual / explanatory ML & CS questions answerable
  from arXiv cs.{CL,LG,AI,IR} *abstract-level* knowledge.
- ``sql_query`` --- aggregate statistics over the NYC yellow-taxi trips table.
  The model authors the SQL, so the table schema is embedded in the description.
- ``web_search`` --- live / current / post-cutoff facts outside both substrates.

The system prompt biases the model toward answering only from tool evidence and
against fabricating from parametric memory; full refusal semantics (evidence
grading, zero-citation refusal) arrive in D5.
"""

from __future__ import annotations

from typing import Any

from agentic_rag_router.tools.envelope import (
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
)

# The corpus cutoff is a *contract value*, not trivia: the frozen eval set's
# `web_only` class is defined as "facts after the corpus cutoff". The canonical
# source is `docs/DATA_SOURCES.md` ("Corpus cutoff date"); kept here as a
# constant so the two tool descriptions that reference it stay in lockstep.
CORPUS_CUTOFF = "2026-06-11"

# The NYC TLC yellow-taxi schema the model writes SQL against. Bound from the
# frozen `docs/EVAL_RUBRIC.md` section 4. Column names are presented lowercase
# because that is how they are stored (`docs/DATA_SOURCES.md`): Postgres folds
# unquoted identifiers to lowercase, so the model's SQL resolves either way.
TAXI_SCHEMA_BLOCK = """\
- vendor_id (int) --- TPEP provider code
- tpep_pickup_datetime (timestamp) --- trip start
- tpep_dropoff_datetime (timestamp) --- trip end
- passenger_count (int) --- reported passengers
- trip_distance (float) --- miles, as metered
- pulocationid (int) --- TLC taxi-zone of pickup
- dolocationid (int) --- TLC taxi-zone of dropoff
- rate_code_id (int) --- rate class
- store_and_fwd_flag (char) --- Y/N store-and-forward
- payment_type (int) --- 1=credit card, 2=cash, 3=no charge, 4=dispute
- fare_amount (float) --- metered fare
- extra (float) --- misc. extras / surcharges
- mta_tax (float) --- fixed MTA tax
- tip_amount (float) --- tip (card trips only, generally)
- tolls_amount (float) --- tolls paid
- improvement_surcharge (float) --- fixed surcharge
- total_amount (float) --- total charged to passenger
- congestion_surcharge (float) --- congestion-zone surcharge
- airport_fee (float) --- airport pickup fee"""

# --- Routing-contract descriptions -----------------------------------------
# These strings are the tunable surface. They are deliberately explicit about
# what each tool IS and IS NOT for, because the only signal the router has is
# the description text.

_VECTOR_SEARCH_DESCRIPTION = f"""\
Semantic search over a corpus of ~11,000 arXiv computer-science paper ABSTRACTS \
(categories cs.CL, cs.LG, cs.AI, cs.IR; published up to {CORPUS_CUTOFF}).

USE THIS for CONCEPTUAL or EXPLANATORY questions about established \
machine-learning, NLP, information-retrieval, and computer-science ideas, \
methods, architectures, trade-offs, and definitions --- the kind of question \
answerable from abstract-level knowledge. Typical shapes: "what is X", "how \
does X work", "how does X differ from Y", "what problem does X address", "why \
use X". Examples of in-scope topics: self-attention and transformers, \
contrastive and self-supervised learning, batch/layer normalization, dense vs. \
sparse (BM25) retrieval, catastrophic forgetting, variational autoencoders, \
diffusion models, mixture-of-experts.

DO NOT use this for: live or current facts, dates, prices, releases, or events; \
numeric statistics computed over a specific dataset (use sql_query); or anything \
requiring information published after {CORPUS_CUTOFF} (use web_search).

Args: query (a natural-language description of the concept to find), \
k (optional: how many abstracts to return, default 5)."""

_SQL_QUERY_DESCRIPTION = f"""\
Run a single read-only SQL SELECT against `taxi_trips`, a Postgres table of NYC \
TLC yellow-taxi trips for January 2024 (~3,000,000 rows).

USE THIS whenever the answer is a STATISTIC computed over the trip records: \
counts, sums, averages, minimums/maximums, rankings, distributions, or \
filtered aggregates. Typical shapes: "how many trips ...", "what is the average \
/ total / median ...", "which pickup zone is busiest", "what share of trips \
paid by cash", "busiest hour of day". You author the SQL yourself.

Rules: the SQL must be a SINGLE statement beginning with SELECT (SELECT-only --- \
no INSERT/UPDATE/DELETE/DDL/multiple statements); at most 200 rows are returned, \
so aggregate rather than dumping raw rows.

Table `taxi_trips` columns (all lowercase):
{TAXI_SCHEMA_BLOCK}

This table has ONLY the columns above. A question that depends on a field not \
listed --- driver pay or experience, medallion owner, passenger satisfaction, \
ride ratings, cancellations, weather --- is NOT answerable here; do not invent \
columns. Args: sql (the single SELECT statement)."""

_WEB_SEARCH_DESCRIPTION = f"""\
Live web search (Tavily). USE THIS for CURRENT, real-time, or TIME-SENSITIVE \
facts, and for anything dated AFTER the corpus cutoff of {CORPUS_CUTOFF}: \
current prices or exchange rates, recent news and events, software/version \
releases, sports results and standings, who currently holds an office or title, \
weather, and real-time status.

USE THIS as the fallback whenever a question is neither (a) an established \
CS/ML concept available in the arXiv abstract corpus (use vector_search) nor \
(b) a statistic derivable from the NYC taxi-trips table (use sql_query). Words \
like "latest", "current", "today", "now", "this year", or any specific date \
after {CORPUS_CUTOFF} are strong signals for this tool.

Args: query (the web search string), max_results (optional: how many results to \
return, default 5)."""

SYSTEM_PROMPT = """\
You are a retrieval router. Answer the user's question using ONLY evidence \
returned by the provided tools:
- vector_search: conceptual ML/CS knowledge from arXiv paper abstracts.
- sql_query: statistics over a NYC yellow-taxi trips database (you write SQL).
- web_search: live, current, or post-cutoff facts from the web.

Choose the single most appropriate tool for the question and call it. If a tool \
result does not answer the question, you may try a different tool. Base your \
final answer strictly on the evidence the tools return, and ground it in that \
evidence. Do NOT answer from your own prior knowledge or memory: if the tools \
return no evidence that supports an answer, say plainly that you cannot answer \
rather than fabricating one."""


def build_tools() -> list[dict[str, Any]]:
    """Return the three Anthropic tool definitions, freshly built.

    Each is a plain ``dict`` in the Anthropic Messages ``tools`` shape
    (``name`` / ``description`` / ``input_schema``). A function (rather than a
    bare module constant the caller might mutate) keeps the routing contract
    immutable from the router's point of view --- the probe and the loop each
    get their own copy.
    """
    return [
        {
            "name": TOOL_VECTOR_SEARCH,
            "description": _VECTOR_SEARCH_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the concept to find.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many abstracts to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": TOOL_SQL_QUERY,
            "description": _SQL_QUERY_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single read-only SELECT against taxi_trips.",
                    },
                },
                "required": ["sql"],
            },
        },
        {
            "name": TOOL_WEB_SEARCH,
            "description": _WEB_SEARCH_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    ]


# Module-level convenience copy for callers that just want "the tools".
TOOLS: list[dict[str, Any]] = build_tools()
