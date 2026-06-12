# Data sources (D2)

The router's two retrieval substrates and their provenance. Both are loaded by
idempotent scripts under `scripts/`; see the README "Data layer" section for the
setup sequence. Raw downloads live under `data/raw/` (gitignored).

## SQL substrate — NYC TLC yellow-taxi trips

| Field | Value |
| --- | --- |
| Table | `taxi_trips` (SELECT-only via the `router_ro` role) |
| Source | NYC Taxi & Limousine Commission trip-record data (public) |
| Page | <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> |
| File | `yellow_tripdata_2024-01.parquet` |
| URL | <https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet> |
| Month | 2024-01 |
| Rows loaded | 2,964,624 (≥ 500k required) |
| Loader | `python -m scripts.ingest_taxi` (download → `TRUNCATE` + `COPY`) |

The table columns bind the **frozen** `EVAL_RUBRIC.md` §4 schema. The mapping
layer (`scripts/_taxi_mapping.py`) absorbs the parquet's CamelCase drift; the
rubric never moves:

| Parquet column | `taxi_trips` column |
| --- | --- |
| `VendorID` | `vendor_id` |
| `RatecodeID` | `rate_code_id` |
| `PULocationID` | `pulocationid` |
| `DOLocationID` | `dolocationid` |
| `Airport_fee` | `airport_fee` |
| _(all others)_ | unchanged (already lowercase) |

Columns are stored lowercase; because Postgres folds unquoted identifiers to
lowercase, SQL written with the rubric's mixed-case names still resolves.

## Vector substrate — arXiv cs.* abstracts

| Field | Value |
| --- | --- |
| Table | `corpus_docs` (`arxiv_id`, `title`, `abstract`, `categories`, `published_date`, `embedding vector(384)`) |
| Source | arXiv API (<https://export.arxiv.org/api/query>) |
| Categories | cs.CL, cs.LG, cs.AI, cs.IR |
| Query | `cat:<category>`, `sortBy=submittedDate`, `sortOrder=descending` (newest first) |
| Etiquette | 200/page, ≥ 3 s between requests, descriptive User-Agent |
| Docs loaded | 11,194 (≥ 10,000 required), deduplicated by version-less `arxiv_id` |
| `published_date` range | 2025-10-25 … 2026-06-11 |
| Loader | `python -m scripts.ingest_corpus` (fetch → embed → upsert) |

### Embedding model (pinned)

| Field | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Dimensions | 384 |
| Normalisation | L2-normalised (`normalize_embeddings=True`) |
| Index | HNSW, `vector_cosine_ops` |

Embeddings are computed locally — no API key, deterministic, hermetic. Claude
stays the only LLM API in the project.

## Corpus cutoff date

**2026-06-11** — the latest `published_date` ingested into `corpus_docs`.

This is a **contract value**, not trivia: the frozen eval set's `web_only` class
is defined as "facts after the corpus cutoff", so questions in that class must
concern events/papers dated after 2026-06-11. Re-running `ingest_corpus` later
will advance this date; update this section in the same change.
