"""Fetch arXiv cs.* abstracts, embed them locally, and load `corpus_docs`.

    python -m scripts.init_db        # once, to create the table
    python -m scripts.ingest_corpus  # fetch + embed + load (re-run safe)

Abstracts are paged from the arXiv API sorted by submission date (newest first)
so the corpus skews recent and the recorded cutoff is a clean, near-present
boundary — the frozen eval set's `web_only` class is defined as "facts after the
corpus cutoff". Embeddings are computed with a local, version-pinned
sentence-transformers model (no API key, deterministic, hermetic). Upserts key
on `arxiv_id`, so re-runs converge instead of duplicating.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence

import httpx
from sentence_transformers import SentenceTransformer

from scripts import _db
from scripts._corpus_parse import ArxivRecord, batched, parse_feed
from scripts.init_db import EMBEDDING_DIM

# cs.* categories the eval set draws vector questions from. Each contributes a
# quota of *new* (deduplicated) abstracts so the corpus is balanced across all
# four, rather than one busy category filling the whole target.
CATEGORIES = ("cs.CL", "cs.LG", "cs.AI", "cs.IR")
PER_CATEGORY_NEW = 2_750  # 4 * 2750 = 11k unique, margin over the 10k floor

ARXIV_API = "https://export.arxiv.org/api/query"
PAGE_SIZE = 200  # arXiv-friendly page; deep offsets get flaky, so quotas stay shallow
REQUEST_DELAY_S = 3.0  # arXiv asks for >= 3s between programmatic requests
MAX_PAGES_PER_CATEGORY = 40  # runaway guard (kept well clear of deep-offset limits)
USER_AGENT = "agentic-rag-router/0.1 (arXiv metadata harvest for a local RAG corpus)"

# Pinned model: 384-dim, small, CPU-friendly. Revision pins the exact weights.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_BATCH = 256

UPSERT_SQL = """
INSERT INTO corpus_docs (arxiv_id, title, abstract, categories, published_date, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (arxiv_id) DO UPDATE SET
    title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    categories = EXCLUDED.categories,
    published_date = EXCLUDED.published_date,
    embedding = EXCLUDED.embedding
"""


def _fetch_page(client: httpx.Client, category: str, start: int) -> list[ArxivRecord]:
    resp = client.get(
        ARXIV_API,
        params={
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    resp.raise_for_status()
    return parse_feed(resp.content)


def fetch_records() -> dict[str, ArxivRecord]:
    """Collect a per-category quota of new abstracts, newest first, deduped by id."""
    unique: dict[str, ArxivRecord] = {}
    with httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT}) as client:
        for category in CATEGORIES:
            added = 0
            for page in range(MAX_PAGES_PER_CATEGORY):
                records = _fetch_page(client, category, page * PAGE_SIZE)
                if not records:
                    break  # category exhausted
                new = sum(_remember(unique, rec) for rec in records)
                added += new
                print(f"  {category} page {page}: +{new} new -> {len(unique)} unique total")
                if added >= PER_CATEGORY_NEW:
                    break
                time.sleep(REQUEST_DELAY_S)
    return unique


def _remember(unique: dict[str, ArxivRecord], rec: ArxivRecord) -> int:
    """Insert rec if unseen; return 1 if it was new, else 0."""
    if rec.arxiv_id in unique:
        return 0
    unique[rec.arxiv_id] = rec
    return 1


def _format_vector(values: Sequence[float]) -> str:
    """pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def embed_and_load(records: list[ArxivRecord]) -> tuple[int, int]:
    """Embed abstracts and upsert into corpus_docs. Returns (loaded, dim)."""
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    dim = int(model.get_sentence_embedding_dimension())
    if dim != EMBEDDING_DIM:
        raise SystemExit(f"model dim {dim} != table dim {EMBEDDING_DIM}; update init_db")

    loaded = 0
    with _db.connect() as conn, conn.cursor() as cur:
        for chunk in batched(records, EMBED_BATCH):
            vectors = model.encode(
                [r.abstract for r in chunk],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            params = [
                (r.arxiv_id, r.title, r.abstract, r.categories, r.published_date, _format_vector(v))
                for r, v in zip(chunk, vectors, strict=True)
            ]
            cur.executemany(UPSERT_SQL, params)
            loaded += len(params)
            print(f"  embedded+loaded {loaded}/{len(records)}")
        conn.commit()

        cur.execute("SELECT count(*), max(published_date) FROM corpus_docs")
        row = cur.fetchone()
    total = row[0] if row else 0
    cutoff = row[1] if row else None
    print(f"corpus_docs: {total} docs, embedding dim {dim}")
    print(f"corpus cutoff date (max published_date): {cutoff}")
    return loaded, dim


def main() -> None:
    print(f"fetching arXiv abstracts ({', '.join(CATEGORIES)}), newest first")
    unique = fetch_records()
    if len(unique) < 10_000:
        print(f"WARNING: only {len(unique)} unique docs fetched (< 10k floor)", file=sys.stderr)
    embed_and_load(list(unique.values()))


if __name__ == "__main__":
    main()
