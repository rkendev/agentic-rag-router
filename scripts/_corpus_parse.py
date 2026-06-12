"""Pure parsing + batching helpers for the arXiv vector substrate.

Stdlib only (`xml.etree`) so it loads without the `ingest` dependency group and
is unit-testable without network or `sentence-transformers`. `ingest_corpus`
supplies the HTTP fetch, the embedding model, and the database writes; the
fragile bits — Atom feed shape, id/version stripping, whitespace normalisation,
embedding batch sizing — live here where tests can pin them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

# Input is a trusted arXiv API feed; staying stdlib-only keeps this module
# importable without the `ingest` dependency group (the unit tests rely on it).
from xml.etree import ElementTree as ET  # nosec B405

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# Trailing version suffix on an arXiv id, e.g. "2406.01234v3" -> "2406.01234".
_VERSION_RE = re.compile(r"v\d+$")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ArxivRecord:
    """One arXiv abstract, normalised for the `corpus_docs` table."""

    arxiv_id: str  # version-stripped, e.g. "2406.01234"
    title: str
    abstract: str
    categories: list[str]  # all listed category terms; primary first
    published_date: date


def _normalise(text: str | None) -> str:
    """Collapse the line-wrapped whitespace arXiv puts in titles/abstracts."""
    return _WS_RE.sub(" ", (text or "").strip())


def strip_version(raw_id: str) -> str:
    """Reduce an arXiv id URL/string to its version-less identifier.

    Handles modern ids (``2406.01234``) and old-style category-prefixed ids
    (``cs/0501001``); only the ``.../abs/`` URL prefix and a trailing ``vN`` are
    removed, never the category segment.
    """
    tail = raw_id.rstrip("/")
    if "/abs/" in tail:
        tail = tail.split("/abs/", 1)[1]
    return _VERSION_RE.sub("", tail)


def _parse_entry(entry: ET.Element) -> ArxivRecord | None:
    raw_id = entry.findtext(f"{_ATOM}id")
    published = entry.findtext(f"{_ATOM}published")
    title = _normalise(entry.findtext(f"{_ATOM}title"))
    abstract = _normalise(entry.findtext(f"{_ATOM}summary"))
    if not raw_id or not published or not title or not abstract:
        return None

    # Primary category first (if present), then any remaining category terms,
    # de-duplicated while preserving order.
    ordered: list[str] = []
    primary = entry.find(f"{_ARXIV}primary_category")
    if primary is not None and primary.get("term"):
        ordered.append(primary.get("term", ""))
    for cat in entry.findall(f"{_ATOM}category"):
        term = cat.get("term")
        if term and term not in ordered:
            ordered.append(term)

    return ArxivRecord(
        arxiv_id=strip_version(raw_id),
        title=title,
        abstract=abstract,
        categories=ordered,
        published_date=date.fromisoformat(published[:10]),
    )


def parse_feed(xml_bytes: bytes) -> list[ArxivRecord]:
    """Parse one arXiv Atom response into records, skipping malformed entries."""
    root = ET.fromstring(xml_bytes)  # nosec B314 - trusted arXiv API feed
    records = [rec for entry in root.findall(f"{_ATOM}entry") if (rec := _parse_entry(entry))]
    return records


def batched(items: Iterable[ArxivRecord], size: int) -> Iterator[list[ArxivRecord]]:
    """Yield successive `size`-length chunks (the last may be shorter)."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    chunk: list[ArxivRecord] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
