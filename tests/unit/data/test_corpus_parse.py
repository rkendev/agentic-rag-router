"""Unit tests for the pure arXiv Atom parsing + batching (no network)."""

from __future__ import annotations

from datetime import date

import pytest

from scripts import _corpus_parse as cp

# A trimmed two-entry arXiv Atom feed: one well-formed, one missing its summary
# (which must be skipped). Whitespace in the title/summary mimics arXiv's line
# wrapping so the normalisation is exercised.
SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2406.01234v2</id>
    <published>2024-06-03T17:59:59Z</published>
    <title>A   Study   of
      Routing</title>
    <summary>We explore
      tool routing.</summary>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2406.05678v1</id>
    <published>2024-06-01T00:00:00Z</published>
    <title>No Summary Here</title>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>
"""


def test_parse_feed_extracts_well_formed_entry() -> None:
    records = cp.parse_feed(SAMPLE_FEED)
    assert len(records) == 1  # the summary-less entry is skipped
    rec = records[0]
    assert rec.arxiv_id == "2406.01234"  # version stripped
    assert rec.title == "A Study of Routing"  # whitespace collapsed
    assert rec.abstract == "We explore tool routing."
    assert rec.published_date == date(2024, 6, 3)


def test_parse_feed_orders_categories_primary_first_deduped() -> None:
    rec = cp.parse_feed(SAMPLE_FEED)[0]
    assert rec.categories == ["cs.CL", "cs.LG"]  # primary cs.CL not duplicated


def test_parse_feed_empty() -> None:
    empty = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert cp.parse_feed(empty) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2406.01234v2", "2406.01234"),
        ("http://arxiv.org/abs/2406.01234", "2406.01234"),
        ("2406.01234v15", "2406.01234"),
        ("cs/0501001v1", "cs/0501001"),  # old-style id
    ],
)
def test_strip_version(raw: str, expected: str) -> None:
    assert cp.strip_version(raw) == expected


def test_batched_chunks_with_short_tail() -> None:
    records = cp.parse_feed(SAMPLE_FEED)  # 1 record
    items = records * 5  # 5 records
    chunks = list(cp.batched(items, 2))
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_batched_exact_multiple() -> None:
    records = cp.parse_feed(SAMPLE_FEED) * 4
    chunks = list(cp.batched(records, 2))
    assert [len(c) for c in chunks] == [2, 2]


def test_batched_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="batch size"):
        list(cp.batched([], 0))
