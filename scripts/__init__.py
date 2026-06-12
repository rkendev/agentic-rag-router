"""Operational scripts for the data layer (D2).

This package holds the ingestion and database-setup scripts plus their pure,
unit-testable helpers (`_db`, `_taxi_mapping`, `_corpus_parse`). The heavy
`init_db` / `ingest_taxi` / `ingest_corpus` modules are run as
``python -m scripts.<name>``; the underscore-prefixed helpers carry the pure
logic that `tests/unit/data/` exercises without a database or heavy deps.
"""
