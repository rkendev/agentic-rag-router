"""Unit tests for the pure taxi parquet->schema mapping (no DB, no pyarrow)."""

from __future__ import annotations

from scripts import _taxi_mapping as tm

# The frozen EVAL_RUBRIC.md §4 column set, as table identifiers (lowercase).
RUBRIC_COLUMNS = {
    "vendor_id",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "rate_code_id",
    "store_and_fwd_flag",
    "pulocationid",
    "dolocationid",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
}


def test_table_columns_match_rubric_schema() -> None:
    assert set(tm.table_columns()) == RUBRIC_COLUMNS


def test_drifting_parquet_names_are_mapped() -> None:
    # The parquet's CamelCase columns map to the rubric's snake_case identifiers.
    mapping = dict(zip(tm.table_columns(), tm.parquet_columns(), strict=True))
    assert mapping["vendor_id"] == "VendorID"
    assert mapping["rate_code_id"] == "RatecodeID"
    assert mapping["airport_fee"] == "Airport_fee"
    assert mapping["pulocationid"] == "PULocationID"
    assert mapping["dolocationid"] == "DOLocationID"


def test_no_duplicate_columns() -> None:
    assert len(tm.table_columns()) == len(set(tm.table_columns()))
    assert len(tm.parquet_columns()) == len(set(tm.parquet_columns()))


def test_create_table_sql_lists_every_column_in_order() -> None:
    ddl = tm.create_table_sql()
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS taxi_trips")
    positions = [ddl.index(col) for col in tm.table_columns()]
    assert positions == sorted(positions)  # DDL order == canonical order


def test_copy_columns_sql_is_canonical_order() -> None:
    assert tm.copy_columns_sql() == ", ".join(tm.table_columns())


def test_project_row_reads_by_parquet_name_into_canonical_order() -> None:
    record = {pq: i for i, pq in enumerate(tm.parquet_columns())}
    assert tm.project_row(record) == tuple(range(len(tm.parquet_columns())))


def test_project_row_missing_parquet_column_raises() -> None:
    record = {pq: 0 for pq in tm.parquet_columns()}
    del record["VendorID"]
    try:
        tm.project_row(record)
    except KeyError as exc:
        assert "VendorID" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("expected KeyError for a missing parquet column")


def test_project_batch_transposes_columns_to_rows() -> None:
    columns = {pq: [f"{pq}-0", f"{pq}-1"] for pq in tm.parquet_columns()}
    rows = tm.project_batch(columns)
    assert len(rows) == 2
    # Row 0 holds the "-0" value of each parquet column, in canonical order.
    assert rows[0] == tuple(f"{pq}-0" for pq in tm.parquet_columns())
    assert rows[1] == tuple(f"{pq}-1" for pq in tm.parquet_columns())


def test_project_batch_empty() -> None:
    columns: dict[str, list[str]] = {pq: [] for pq in tm.parquet_columns()}
    assert tm.project_batch(columns) == []
