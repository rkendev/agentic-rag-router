"""Pure schema + column mapping for the `taxi_trips` SQL substrate.

This module is the single source of truth that keeps three things in lockstep:
the `CREATE TABLE` DDL (`init_db`), the `COPY` projection (`ingest_taxi`), and
the unit tests. It imports only the standard library so it loads without the
`ingest` dependency group.

The table binds the **frozen** EVAL_RUBRIC.md §4 yellow-taxi schema. Rubric
column names are used verbatim where they are already valid lowercase
identifiers (`vendor_id`, `rate_code_id`, ...). The rubric's mixed-case
`PULocationID` / `DOLocationID` are stored as the lowercase `pulocationid` /
`dolocationid`; because Postgres folds unquoted identifiers to lowercase, SQL
written with the rubric's casing still resolves. This table *is* the mapping
layer that absorbs the parquet's naming drift (`VendorID`, `RatecodeID`,
`Airport_fee`); the rubric never moves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Column:
    """One taxi_trips column: its table name, parquet source, and SQL type."""

    table: str  # identifier in the taxi_trips table (lowercase)
    parquet: str  # exact column name in the NYC TLC parquet
    sql_type: str  # Postgres type for the DDL


TABLE_NAME = "taxi_trips"

# Order matters: it defines both the DDL column order and the COPY tuple order.
# Every column below maps to a real parquet column verified against the
# 2024-01 yellow-taxi file. All are nullable — TLC data carries nulls in
# passenger_count, rate_code_id, congestion_surcharge, airport_fee, etc.
COLUMNS: tuple[Column, ...] = (
    Column("vendor_id", "VendorID", "integer"),
    Column("tpep_pickup_datetime", "tpep_pickup_datetime", "timestamp"),
    Column("tpep_dropoff_datetime", "tpep_dropoff_datetime", "timestamp"),
    Column("passenger_count", "passenger_count", "integer"),
    Column("trip_distance", "trip_distance", "double precision"),
    Column("rate_code_id", "RatecodeID", "integer"),
    Column("store_and_fwd_flag", "store_and_fwd_flag", "char(1)"),
    Column("pulocationid", "PULocationID", "integer"),
    Column("dolocationid", "DOLocationID", "integer"),
    Column("payment_type", "payment_type", "integer"),
    Column("fare_amount", "fare_amount", "double precision"),
    Column("extra", "extra", "double precision"),
    Column("mta_tax", "mta_tax", "double precision"),
    Column("tip_amount", "tip_amount", "double precision"),
    Column("tolls_amount", "tolls_amount", "double precision"),
    Column("improvement_surcharge", "improvement_surcharge", "double precision"),
    Column("total_amount", "total_amount", "double precision"),
    Column("congestion_surcharge", "congestion_surcharge", "double precision"),
    Column("airport_fee", "Airport_fee", "double precision"),
)


def table_columns() -> tuple[str, ...]:
    """Table column names in canonical (DDL / COPY) order."""
    return tuple(c.table for c in COLUMNS)


def parquet_columns() -> tuple[str, ...]:
    """Parquet source column names, in the same order as `table_columns()`."""
    return tuple(c.parquet for c in COLUMNS)


def create_table_sql() -> str:
    """`CREATE TABLE IF NOT EXISTS taxi_trips (...)` for the rubric §4 schema."""
    cols = ",\n".join(f"    {c.table} {c.sql_type}" for c in COLUMNS)
    return f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (\n{cols}\n)"


def copy_columns_sql() -> str:
    """Comma-joined table column list for a `COPY taxi_trips (...)` statement."""
    return ", ".join(c.table for c in COLUMNS)


def project_row(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Project a parquet row (keyed by parquet column name) into COPY order.

    Raises `KeyError` if a required parquet column is missing — that is the
    signal that the live file diverged from the rubric and the load must stop.
    """
    return tuple(record[c.parquet] for c in COLUMNS)


def project_batch(columns: Mapping[str, Sequence[Any]]) -> list[tuple[Any, ...]]:
    """Project a columnar batch (parquet name -> column values) into row tuples.

    `ingest_taxi` passes pyarrow columns already materialised as Python lists;
    keeping the transposition here makes it unit-testable without pyarrow.
    """
    names = parquet_columns()
    length = len(columns[names[0]]) if names else 0
    return [tuple(columns[name][i] for name in names) for i in range(length)]
