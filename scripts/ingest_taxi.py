"""Load one month of NYC TLC yellow-taxi trips into `taxi_trips`.

    python -m scripts.init_db        # once, to create the table
    python -m scripts.ingest_taxi    # load (re-run safe)

Idempotent by construction: the load is `TRUNCATE` + `COPY`, so the table's end
state is identical no matter how many times it runs. The source parquet is
cached under `data/raw/` (gitignored); re-runs reuse it instead of re-fetching.
Source URL + month are recorded in docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from scripts import _db
from scripts._taxi_mapping import TABLE_NAME, copy_columns_sql, parquet_columns, project_batch

# One recent month. ~2.96M rows in 2024-01, comfortably above the 500k floor.
TAXI_MONTH = os.environ.get("TAXI_MONTH", "2024-01")
SOURCE_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{TAXI_MONTH}.parquet"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PARQUET_PATH = RAW_DIR / f"yellow_tripdata_{TAXI_MONTH}.parquet"

# Rows per parquet batch / COPY flush. Big enough to amortise Python overhead,
# small enough to keep memory flat.
BATCH_SIZE = 65_536


def download_if_missing() -> Path:
    """Fetch the month's parquet into data/raw/ unless it is already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 0:
        return PARQUET_PATH
    print(f"downloading {SOURCE_URL}")
    # Trusted, hard-coded CloudFront HTTPS host (NYC TLC public data).
    urllib.request.urlretrieve(SOURCE_URL, PARQUET_PATH)  # nosec B310 - hard-coded HTTPS host
    return PARQUET_PATH


def load(path: Path) -> int:
    """TRUNCATE then COPY every row of the parquet into taxi_trips."""
    parquet = pq.ParquetFile(path)
    names = list(parquet_columns())
    copy_sql = f"COPY {TABLE_NAME} ({copy_columns_sql()}) FROM STDIN"

    loaded = 0
    with _db.connect() as conn, conn.cursor() as cur:
        cur.execute(f"TRUNCATE {TABLE_NAME}")
        with cur.copy(copy_sql) as copy:
            for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=names):
                columns = {name: batch.column(name).to_pylist() for name in names}
                for row in project_batch(columns):
                    copy.write_row(row)
                loaded += batch.num_rows
        conn.commit()
    return loaded


def main() -> None:
    path = download_if_missing()
    rows = load(path)
    print(f"loaded {rows} rows into {TABLE_NAME} from {path.name}")


if __name__ == "__main__":
    main()
