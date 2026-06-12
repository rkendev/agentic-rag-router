"""Idempotent database setup for the D2 data layer.

Run once (re-run safe) against the docker-compose Postgres before ingesting:

    python -m scripts.init_db

It enables the `vector` extension, creates the `taxi_trips` and `corpus_docs`
tables (with an HNSW cosine index), and provisions the SELECT-only `router_ro`
role the D3 SQL tool will connect as. Every statement uses `IF NOT EXISTS` or
its role/grant equivalent, so repeated runs converge without error.
"""

from __future__ import annotations

from typing import Any

from psycopg import Cursor, sql

from scripts import _db
from scripts._taxi_mapping import create_table_sql

# Embedding dimensionality for sentence-transformers/all-MiniLM-L6-v2 (pinned in
# ingest_corpus.py). Kept here too so the DDL and the model never drift apart.
EMBEDDING_DIM = 384

CORPUS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS corpus_docs (
    arxiv_id       text PRIMARY KEY,
    title          text NOT NULL,
    abstract       text NOT NULL,
    categories     text[] NOT NULL,
    published_date date NOT NULL,
    embedding      vector({EMBEDDING_DIM}) NOT NULL
)
"""

# HNSW with cosine ops: the vector tool ranks by cosine distance (`<=>`).
CORPUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS corpus_docs_embedding_hnsw
    ON corpus_docs USING hnsw (embedding vector_cosine_ops)
"""


def _ensure_role(cur: Cursor[Any]) -> None:
    """Create router_ro if absent and (re)assert its SELECT-only grants."""
    role, password = _db.router_ro_credentials()
    ident = sql.Identifier(role)
    # CREATE ROLE has no IF NOT EXISTS; guard with a catalog check via DO block.
    cur.execute(
        sql.SQL(
            "DO $$ BEGIN "
            "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {role_lit}) THEN "
            "CREATE ROLE {role_ident} LOGIN PASSWORD {pw_lit}; "
            "END IF; END $$;"
        ).format(
            role_lit=sql.Literal(role),
            role_ident=ident,
            pw_lit=sql.Literal(password),
        )
    )
    # Idempotent privilege floor: connect + read taxi_trips, nothing more.
    cur.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {db} TO {role}").format(
            db=sql.Identifier(_db.conn_params()["dbname"]), role=ident
        )
    )
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {role}").format(role=ident))
    cur.execute(sql.SQL("GRANT SELECT ON taxi_trips TO {role}").format(role=ident))


def main() -> None:
    with _db.connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(create_table_sql())
        cur.execute(CORPUS_TABLE_SQL)
        cur.execute(CORPUS_INDEX_SQL)
        _ensure_role(cur)
        conn.commit()

        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    version = row[0] if row else "MISSING"
    role, _ = _db.router_ro_credentials()
    print(f"vector extension: {version}")
    print(f"tables ready: taxi_trips, corpus_docs (embedding dim {EMBEDDING_DIM})")
    print(f"role ready: {role} (SELECT on taxi_trips)")


if __name__ == "__main__":
    main()
