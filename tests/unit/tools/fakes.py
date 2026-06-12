"""Honest test doubles for the tool adapters.

Two flavours:

* **Port fakes** (`FakeEmbedder`, `FakeVectorRepository`, `FakeSqlExecutor`) ---
  in-memory implementations of the tool ports, used to exercise the
  orchestration functions without any substrate.
* **Module fakes** (`install_fake_psycopg`, `install_fake_sentence_transformers`)
  --- minimal stand-ins injected into ``sys.modules`` so the *real* lazy-import
  adapters (`PgVectorRepository`, `RouterRoExecutor`,
  `SentenceTransformerEmbedder`) run end-to-end with no `ingest`-group
  dependency installed. This is what lets the heavy code paths reach 100%
  line coverage in CI, which omits psycopg / sentence-transformers entirely.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

# ---------------------------------------------------------------------------
# Port fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """`EmbedderPort` that returns a fixed, non-zero vector and records calls."""

    def __init__(self, dim: int = 384, fill: float = 0.1) -> None:
        self.dim = dim
        self._fill = fill
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [self._fill] * self.dim


class FakeVectorRepository:
    """`VectorRepository` returning preset rows; records the embedding and k."""

    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._rows = rows if rows is not None else []
        self._error = error
        self.last_embedding: list[float] | None = None
        self.last_k: int | None = None

    def top_k(self, embedding: list[float], k: int) -> list[dict[str, object]]:
        self.last_embedding = embedding
        self.last_k = k
        if self._error is not None:
            raise self._error
        return self._rows[:k]


class FakeSqlExecutor:
    """`SqlExecutor` returning preset rows or raising; records the SQL it ran."""

    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._rows = rows if rows is not None else []
        self._error = error
        self.executed: list[str] = []

    def execute(self, sql: str) -> list[dict[str, object]]:
        self.executed.append(sql)
        if self._error is not None:
            raise self._error
        return self._rows


# ---------------------------------------------------------------------------
# Module fakes for the real lazy-import adapters
# ---------------------------------------------------------------------------


class FakeCursor:
    """Stand-in for a psycopg cursor used as a context manager."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((sql, params))

    def fetchmany(self, size: int) -> list[dict[str, object]]:
        return self._rows[:size]

    def fetchall(self) -> list[dict[str, object]]:
        return list(self._rows)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeConnection:
    """Stand-in for a psycopg connection used as a context manager."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_obj = FakeCursor(rows)
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def rollback(self) -> None:
        self.rolled_back = True

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeConnectRecord:
    """Captures what the real adapter passed to `psycopg.connect`."""

    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.conninfo: str | None = None

    @property
    def cursor(self) -> FakeCursor:
        return self.connection.cursor_obj


def install_fake_psycopg(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> FakeConnectRecord:
    """Inject a fake `psycopg` (+ `.conninfo`, `.rows`) returning `rows`.

    Returns a record exposing the built conninfo string and the cursor, so a
    test can assert the adapter set the statement-timeout option, bound the
    right query parameters, and rolled the transaction back.
    """
    connection = FakeConnection(rows)
    record = FakeConnectRecord(connection)

    def make_conninfo(conninfo: str = "", **kwargs: Any) -> str:
        parts = [conninfo] if conninfo else []
        parts += [f"{key}={value}" for key, value in kwargs.items()]
        built = " ".join(parts)
        record.conninfo = built
        return built

    def connect(conninfo: str, **kwargs: Any) -> FakeConnection:
        return connection

    psycopg_mod = types.ModuleType("psycopg")
    conninfo_mod = types.ModuleType("psycopg.conninfo")
    rows_mod = types.ModuleType("psycopg.rows")
    # Populate via __dict__ so mypy permits the dynamic attributes (a plain
    # `mod.attr = ...` trips `attr-defined` on ModuleType) and ruff leaves it be
    # (`setattr` would be rewritten to the same flagged assignment by B010).
    psycopg_mod.__dict__["connect"] = connect
    psycopg_mod.__dict__["conninfo"] = conninfo_mod
    psycopg_mod.__dict__["rows"] = rows_mod
    conninfo_mod.__dict__["make_conninfo"] = make_conninfo
    rows_mod.__dict__["dict_row"] = object()

    monkeypatch.setitem(sys.modules, "psycopg", psycopg_mod)
    monkeypatch.setitem(sys.modules, "psycopg.conninfo", conninfo_mod)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows_mod)
    return record


class FakeSentenceTransformer:
    """Stand-in model: records construction args, returns 384-dim rows."""

    instances: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(self, model_name: str, revision: str | None = None, dim: int = 384) -> None:
        self.model_name = model_name
        self.revision = revision
        self._dim = dim
        self.encode_calls = 0
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], normalize_embeddings: bool = False) -> list[list[float]]:
        self.encode_calls += 1
        return [[0.05] * self._dim for _ in texts]


def install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake `sentence_transformers` exposing `SentenceTransformer`."""
    FakeSentenceTransformer.instances.clear()
    module = types.ModuleType("sentence_transformers")
    module.__dict__["SentenceTransformer"] = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
