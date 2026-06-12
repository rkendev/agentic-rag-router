"""Freeze guard for the T001 evaluation set.

Pins the sha256 of the golden question set and the evaluation rubric, and
validates the JSONL schema. Per ``docs/EVAL_RUBRIC.md`` section 7, both files
are immutable after merge: changing them means adding a new versioned file plus
a logged ``CHANGELOG.md`` decision, never editing the frozen originals. A failing
hash assertion here means the eval set was modified in place -- revert it or
follow the freeze policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"
RUBRIC_PATH = REPO_ROOT / "docs" / "EVAL_RUBRIC.md"

# Recorded over the pre-commit-normalized files (LF endings, trailing newline).
# Regenerate with `sha256sum` only when intentionally re-freezing per the policy.
GOLDEN_SHA256 = "0b4c5e12092fbe2001f0afd50ed693e5da0cd84d35f0ba29c5330459f1c2cadd"
RUBRIC_SHA256 = "62205514c6e33d7a0f98d9c5a8be4d7e35ce651fc3f34cd92110c298014c9380"

EXPECTED_COUNT = 60
MIN_ADVERSARIAL = 6
EXPECTED_KEYS = {"id", "question", "label", "acceptable_tools", "adversarial", "rationale"}
LABELS = frozenset({"vector_only", "sql_only", "web_only", "no_answer", "hybrid"})
TOOLS = frozenset({"vector_search", "sql_query", "web_search"})
SINGLE_TOOL = {
    "vector_only": "vector_search",
    "sql_only": "sql_query",
    "web_only": "web_search",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows() -> list[dict[str, Any]]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_golden_set_hash_frozen() -> None:
    assert (
        _sha256(GOLDEN_PATH) == GOLDEN_SHA256
    ), "golden_questions.jsonl changed; see docs/EVAL_RUBRIC.md section 7 freeze policy"


def test_rubric_hash_frozen() -> None:
    assert (
        _sha256(RUBRIC_PATH) == RUBRIC_SHA256
    ), "EVAL_RUBRIC.md changed; see its section 7 freeze policy"


def test_line_count() -> None:
    assert len(_load_rows()) == EXPECTED_COUNT


def test_schema_per_line() -> None:
    for i, row in enumerate(_load_rows(), start=1):
        assert set(row) == EXPECTED_KEYS, f"row {i} keys: {sorted(row)}"
        assert row["id"] == f"G{i:03d}"
        assert isinstance(row["question"], str) and row["question"].strip()
        assert row["label"] in LABELS
        assert isinstance(row["acceptable_tools"], list)
        assert set(row["acceptable_tools"]) <= TOOLS
        assert isinstance(row["adversarial"], bool)
        assert isinstance(row["rationale"], str) and row["rationale"].strip()


def test_label_tools_consistency() -> None:
    for row in _load_rows():
        label = row["label"]
        tools = row["acceptable_tools"]
        if label == "no_answer":
            assert tools == [], f"{row['id']}: no_answer must have empty acceptable_tools"
        elif label == "hybrid":
            assert len(tools) >= 2, f"{row['id']}: hybrid must list >= 2 acceptable_tools"
        else:
            assert tools == [SINGLE_TOOL[label]], f"{row['id']}: wrong tool for {label}"


def test_adversarial_count_and_placement() -> None:
    adversarial = [row for row in _load_rows() if row["adversarial"]]
    assert len(adversarial) >= MIN_ADVERSARIAL
    # The adversarial flag marks refusal-gate near-misses, which are all no_answer.
    assert all(row["label"] == "no_answer" for row in adversarial)
