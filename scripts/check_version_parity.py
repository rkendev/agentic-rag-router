#!/usr/bin/env python3
"""Enforce ruff/mypy/bandit pin parity across pyproject.toml and .pre-commit-config.yaml.

Drift between these two files silently produces local-green / CI-red incidents
(the StockStream post-mortem — CIMRA D19). This script is run both as the final
pre-commit hook and in CI. See docs/DECISIONS.md D8.

Exit codes:
    0  pins agree across both files.
    1  pin drift detected.
    2  required pin or repo entry is missing from one of the files.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import yaml  # provided transitively by pre-commit; also a dev dep if needed directly

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"

# Pre-commit repo URL → tool name we track.
REPO_TO_TOOL: dict[str, str] = {
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
    "https://github.com/pre-commit/mirrors-mypy": "mypy",
    "https://github.com/PyCQA/bandit": "bandit",
}
TOOLS: tuple[str, ...] = ("ruff", "mypy", "bandit")


def read_pyproject_pins() -> dict[str, str]:
    """Read exact '==X.Y.Z' pins for each tool from [project.optional-dependencies].dev."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_deps: list[str] = data["project"]["optional-dependencies"]["dev"]
    pins: dict[str, str] = {}
    for spec in dev_deps:
        for tool in TOOLS:
            prefix = f"{tool}=="
            if spec.startswith(prefix):
                pins[tool] = spec[len(prefix) :].strip()
    missing = [t for t in TOOLS if t not in pins]
    if missing:
        print(
            f"ERROR: no exact '==X.Y.Z' pin for {missing} in "
            f"pyproject.toml [project.optional-dependencies].dev",
            file=sys.stderr,
        )
        sys.exit(2)
    return pins


def read_precommit_pins() -> dict[str, str]:
    """Read `rev:` for each tracked repo from .pre-commit-config.yaml (strip 'v' prefix)."""
    data = yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    for repo in data.get("repos", []):
        tool = REPO_TO_TOOL.get(repo.get("repo", ""))
        if tool is None:
            continue
        pins[tool] = str(repo.get("rev", "")).lstrip("v")
    missing = [t for t in TOOLS if t not in pins]
    if missing:
        print(
            f"ERROR: missing repo entry for {missing} in .pre-commit-config.yaml",
            file=sys.stderr,
        )
        sys.exit(2)
    return pins


def main() -> int:
    pyproject_pins = read_pyproject_pins()
    precommit_pins = read_precommit_pins()
    drifted = [t for t in TOOLS if pyproject_pins[t] != precommit_pins[t]]

    if not drifted:
        summary = ", ".join(f"{t}={pyproject_pins[t]}" for t in TOOLS)
        print(f"OK pin parity: {summary}")
        return 0

    print("FAIL: ruff/mypy/bandit pin drift between pyproject.toml and .pre-commit-config.yaml")
    print()
    print(f"    {'tool':<8} {'pyproject':<12} pre-commit")
    print(f"    {'-' * 8} {'-' * 12} {'-' * 12}")
    for tool in TOOLS:
        mark = "!" if tool in drifted else " "
        print(f"  {mark} {tool:<8} {pyproject_pins[tool]:<12} {precommit_pins[tool]}")
    print()
    print("Fix: align the two files — update either the `rev:` in .pre-commit-config.yaml")
    print("or the `==X.Y.Z` pin in pyproject.toml [project.optional-dependencies].dev.")
    print("See docs/DECISIONS.md D8.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
