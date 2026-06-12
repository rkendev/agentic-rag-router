"""Evaluation harness --- deterministic scoring over the frozen golden set (D6).

`scoring` holds the pure metric functions (no API, no DB, no LLM-as-judge): they
take rows derived from `router.loop.RouterResponse` fields and the frozen goldens
and return the routing / refusal / citation metrics, the per-class confusion
table, and the naive single-tool baselines. The live runner
(`scripts/run_eval.py`) calls these and commits the report; the CI gate
(`tests/test_eval_gates.py`) asserts against the committed report.
"""

from __future__ import annotations
