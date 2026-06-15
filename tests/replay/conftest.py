"""pytest-recording (VCR) configuration for the offline router replay demo.

These cassettes record the Anthropic Messages-API turns of a full ``run_router``
call so the two README examples replay with no API key and no network. The
substrates are in-memory port fakes (see ``test_router_replay.py``), so the only
HTTP traffic to record is the Sonnet calls; SQL and vector never leave the
process.

Two safety rules:

* ``x-api-key`` is scrubbed from every recorded request so the committed
  cassette carries no credential. ``authorization`` is filtered too as defence
  in depth (Tavily is not exercised here, but the rule is cheap).
* The record mode is left at pytest-recording's default (``none`` --- replay
  only), so a normal test run never reaches the network. Re-record deliberately
  with ``--record-mode=once`` and a real key when the prompt or loop changes.

An autouse fixture supplies a dummy ``ANTHROPIC_API_KEY`` when one is absent, so
replay needs no environment at all (the SDK validates a key string at
construction time, before VCR intercepts the call).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    return {
        "filter_headers": [("x-api-key", "DUMMY"), ("authorization", "DUMMY")],
        "filter_post_data_parameters": [("api_key", "DUMMY")],
    }


@pytest.fixture(autouse=True)
def _dummy_key_for_replay() -> None:
    """Supply a dummy key in replay only, never during a recording run.

    In replay (the default) the SDK still validates that *some* key string is
    present at construction, so a dummy lets the tests run with no real
    credential. During a recording run (``RUN_LIVE=1``) we must NOT inject a
    dummy: the real key has to come from the environment, and masking it would
    send an invalid key and fail with a confusing 401.
    """
    if os.environ.get("RUN_LIVE") == "1":
        return
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-replay-key")
