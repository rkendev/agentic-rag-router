"""pytest-recording (VCR) configuration for the web_search cassette tests.

The Tavily API key travels in the ``Authorization: Bearer`` header, so a single
``filter_headers`` rule keeps it out of the recorded cassette; ``api_key`` is
filtered from any request body as defence in depth. The record mode is left to
pytest-recording's default (``none`` --- replay only, so CI never reaches the
network); re-record deliberately with ``--record-mode=once`` and a real key when
the request shape changes.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    return {
        "filter_headers": [("authorization", "DUMMY")],
        "filter_post_data_parameters": [("api_key", "DUMMY")],
    }
