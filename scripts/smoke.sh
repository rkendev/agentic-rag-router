#!/usr/bin/env bash
# agentic-rag-router — docker compose smoke test
#
# Brings up the default compose stack (Ollama only), waits for its
# healthcheck to clear, hits the /api/tags endpoint to prove the HTTP
# server is serving, then tears everything down. Exits non-zero with a
# clear message at whichever step fails — callers (CI, Makefile, or a
# local developer) can use the exit code directly without parsing output.
#
# Usage:    ./scripts/smoke.sh
# Requires: docker + docker compose v2 (the `docker compose` subcommand,
#           not the legacy `docker-compose` binary).

set -euo pipefail

# Resolve to the repo root so the script works from any CWD — someone
# running `bash scripts/smoke.sh` from a subdirectory should get the
# same behaviour as running it from the repo root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

cleanup() {
  # Always tear the stack down, even on healthcheck failure, so
  # repeated local runs don't leave orphan containers blocking :11434.
  # --volumes drops the ollama volume too — keeps the smoke run
  # idempotent at the cost of re-downloading any pulled models on the
  # next run (none are pulled here, so zero actual cost).
  docker compose down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "→ docker compose up -d --wait ollama"
# --wait blocks until every service's healthcheck reports healthy or
# times out with non-zero. Compose v2 native; no custom polling loop
# required. The healthcheck in docker-compose.yml does the actual work.
docker compose up -d --wait ollama

echo "→ GET http://localhost:11434/api/tags"
# /api/tags returns {"models":[]} when no models are pulled — that's
# fine. We're proving the HTTP server is alive, not that any specific
# model is loaded. --fail turns 4xx/5xx into a non-zero exit; --max-time
# caps the call at 10s so a hung server doesn't stall CI indefinitely.
if ! curl --fail --silent --show-error --max-time 10 \
     http://localhost:11434/api/tags >/dev/null; then
  echo "✗ Ollama HTTP endpoint did not respond on :11434" >&2
  exit 1
fi

echo "✓ smoke ok: compose up + ollama healthcheck + /api/tags all green"
