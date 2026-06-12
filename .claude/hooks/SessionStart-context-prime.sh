#!/usr/bin/env bash
# SessionStart hook — primes Claude Code's context at session start with
# shipped state. Echoes recent commits, branch + working-tree status, and
# latest tag so Claude can answer "where was I" without grep.
#
# Council plan 2026-05-01 §7 item 2 (saves reorientation cost per session).
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

echo "─── Branch + recent commits ───"
git log --oneline -3 2>&1 | head -3
echo ""
echo "─── Local status ───"
git status -sb 2>&1 | head -10
echo ""
echo "─── Latest tag ───"
git describe --tags --abbrev=0 2>/dev/null || echo "(no tags)"
