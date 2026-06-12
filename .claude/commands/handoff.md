---
description: Generate a session handoff document from `git diff` against the last tag, so the next session can pick up without re-reading everything.
---

# /handoff

Produce a handoff document summarising what happened in this working
session. Follow the `write-handoff` skill — invoke it, don't re-derive
the structure.

## Inputs to collect before writing

```bash
git describe --tags --abbrev=0        # baseline tag
git diff <tag>..HEAD --stat           # what files moved
git log <tag>..HEAD --oneline         # commit list
git status                            # uncommitted state
```

Also read (if present):

- `TASKS.md` — any rows that flipped from `[ ]` to `[x]` this session
- `CHANGELOG.md` `[Unreleased]` — what's queued for the next release
- `VERIFICATION.md` — any OT-ID rows the work touched

## Output location

Write to `HANDOFF.md` at the repo root by default. If the project keeps
a dated history (look for a `docs/handoffs/` directory), write to
`docs/handoffs/YYYY-MM-DD.md` instead — use today's actual date, not a
placeholder.

Ask the user before overwriting an existing `HANDOFF.md`; they may want
to keep both side-by-side until they've read the new one.

## What to cover

Follow the five-section structure from the `write-handoff` skill:

1. What shipped
2. What's in flight
3. Next actions
4. Context that won't be obvious from the diff
5. Known gaps / deferred

## Length target

40-80 lines. If the session was long and you're tempted to go longer,
split by area (e.g., "Backend" / "Frontend" / "Infra") or link out to
per-area notes rather than padding one file.

## Sanity check before finishing

- A cold reader should know *what to do next* from this doc alone.
- Every "in flight" item should be recoverable from git (stash, branch,
  or uncommitted working tree). If not, say so explicitly.
- Link concrete file paths and function names, not "the thing we were
  working on".
