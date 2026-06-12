---
name: write-handoff
description: Draft a session-to-session handoff document from a recent diff. Use when the user asks to write up what changed, prepare a WIP note for a teammate or their next session, or summarise work-in-progress before stepping away. Trigger phrases include "handoff", "WIP note", "write up what I did", "summarise this session".
---

# Skill: write-handoff

A handoff is a short document that lets the next session (you, a teammate,
or Claude in a fresh context) pick up where the current one left off
without re-reading the full diff. Keep it narrow and concrete.

## 1. Gather the material

```bash
# Diff since the last tag (or HEAD~N if no tag fits)
git describe --tags --abbrev=0              # find the baseline tag
git diff <tag>..HEAD --stat                 # what files moved
git log <tag>..HEAD --oneline               # commits included

# What's still open
cat TASKS.md | grep -E '^\s*- \[ \]'        # incomplete task rows
git status                                  # uncommitted / untracked
```

If the session is mid-task (not yet committed), include the uncommitted
diff — that's usually the most important part of the handoff, not the
already-landed commits.

## 2. Draft structure

Write to `HANDOFF.md` at the repo root (or `docs/handoffs/YYYY-MM-DD.md`
if the project keeps a dated history). Sections:

1. **What shipped** — one sentence per merged commit or landed change.
   Link the commit SHA. Skip mechanical changes (formatting, lockfile).
2. **What's in flight** — files touched but not committed, with a
   one-line "intent" per file. This is the section that prevents the
   next session from re-deriving what you were doing.
3. **Next actions** — three-to-five bullets, each concrete enough to
   start immediately. "Fix the bug" is not concrete; "add a
   `LLMContentError` branch to `_extract_text` in `openai_adapter.py`
   when `choices[0].finish_reason == "content_filter"`" is.
4. **Context that won't be obvious from the diff** — decisions made,
   things tried and discarded, conversations with stakeholders,
   external constraints. This is what memory can't capture from code
   alone.
5. **Known gaps / deferred** — anything you noticed but chose not to
   fix, with a one-line reason.

## 3. Tone and length

- Target 40-80 lines. Anything longer and the reader will skim, which
  defeats the purpose. If you need more, split by area or link out.
- Past tense for what happened, imperative for what's next.
- Name files and functions explicitly. "The adapter" is ambiguous;
  `infrastructure/openai_adapter.py::_extract_text` is not.
- Link OT-IDs from VERIFICATION.md and task IDs from TASKS.md when the
  work references them — the reader can trace back.

## 4. Sanity check before handing off

- Would a cold reader know *what to do next* from this doc alone?
- Is every "in flight" item recoverable from `git stash list` or an
  uncommitted diff? If not, flag it.
- Any temporary `# TODO` or `# HACK` markers in the code that the next
  session should know about? List them in the handoff, or the next
  reader won't find them.

---

**Verified by:** [OT-5](../../../VERIFICATION.md#ot-5--claude-code-scaffold-ships-with-the-project) — handoff skill is part of the shipped `.claude/` surface.
