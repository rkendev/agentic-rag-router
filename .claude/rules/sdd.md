# Rule: SDD discipline (spec-driven development)

Every meaningful change flows through the spec chain before it reaches
`src/`. The chain is the difference between a codebase that can be
maintained by someone who wasn't there when it was written, and a
codebase that can't.

## The chain

```
PROJECT_CHARTER.md   → why does this project exist? (problem, users, non-goals)
SPECIFICATION.md     → what does "done" look like? (OT-1..OT-N observable truths)
TECHNICAL_PLAN.md    → how will it be shaped? (layers, dependencies, risks)
TASKS.md             → what discrete work items ship it? (T000..T0NN)
docs/DECISIONS.md    → ADR log — every non-obvious choice, dated, linked
VERIFICATION.md      → every claim in the spec → a runnable command
CHANGELOG.md         → Keep-a-Changelog, one entry per release
```

The chain reads top-to-bottom: the charter constrains the spec, the spec
constrains the plan, the plan constrains the tasks. When code diverges
from the chain, either the chain is stale (update it) or the code is
wrong (fix it). Silent divergence is the rot.

## Rules for working in the chain

1. **Add the row to VERIFICATION.md *before* shipping the claim.** Every
   observable truth (`OT-N`) the spec makes about itself earns a runnable
   command in VERIFICATION.md. If you can't write the command, the claim
   is too vague — sharpen the claim.
2. **Write the DECISION before you hit a dead-end twice.** First time you
   almost-go-down-a-path-and-back-out, write the ADR (`docs/DECISIONS.md`,
   `DN - YYYY-MM-DD - Title`). The ADR is cheap; re-learning the lesson
   six months later isn't.
3. **TASKS.md entries are atomic and verifiable.** Each `T0NN` should have
   a single deliverable, a clear "done" signal (usually a command or a
   file listing), and a link to the acceptance criterion it satisfies.
   If a task needs more than a day of focused work, split it.
4. **Convert relative dates to absolute.** "Next Thursday" in a TASKS
   entry means nothing six months later. Use `YYYY-MM-DD`.
5. **Update CHANGELOG.md in the same commit that ships the change.**
   Keep-a-Changelog 1.1.0 format. `[Unreleased]` is fine during
   development; the release commit promotes it.

## How to use `/sdd`

`.claude/commands/sdd.md` is a full-featured slash command that takes a
one-liner project description and drafts charter prose, a spec skeleton,
a plan outline, and a task breakdown. It is *not* a replacement for
thinking — the draft is a starting point that you refine. But it
captures the structure so you're editing prose instead of staring at a
blank file.

Treat its output the way you'd treat a careful junior's first draft:
the bones are right, the muscle needs work. Read it, strike what's
wrong, sharpen the claims, then commit.

## The feedback loop

- When you finish a task, update the VERIFICATION row if the command
  shifted. Update the CHANGELOG.
- When a test reveals a misstated spec, update SPECIFICATION.md first,
  then the test.
- When a design forces an unexpected choice, write the ADR the same day.
  "I'll write it next week" never holds.

## Red flags

- A `TODO: spec this later` comment in `src/` is a signal the chain is
  behind. Cost of catching up is only going up.
- A `[Unreleased]` CHANGELOG entry that's been there across three
  releases means someone keeps forgetting to promote it — and it's
  stopped meaning anything.
- A VERIFICATION row whose command hasn't run in two months is
  either obsolete (remove it) or quietly broken (fix it). Either way
  it's not doing its job.

---

**Verified by:** [OT-1](../../VERIFICATION.md#ot-1--template-forks-cleanly) (the template itself forks into a working scaffolded project, proving the spec chain round-trips from template author → consumer cleanly), [OT-5](../../VERIFICATION.md#ot-5--claude-code-scaffold-ships-with-the-project) (the `.claude/` scaffold — including this very rule — ships with every fork; bloat in the agent brief is a forcing function against drift elsewhere).
