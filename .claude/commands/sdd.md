---
description: Draft the top of the spec chain — charter, specification, technical plan, and task breakdown — from a one-liner project description. Use when bootstrapping a new project or pivoting an existing one.
---

# /sdd

Draft the top of the spec chain from a one-liner. The output is a
starting point, not a finished chain — the user will edit the drafts
into the real thing. Your job is to give them *structured prose to
strike through*, not a blank page.

## Input

A single sentence describing the project. If the user ran `/sdd` with
arguments, treat `$ARGUMENTS` as the one-liner. If they ran it bare,
ask for one before producing anything.

Examples of usable one-liners:

- "Real-time stock price dashboard that pulls Yahoo Finance and streams to a React UI."
- "CLI that turns a git repo into a lecture-style walkthrough with chapter markers."
- "GitHub Action that runs an LLM-backed code reviewer on every PR."

Each names a *what*, hints at a *why*, and drops a *technology
constraint*. If the user's sentence is missing one of those three, ask
one clarifying question — don't generate from a sentence you can't
constrain.

## What to produce

Four drafts at the repo root, in this order (each informs the next):

1. **`PROJECT_CHARTER.md`** — 30-60 lines. Four sections:
   - *Problem* — the thing the world doesn't have yet
   - *Users* — who benefits, with a rough persona
   - *Non-goals* — what this is *not* (keeps scope honest)
   - *Success criteria* — how we'll know it worked (3-5 bullets)
2. **`SPECIFICATION.md`** — 80-150 lines.
   - *OT-1..OT-N observable truths* — each OT is a claim that can be
     checked by a runnable command. Be specific: "`tests/` contains
     three subdirs named `unit/`, `contract/`, `integration/`" beats
     "tests are organized."
   - *Acceptance criteria* — the OT list *is* the acceptance
   - *Out of scope* — mirror Non-goals from the charter
3. **`TECHNICAL_PLAN.md`** — 60-120 lines.
   - *Architecture* — layers, modules, dependency rule (use DDD-lite
     or hexagonal if the charter implies a clean core/edge separation)
   - *Dependencies* — third-party libs/services, each with a one-line
     reason; version pins deferred to `T002` pyproject work
   - *Risks* — 3-5 *real* risks with mitigations, not boilerplate
   - *Milestones* — M1, M2, … — each mapped to a subset of OTs
4. **`TASKS.md`** — 50-100 lines.
   - *T000..T0NN* atomic work items. Each row: deliverable, done signal
     (usually a command or file list), OT-anchor (`→ OT-3`)
   - Sequence so each T depends only on earlier Ts
   - First few Ts are scaffolding (pyproject, pre-commit, folder layout);
     last few are docs + release (`CHANGELOG`, `README`, `v0.1.0` tag)

Cross-link: every T in `TASKS.md` references its OT in
`SPECIFICATION.md`; every OT references the charter success criterion
it satisfies.

## What not to produce (yet)

- `docs/DECISIONS.md` entries. ADRs earn their place when a decision
  was contested; a fresh draft has no contested decisions.
- `VERIFICATION.md` rows with commands. The OT list lives in the spec;
  the runnable commands come later when the implementation exists.
- `CHANGELOG.md` entries. The first real entry is the initial scaffold
  commit, not the draft.

Empty stubs for those files are fine. Rich pre-implementation content
is premature and goes stale fast.

## Discipline

- **Absolute dates only.** "Next week" in a task is dead weight in six
  months. Use `YYYY-MM-DD`.
- **Mark assumptions inline.** If the one-liner doesn't specify
  deployment target, write `ASSUMED: deploys as a single container —
  confirm before T005` right in the plan. Never silently decide for
  the user.
- **Prefer few meaty tasks over many thin ones.** 12-18 well-scoped Ts
  beat 40 micro-tasks. Each T should be a full PR's worth of work.
- **Don't invent stakeholders.** If the one-liner says nothing about
  who this is for, ask — don't fabricate a persona.
- **Don't overwrite silently.** If any of the four files already
  exists, stop and ask: update-in-place, append, or skip?

## Quality bar

Before handing back, a cold reader should be able to:

1. Say what this project is in 10 seconds after reading the charter.
2. Name the first three tasks without reading the spec.
3. Name one risk that isn't "we might not finish."

If any of those fail, the draft isn't ready — tighten and try again.

## After producing the drafts

Tell the user to:

- Read each file top-to-bottom and strike what's wrong.
- Commit the four files in one commit:
  `chore(scaffold): initial spec chain from /sdd`.
- Open a PR against `main` even for solo work — the PR is the forcing
  function for chain reviews (see `rules/commit.md`).

The chain is a living document. What `/sdd` drafts on day 1 should
look very different by v0.1.0 — that's the point.
