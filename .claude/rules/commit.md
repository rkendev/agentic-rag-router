# Rule: commits & PRs

Commits are the archaeology future-you will rely on. Optimise them for
someone who opens the repo two years from now knowing nothing. The
investment is the moment before `git commit`; the payoff is every time
someone runs `git log`.

## Branch conventions

- `main` is always green. Never push directly. Every change goes through
  a PR even for solo work — the PR is the forcing function for the
  spec-chain update.
- Feature branches: `t<NNN>-<short-slug>` for task-scoped work
  (`t022-claude-dir`), or `fix/<short-slug>` for off-spec fixes.
- One PR per task unless the task is a genuine umbrella. A PR that
  bundles three unrelated changes is a PR that's hard to review and
  impossible to revert cleanly.

## Commit message format

Subject line, blank line, body. Subject in imperative mood (`add`, not
`added`), ≤ 72 chars, no trailing period. Body wraps at ~72 chars and
explains the *why* — the diff already shows the *what*.

```
feat(t022): ship .claude/ scaffold — rules, skills, commands, settings

Forked projects now inherit a working Claude Code environment on day 1:
- CLAUDE.md orientation doc (jinja-substituted)
- 4 rules with OT-ID cross-references to VERIFICATION.md
- 3 skills: add-adapter (with full code template), write-handoff, ship-release
- 3 slash commands: /verify, /handoff, /sdd
- settings.json with safe-default allow/deny lists

Closes OT-5. See DECISIONS.md D14 for scope rationale.
```

Prefix conventions: `feat(tNNN)`, `fix(tNNN)`, `docs(...)`, `test(...)`,
`chore(...)`, `refactor(...)`. The `(tNNN)` scope ties the commit to a
TASKS.md row — grep-friendly a year later.

## The heredoc rule

Commit messages containing backticks (`` ` ``), `$`, `!`, or double
quotes go through `git commit -F - <<'EOF'`, never `git commit -m "..."`.
In a double-quoted `-m "..."`, bash interprets `` `some_code` `` as
command substitution and `$VAR` as variable expansion, which mangles the
message silently.

Correct pattern:

```bash
git commit -F - <<'EOF'
feat(tNNN): subject line goes here

Body can freely use `filename.py`, $VARIABLE references,
"quoted phrases", and !bang — all literal.
EOF
```

The critical bit is `<<'EOF'` with **single quotes** around the `EOF`
delimiter. That disables all shell expansion inside the heredoc. Unquoted
`<<EOF` still expands. For `--amend` with a new message, same pattern:
`git commit --amend -F - <<'EOF' ... EOF`.

## Before opening a PR

Run the full local gate. Don't trust "CI will catch it" — CI catching
it means burning five minutes of a green-build slot for something you
could have caught in ten seconds locally.

```bash
make check                          # ruff + ruff-format + mypy + bandit + tests
make template-test                  # (template repo only) fresh copier render
```

PR description template: link the TASKS.md row, summarise the *why*, list
any VERIFICATION.md or DECISIONS.md rows added or changed. Reviewers —
even if that's just future-you — appreciate the trail.

## After merge

- Squash-merge unless the branch has a genuinely meaningful intermediate
  history. A 12-commit "WIP / fix typo / fix typo / fix typo" branch is
  best squashed; a carefully-structured multi-commit branch that tells a
  story is best rebased-merged.
- Delete the feature branch. Stale branches on `origin` are noise that
  hides the branches that matter.
- If the PR closed an OT, spot-check that the VERIFICATION.md row runs
  green on `main` immediately after merge. Merge green is not the same
  as rebased-on-main green.

## What never to commit

- Secrets. `.env`, API keys, tokens, service account files. The
  `.gitignore` covers the common ones; anything new that might carry
  credentials gets a line added *before* you stage it.
- Generated artefacts that rebuild in seconds. `dist/`, `htmlcov/`,
  `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`,
  `.ruff_cache/` are all ignored — confirm before adding anything new
  to the repo root.
- Large binaries without Git LFS. Once they're in history, they're
  expensive to remove; easier to catch at commit time.

---

**Verified by:** [OT-3](../../VERIFICATION.md#ot-3--pre-commit-versions-match-pyprojecttoml) (pin-parity check prevents the local-green-CI-red drift that undermines commit gates), [OT-6](../../VERIFICATION.md#ot-6--ci-is-green-on-main) (two consecutive green runs on `main` is the acceptance criterion that "commit discipline held").
