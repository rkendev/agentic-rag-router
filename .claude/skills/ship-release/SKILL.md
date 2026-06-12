---
name: ship-release
description: Cut a release — bump the version, finalise CHANGELOG, tag, push, and open a GitHub release. Use when the user asks to "ship", "release", "cut v0.X.Y", "publish", or "tag and release". Covers semver bump discipline, Keep-a-Changelog conventions, and the tag-push-release sequence.
---

# Skill: ship-release

Releases here follow semver + Keep-a-Changelog. Never skip the gate
checks — `make check` green on the exact commit you tag is the promise
the version number makes.

## 1. Decide the version bump

| Bump | When |
|---|---|
| `MAJOR` | Backwards-incompatible change to a shipped API, port, or rendered-project contract |
| `MINOR` | Added capability, new adapter, new skill, new OT-ID — no breakage |
| `PATCH` | Bug fix, doc fix, dependency bump — behaviour unchanged |

If in doubt, bias toward MINOR. The cost of an over-bump is low; the
cost of hiding a breaking change inside a PATCH is high.

## 2. Pre-flight (do not skip any of these)

```bash
git switch main && git pull --ff-only
make check                          # 219+ tests green on main
git status                          # working tree clean
git log $(git describe --tags --abbrev=0)..HEAD --oneline   # changes since last tag
```

Read the last-tag-to-HEAD log — every line should either already be in
CHANGELOG under `[Unreleased]` or be a merge/format commit that doesn't
need a changelog entry. If something's missing, stop and add it first.

## 3. Finalise CHANGELOG.md

Promote the `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD`:

```markdown
## [Unreleased]

## [0.3.0] - 2026-04-20

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Add a fresh empty `## [Unreleased]` at the top for the next cycle.
Add a link reference at the bottom:
`[0.3.0]: https://github.com/<org>/<repo>/releases/tag/v0.3.0`.

## 4. Bump the version in pyproject.toml

```toml
[project]
version = "0.3.0"
```

If any other file pins the version (`__init__.py::__version__`, docs,
badges, docker tags), bump those in the same commit. Search:
`grep -rn '0\.2\.' --include='*.py' --include='*.toml' --include='*.md'`.

## 5. Commit, tag, push

```bash
git add pyproject.toml CHANGELOG.md
git commit -F - <<'EOF'
chore(release): v0.3.0

- Finalise CHANGELOG
- Bump version to 0.3.0
EOF

git tag -a v0.3.0 -m "v0.3.0"
git push origin main
git push origin v0.3.0
```

Use `-F - <<'EOF'` if the release notes contain backticks, `$`, or `!`
— `-m "..."` gets bash-interpreted and will corrupt them.

## 6. Create the GitHub release

```bash
gh release create v0.3.0 \
  --title "v0.3.0" \
  --notes-file <(sed -n '/^## \[0\.3\.0\]/,/^## \[/p' CHANGELOG.md | sed '$d')
```

The `sed` dance extracts just the section for this version from
CHANGELOG. Inspect before publishing if you're unsure — `--draft` gives
you a dry-run.

## 7. Post-release

- Announce (if the project has a channel for that).
- Close any GitHub milestone / project-board column pegged to this
  version.
- Open an issue or note for the next version's themes if something's
  already queued.

---

**Verified by:** [OT-5](../../../VERIFICATION.md#ot-5--claude-code-scaffold-ships-with-the-project) — ship-release skill is part of the shipped `.claude/` surface.
