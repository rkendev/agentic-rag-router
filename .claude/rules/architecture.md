# Rule: architecture

Strict hexagonal / three-layer dependency direction:

```
infrastructure/  →  application/  →  domain/
```

Read the arrows as "imports from". `domain/` is the stable core and imports
nothing outside stdlib + pydantic. `application/` imports from `domain/`
only. `infrastructure/` imports from both. `main.py` is the only
composition root — the single place where the object graph is wired.

## What lives where

- **`domain/`** — frozen Pydantic models, StrEnums, exception hierarchy. No
  HTTP, no SDKs, no filesystem, no env reads, no threads.
- **`application/`** — ports (`Protocol` classes) and orchestrators
  (`FallbackModel`, policy functions). Ports define *behaviour contracts*
  for things the application needs; adapters in `infrastructure/` satisfy
  them. Orchestrators use only ports + domain types — never a concrete
  adapter class.
- **`infrastructure/`** — SDK adapters (one file per vendor), `Settings`
  (pydantic-settings). Each adapter implements exactly one port, raises
  domain errors from `domain/exceptions.py`, and returns domain types.
- **`main.py`** — builds the graph: reads `Settings`, constructs adapters,
  assembles the orchestrator, returns the application-level entry point.
  Nothing else should ever `import` from `main.py`.

## When you add a new capability

1. Name the behaviour you need. Does it fit an existing port? If yes, add
   a new adapter in `infrastructure/`. If no, add a new port in
   `application/ports.py` first, then the adapter.
2. Errors the adapter can raise go in `domain/exceptions.py` — the
   existing `LLMError` hierarchy is the pattern to match. Never leak an
   SDK exception past the adapter boundary.
3. If the adapter needs config (API key, model name, endpoint), add
   `Settings` fields in `infrastructure/settings.py` with an env prefix
   that doesn't collide with existing ones.
4. Wire the new instance in `main.py` — never anywhere else.

## When you catch yourself breaking the rule

If an adapter needs to reach into another adapter, that's a smell. The
normal fix is to extract the shared concept as a new port in
`application/` that both adapters depend on. If a domain type needs to
"know" about an SDK object (timestamps, token counts, etc.), translate at
the adapter boundary — the domain type stays clean.

If the dependency really is unavoidable, document it in `docs/DECISIONS.md`
as an ADR with the failure mode that forced the compromise. The rule
bends when it needs to; the ADR is how we don't forget.

## Enforcement

Two lines of defence:

- **Imports at the top of every source file.** `ruff` (pre-commit + CI)
  flags obvious cross-layer imports via `isort` + unused-import checks.
  Code review catches anything ruff misses.
- **Test layout.** `tests/unit/domain/` has no `from agentic_rag_router.infrastructure` line in it,
  and never should. `tests/unit/application/` only imports domain types
  and ports. Contract tests (`tests/contract/`) are allowed to import
  adapters — that's literally their job.

---

**Verified by:** [OT-2](../../VERIFICATION.md#ot-2--every-adapter-conforms-to-llmport) — the parametrized contract suite proves every adapter implements `LLMPort` honestly; a broken layer boundary surfaces as a vendor-tagged contract failure rather than a subtle shape mismatch.
