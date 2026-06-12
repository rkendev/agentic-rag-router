---
name: add-adapter
description: Scaffold a new LLMPort implementation by copying `_adapter_template.py.jinja`, renaming placeholders, and wiring it into Settings, main.py, the contract suite, and CHANGELOG. Use when the user asks to add a new LLM backend (Groq, Gemini, Bedrock, Cohere, Mistral, etc.).
---

# Skill: add-adapter

When the user asks to add a new LLM backend, walk them through these
steps in order. Do not skip step 5 (contract enrolment) — it's what
turns N adapters into 8N parametrized test cases, and is the single
most common rot-vector if missed.

## 1. Confirm the port contract is unchanged

Read [`.claude/rules/architecture.md`](../../rules/architecture.md) and
[`src/agentic_rag_router/application/ports.py`](../../../src/agentic_rag_router/application/ports.py).
The new adapter lives in `infrastructure/` and implements `LLMPort` —
nothing in `application/` or `domain/` changes. If the user's request
implies a different port (streaming, function-calling, vision), stop
and propose a new port first via `/sdd`. Don't stretch `LLMPort`.

## 2. Copy and rename the adapter template

```bash
# In the generated project root:
cp .claude/skills/add-adapter/_adapter_template.py \
   src/agentic_rag_router/infrastructure/<vendor>_adapter.py
```

The template is pre-annotated. Search-and-replace these placeholders
(case matters — `Vendor` is the class-name form, `vendor` is the
module / env / log form):

| Placeholder | Replace with | Example |
|---|---|---|
| `Vendor` | PascalCase vendor name | `Groq` |
| `vendor` | lowercase vendor name | `groq` |
| `VENDOR` | UPPERCASE vendor name | `GROQ` |
| `<sdk-import>` | the vendor's SDK import line | `import groq` |
| `<sdk-error-classes>` | three-to-five real exception classes from the SDK | `groq.RateLimitError`, etc. |
| `<sdk-call>` | the chat / generate method path on the SDK client | `client.chat.completions.create(...)` |
| `<sdk-response-shape>` | the response object's text-extraction path | `response.choices[0].message.content` |

If the SDK doesn't match the `try: ... except RateLimit / Auth / Status`
shape used by the template, keep the transient-vs-permanent error-class
split from `domain/errors.py` and translate whatever the SDK raises
into one of the three domain errors. Never let an SDK-native exception
escape the adapter.

## 3. Add Settings fields

Edit `src/agentic_rag_router/infrastructure/settings.py`:

```python
<vendor>_api_key: SecretStr | None = Field(
    default=None,
    description="<Vendor> API key. Empty means <VENDOR> tier is disabled.",
)
<vendor>_model: str = Field(
    default="<sensible-default-model-id>",
    min_length=1,
    description="<Vendor> model identifier.",
)
```

Add `<vendor>_api_key` to the `_empty_string_is_none` `@field_validator`
decorator's first positional args. Update the `.env.example` with the
new env var. Add the new hook to `.pre-commit-config.yaml` if bandit
or ruff need to see the new module.

## 4. Write unit tests

Create `tests/unit/infrastructure/test_<vendor>_adapter.py`. Match the
shape of `test_anthropic_adapter.py`: happy path + one test per error
class the adapter can raise. Target 100% line coverage of the new
module — the `make check` gate will fail otherwise.

Mock the SDK client at the boundary (`monkeypatch.setattr(adapter._client.<sdk-call-path>, ...)`).
Don't mock the adapter itself.

## 5. Enrol in the contract suite (do not skip)

Edit `tests/contract/conftest.py`. Copy the `# ================ <Vendor>` block
pattern (search for `# ============ Anthropic` to find a reference) and add:

- `_<vendor>_healthy_response()` — a `SimpleNamespace` mimicking the SDK response shape
- `_<vendor>_status_error()` — factory for building SDK `APIStatusError` / equivalent
- `_<vendor>_build(monkeypatch)` — returns an adapter with mocked SDK
- `_<vendor>_inject_transient` / `_inject_permanent` / `_inject_content`
- An `AdapterSpec(name="<vendor>", tier=LLMTier.<tier>, build=..., inject_*)` entry in `LLM_ADAPTERS`

Run `uv run pytest tests/contract/ -v` and confirm the new `[<vendor>]`
cases appear in the parametrization (8 new tests).

## 6. Wire into main.py (only if a new tier is needed)

If the new adapter fills an existing tier (primary / secondary /
tertiary), `main.py::build_llm` doesn't need changes — replace the
existing adapter at construction. If the adapter introduces a genuinely
new tier concept, update `LLMTier` in `domain/llm.py` first, then
amend `build_llm`. Flag this to the user as a spec-level change
(update SPECIFICATION.md §2.1).

## 7. Close the loop

- Update `CHANGELOG.md` under `[Unreleased]`: `### Added` — "<Vendor>Adapter (LLMPort implementation for <vendor>)".
- Update `VERIFICATION.md` if OT-2's expected-output line needs the new vendor name in the PASSED list.
- Run `make check` — expect 219 + 8 new contract cases green.
- Commit using the heredoc pattern from [`.claude/rules/commit.md`](../../rules/commit.md).

---

**Verified by:** the contract suite (OT-2) running green with the new
`[<vendor>]` parametrization; `make check` staying at 100% `src/`
coverage including the new module.
