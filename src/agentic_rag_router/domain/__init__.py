"""Domain layer — pure types, no I/O, no framework coupling.

Populated at T004: `LLMTier`, `LLMResponse`, and the `LLMError` hierarchy.
Import rules: domain modules must not import from `application` or `infrastructure`.
See `.claude/rules/domain.md` (T016) for the full contract.
"""
