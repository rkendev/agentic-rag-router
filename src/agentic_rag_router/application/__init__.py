"""Application layer — ports (Protocols) and the `FallbackModel` composition.

Populated at T005: `LLMPort`, `ConfigPort`, `LoggerPort`, and the `FallbackModel`
composite that chains primary → secondary → tertiary adapters.
Import rules: may import from `domain`; must not import from `infrastructure`.
"""
