"""Infrastructure layer — concrete adapters implementing the application ports.

Populated across T006-T010:
  - `config.py` — PydanticSettingsConfigAdapter (T006)
  - `logging.py` — StructlogAdapter (T006)
  - `llm/claude_haiku.py` — ClaudeHaikuAdapter (T009)
  - `llm/openai_mini.py` — OpenAIMiniAdapter (T010)
  - `llm/ollama.py` — OllamaAdapter (T008)

Import rules: may import from `domain` and `application`; nothing else in the
package imports from here except `main.py` (the composition root).
"""
