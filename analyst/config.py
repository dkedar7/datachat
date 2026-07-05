"""Configuration: OpenRouter model + limits."""

from __future__ import annotations

import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.environ.get("DATACHAT_MODEL", "anthropic/claude-haiku-4.5").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_ROWS = int(os.environ.get("DATACHAT_MAX_ROWS", "500000"))  # guard huge files


def has_llm() -> bool:
    return bool(OPENROUTER_API_KEY)
