"""Provider-neutral LLM request models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
