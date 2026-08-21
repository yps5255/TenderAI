"""Construct the single V0.2 provider implementation."""

from __future__ import annotations

from ..settings import Settings
from .exceptions import LLMConfigurationError
from .openai_compatible import OpenAICompatibleLLMProvider
from .provider import LLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider != "openai_compatible":
        raise LLMConfigurationError(
            f"Unsupported LLM provider: {settings.llm_provider}. Expected openai_compatible."
        )
    return OpenAICompatibleLLMProvider(settings)
