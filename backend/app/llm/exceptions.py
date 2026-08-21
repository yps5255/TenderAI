"""Explicit errors raised by LLM providers."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM boundary errors."""


class LLMConfigurationError(LLMError):
    """Raised for invalid provider configuration."""


class LLMConnectionError(LLMError):
    """Raised when an LLM endpoint cannot be reached."""


class LLMTimeoutError(LLMConnectionError):
    """Raised when an LLM request exceeds its configured timeout."""


class LLMHTTPError(LLMError):
    """Raised when an LLM endpoint returns a non-success HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"LLM provider returned HTTP {status_code}.")


class LLMResponseError(LLMError):
    """Raised when a successful HTTP response has an unusable provider payload."""


class LLMStructuredOutputError(LLMResponseError):
    """Raised when assistant content is not valid requested structured output."""
