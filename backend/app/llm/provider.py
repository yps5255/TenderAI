"""Minimal provider contract used by future analyzers."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from .models import LLMMessage

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMProvider(Protocol):
    def generate(
        self, messages: list[LLMMessage], response_model: type[ResponseModel]
    ) -> ResponseModel:
        """Return assistant content validated as the requested Pydantic model."""
