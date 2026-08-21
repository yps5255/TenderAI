"""Offline provider helper for analyzer and provider tests."""

from __future__ import annotations

from pydantic import BaseModel

from .models import LLMMessage
from .provider import ResponseModel


class FakeLLMProvider:
    def __init__(self, response: BaseModel | Exception) -> None:
        self.response = response
        self.received_messages: list[list[LLMMessage]] = []

    def generate(self, messages: list[LLMMessage], response_model: type[ResponseModel]) -> ResponseModel:
        self.received_messages.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return response_model.model_validate(self.response.model_dump())
