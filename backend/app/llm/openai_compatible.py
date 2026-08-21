"""HTTP implementation for the portable OpenAI-compatible chat-completions shape."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ..settings import Settings
from .exceptions import LLMConnectionError, LLMHTTPError, LLMResponseError, LLMStructuredOutputError, LLMTimeoutError
from .models import LLMMessage
from .provider import ResponseModel

_FENCED_JSON = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL)


class OpenAICompatibleLLMProvider:
    """Call POST /chat/completions without relying on any provider SDK."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        self._transport = transport

    def _request_payload(self, messages: list[LLMMessage]) -> dict[str, Any]:
        return {
            "model": self.settings.llm_model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.settings.llm_temperature,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return headers

    @staticmethod
    def _extract_content(payload: Any) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response did not contain assistant content.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM response contained empty assistant content.")
        return content.strip()

    @staticmethod
    def _parse_json(content: str) -> Any:
        match = _FENCED_JSON.fullmatch(content)
        candidate = match.group(1).strip() if match else content
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMStructuredOutputError("LLM assistant content was not valid JSON.") from exc

    def generate(self, messages: list[LLMMessage], response_model: type[ResponseModel]) -> ResponseModel:
        payload = self._request_payload(messages)
        attempts = self.settings.llm_max_retries + 1
        response: httpx.Response | None = None
        with httpx.Client(timeout=self.settings.llm_timeout_seconds, transport=self._transport) as client:
            for attempt in range(attempts):
                try:
                    response = client.post(self.url, json=payload, headers=self._headers())
                    break
                except httpx.TimeoutException as exc:
                    if attempt == attempts - 1:
                        raise LLMTimeoutError("LLM request timed out.") from exc
                except httpx.RequestError as exc:
                    if attempt == attempts - 1:
                        raise LLMConnectionError("Unable to connect to LLM provider.") from exc

        assert response is not None
        if not response.is_success:
            raise LLMHTTPError(response.status_code)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM provider returned invalid JSON response body.") from exc
        structured_data = self._parse_json(self._extract_content(body))
        try:
            return response_model.model_validate(structured_data)
        except ValidationError as exc:
            raise LLMStructuredOutputError("LLM structured output did not match the requested schema.") from exc
