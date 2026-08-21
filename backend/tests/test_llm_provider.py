from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from backend.app.llm.exceptions import LLMConfigurationError, LLMConnectionError, LLMHTTPError, LLMResponseError, LLMStructuredOutputError, LLMTimeoutError
from backend.app.llm.factory import create_llm_provider
from backend.app.llm.fake import FakeLLMProvider
from backend.app.llm.models import LLMMessage
from backend.app.llm.openai_compatible import OpenAICompatibleLLMProvider
from backend.app.models import TenderAnalysis
from backend.app.settings import Settings


class Result(BaseModel):
    value: str


def messages() -> list[LLMMessage]:
    return [LLMMessage(role="system", content="Return JSON only."), LLMMessage(role="user", content="Test")]


def provider_with_response(content: str, **settings_values: Any) -> OpenAICompatibleLLMProvider:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    )
    return OpenAICompatibleLLMProvider(Settings(**settings_values), transport=transport)


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_retries == 2


def test_settings_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENDERAI_LLM_BASE_URL", "http://example.test/v1/")
    monkeypatch.setenv("TENDERAI_LLM_MODEL", "synthetic-model")
    assert Settings().llm_base_url == "http://example.test/v1/"
    assert Settings().llm_model == "synthetic-model"


def test_api_key_may_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENDERAI_LLM_API_KEY", "")
    assert Settings().llm_api_key == ""


def test_factory_creates_openai_compatible_provider() -> None:
    assert isinstance(create_llm_provider(Settings()), OpenAICompatibleLLMProvider)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        create_llm_provider(Settings(llm_provider="unknown"))


def test_request_url_model_and_temperature_are_correct() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"value":"ok"}'}}]})

    provider = OpenAICompatibleLLMProvider(
        Settings(llm_base_url="http://provider.test/v1/", llm_model="test-model", llm_temperature=0),
        transport=httpx.MockTransport(handler),
    )
    assert provider.generate(messages(), Result) == Result(value="ok")
    assert str(seen[0].url) == "http://provider.test/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [message.model_dump() for message in messages()]


def test_bearer_header_is_sent_when_api_key_exists() -> None:
    seen: list[httpx.Request] = []
    provider = provider_with_response('{"value":"ok"}', llm_api_key="not-a-real-key")
    provider._transport = httpx.MockTransport(
        lambda request: (seen.append(request) or httpx.Response(200, json={"choices": [{"message": {"content": '{"value":"ok"}'}}]}))
    )
    provider.generate(messages(), Result)
    assert seen[0].headers["authorization"] == "Bearer not-a-real-key"


def test_no_authorization_header_is_sent_when_api_key_is_empty() -> None:
    seen: list[httpx.Request] = []
    provider = OpenAICompatibleLLMProvider(
        Settings(llm_api_key=""),
        transport=httpx.MockTransport(
            lambda request: (seen.append(request) or httpx.Response(200, json={"choices": [{"message": {"content": '{"value":"ok"}'}}]}))
        ),
    )
    provider.generate(messages(), Result)
    assert "authorization" not in seen[0].headers


def test_json_response_is_validated() -> None:
    assert provider_with_response('{"value":"ok"}').generate(messages(), Result) == Result(value="ok")


def test_markdown_fenced_json_is_validated() -> None:
    assert provider_with_response('```json\n{"value":"ok"}\n```').generate(messages(), Result) == Result(value="ok")


def test_malformed_assistant_json_raises_structured_output_error() -> None:
    with pytest.raises(LLMStructuredOutputError, match="not valid JSON"):
        provider_with_response("not json").generate(messages(), Result)


def test_schema_validation_failure_raises_structured_output_error() -> None:
    with pytest.raises(LLMStructuredOutputError, match="did not match"):
        provider_with_response('{"other":"value"}').generate(messages(), Result)


@pytest.mark.parametrize("status_code", [400, 500])
def test_http_errors_are_explicit(status_code: int) -> None:
    provider = OpenAICompatibleLLMProvider(
        Settings(), transport=httpx.MockTransport(lambda _request: httpx.Response(status_code))
    )
    with pytest.raises(LLMHTTPError, match=f"HTTP {status_code}"):
        provider.generate(messages(), Result)


def test_connection_error_is_explicit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    provider = OpenAICompatibleLLMProvider(Settings(llm_max_retries=0), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMConnectionError, match="Unable to connect"):
        provider.generate(messages(), Result)


def test_timeout_is_explicit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = OpenAICompatibleLLMProvider(Settings(llm_max_retries=0), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMTimeoutError, match="timed out"):
        provider.generate(messages(), Result)


def test_connection_retries_are_finite() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    provider = OpenAICompatibleLLMProvider(Settings(llm_max_retries=2), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMConnectionError):
        provider.generate(messages(), Result)
    assert calls == 3


def test_empty_assistant_content_is_explicit() -> None:
    with pytest.raises(LLMResponseError, match="empty assistant content"):
        provider_with_response("   ").generate(messages(), Result)


def test_fake_provider_returns_predefined_model_and_records_prompts() -> None:
    fake = FakeLLMProvider(Result(value="synthetic"))
    supplied_messages = messages()
    assert fake.generate(supplied_messages, Result) == Result(value="synthetic")
    assert fake.received_messages == [supplied_messages]


def test_fake_provider_can_raise_configured_error() -> None:
    fake = FakeLLMProvider(LLMConnectionError("synthetic failure"))
    with pytest.raises(LLMConnectionError, match="synthetic failure"):
        fake.generate(messages(), Result)


def test_tender_analysis_lists_default_to_empty() -> None:
    analysis = TenderAnalysis()
    assert analysis.qualification_requirements == []
    assert analysis.important_dates == []


def test_tender_analysis_rejects_provider_invented_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        TenderAnalysis.model_validate({"metadata": {"invented": "value"}})
