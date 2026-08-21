"""Application settings loaded from environment variables or an optional local .env file."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the V0.2 LLM provider boundary."""

    model_config = SettingsConfigDict(env_prefix="TENDERAI_", env_file=".env", extra="ignore")

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    analysis_chunk_max_chars: int = Field(default=12000, gt=0)
    analysis_chunk_overlap_chars: int = Field(default=1000, ge=0)

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "Settings":
        if self.analysis_chunk_overlap_chars >= self.analysis_chunk_max_chars:
            raise ValueError("analysis_chunk_overlap_chars must be smaller than analysis_chunk_max_chars")
        return self
