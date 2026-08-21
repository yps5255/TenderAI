"""Internal models for chunk-level tender extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..models.analysis import ImportantDate, RequirementItem, ScoringItem


class DocumentChunk(BaseModel):
    """A bounded, locator-rich slice of a parsed document."""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    content: str
    paragraph_indices: list[int] = Field(default_factory=list)
    table_indices: list[int] = Field(default_factory=list)
    page_numbers: list[int | None] = Field(default_factory=list)


class TenderChunkAnalysis(BaseModel):
    """Intermediate extraction result for one source chunk."""

    model_config = ConfigDict(extra="forbid")

    project_name: str | None = None
    project_number: str | None = None
    tenderer: str | None = None
    agency: str | None = None
    deadline: str | None = None
    budget: str | None = None
    qualification_requirements: list[RequirementItem] = Field(default_factory=list)
    scoring_items: list[ScoringItem] = Field(default_factory=list)
    technical_requirements: list[RequirementItem] = Field(default_factory=list)
    commercial_requirements: list[RequirementItem] = Field(default_factory=list)
    rejection_conditions: list[RequirementItem] = Field(default_factory=list)
    required_documents: list[RequirementItem] = Field(default_factory=list)
    important_dates: list[ImportantDate] = Field(default_factory=list)
