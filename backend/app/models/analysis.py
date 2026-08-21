"""Structured models produced by future tender analysis workflows."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnalysisModel(BaseModel):
    """Forbid provider-invented fields in structured tender output."""

    model_config = ConfigDict(extra="forbid")


class EvidenceReference(AnalysisModel):
    page_number: int | None = None
    paragraph_index: int | None = None
    table_index: int | None = None
    quote: str | None = None


class RequirementItem(AnalysisModel):
    text: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ScoringItem(AnalysisModel):
    name: str
    description: str | None = None
    score: float | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ImportantDate(AnalysisModel):
    name: str
    value: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class TenderAnalysis(AnalysisModel):
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
    warnings: list[str] = Field(default_factory=list)
