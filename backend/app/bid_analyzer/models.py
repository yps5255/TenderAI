"""Structured domain models describing the contents of a bid document."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..models import EvidenceReference


RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BidAnalysisModel(BaseModel):
    """Base configuration for stable structured bid output."""

    model_config = ConfigDict(extra="forbid")


class DeclaredResponseStatus(str, Enum):
    """A response status explicitly stated by the bidder, not an assessment."""

    STATED_COMPLIANT = "stated_compliant"
    STATED_POSITIVE_DEVIATION = "stated_positive_deviation"
    STATED_NEGATIVE_DEVIATION = "stated_negative_deviation"
    STATED_NO_DEVIATION = "stated_no_deviation"
    UNCLEAR = "unclear"


class DeviationType(str, Enum):
    """A deviation type explicitly presented in the bid document."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"
    UNCLEAR = "unclear"


class BidTextItem(BidAnalysisModel):
    text: RequiredText
    evidence: list[EvidenceReference] = Field(default_factory=list)


class QualificationMaterial(BidAnalysisModel):
    name: RequiredText
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class TechnicalResponse(BidAnalysisModel):
    subject: str | None = None
    response: RequiredText
    declared_status: DeclaredResponseStatus | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class CommercialResponse(BidAnalysisModel):
    subject: str | None = None
    response: RequiredText
    evidence: list[EvidenceReference] = Field(default_factory=list)


class DeviationItem(BidAnalysisModel):
    subject: str | None = None
    description: RequiredText
    deviation_type: DeviationType
    evidence: list[EvidenceReference] = Field(default_factory=list)


class SubmittedDocument(BidAnalysisModel):
    name: RequiredText
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ExperienceItem(BidAnalysisModel):
    project_name: str | None = None
    client: str | None = None
    contract_amount: str | None = None
    date_or_period: str | None = None
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class CertificationItem(BidAnalysisModel):
    name: RequiredText
    certificate_number: str | None = None
    validity: str | None = None
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class PersonnelItem(BidAnalysisModel):
    name: str | None = None
    role: str | None = None
    qualification: str | None = None
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class EquipmentItem(BidAnalysisModel):
    name: RequiredText
    quantity: str | None = None
    specification: str | None = None
    description: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class BidAnalysis(BidAnalysisModel):
    """What a bidder provided, declared, committed to, or responded with."""

    project_name: str | None = None
    project_number: str | None = None
    bidder: str | None = None
    bid_price: str | None = None
    delivery_commitment: str | None = None
    validity_period: str | None = None
    qualification_materials: list[QualificationMaterial] = Field(default_factory=list)
    technical_responses: list[TechnicalResponse] = Field(default_factory=list)
    commercial_responses: list[CommercialResponse] = Field(default_factory=list)
    deviation_items: list[DeviationItem] = Field(default_factory=list)
    submitted_documents: list[SubmittedDocument] = Field(default_factory=list)
    experience_items: list[ExperienceItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)
    personnel: list[PersonnelItem] = Field(default_factory=list)
    equipment: list[EquipmentItem] = Field(default_factory=list)
    technical_solution: list[BidTextItem] = Field(default_factory=list)
    service_commitments: list[BidTextItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BidChunkAnalysis(BidAnalysisModel):
    """Structured extraction result for one document chunk."""

    project_name: str | None = None
    project_number: str | None = None
    bidder: str | None = None
    bid_price: str | None = None
    delivery_commitment: str | None = None
    validity_period: str | None = None
    qualification_materials: list[QualificationMaterial] = Field(default_factory=list)
    technical_responses: list[TechnicalResponse] = Field(default_factory=list)
    commercial_responses: list[CommercialResponse] = Field(default_factory=list)
    deviation_items: list[DeviationItem] = Field(default_factory=list)
    submitted_documents: list[SubmittedDocument] = Field(default_factory=list)
    experience_items: list[ExperienceItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)
    personnel: list[PersonnelItem] = Field(default_factory=list)
    equipment: list[EquipmentItem] = Field(default_factory=list)
    technical_solution: list[BidTextItem] = Field(default_factory=list)
    service_commitments: list[BidTextItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
