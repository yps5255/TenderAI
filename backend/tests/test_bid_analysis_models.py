from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.bid_analyzer.models import (
    BidAnalysis,
    BidTextItem,
    CertificationItem,
    CommercialResponse,
    DeclaredResponseStatus,
    DeviationItem,
    DeviationType,
    EquipmentItem,
    ExperienceItem,
    PersonnelItem,
    QualificationMaterial,
    SubmittedDocument,
    TechnicalResponse,
)
from backend.app.models import EvidenceReference


def evidence() -> EvidenceReference:
    return EvidenceReference(page_number=3, paragraph_index=8, table_index=None, quote="synthetic quote")


def test_minimal_bid_analysis() -> None:
    assert BidAnalysis() == BidAnalysis(project_name=None, project_number=None, bidder=None)


def test_full_bid_analysis() -> None:
    result = BidAnalysis(
        project_name="Synthetic Project",
        project_number="SYN-001",
        bidder="Example Bidder",
        bid_price="CNY 100",
        delivery_commitment="30 days",
        validity_period="90 days",
        qualification_materials=[QualificationMaterial(name="Business license")],
        technical_responses=[TechnicalResponse(response="Meets the stated parameter")],
        commercial_responses=[CommercialResponse(response="Payment terms accepted")],
        deviation_items=[DeviationItem(description="No deviation", deviation_type=DeviationType.NONE)],
        submitted_documents=[SubmittedDocument(name="License copy")],
        experience_items=[ExperienceItem(project_name="Prior project")],
        certifications=[CertificationItem(name="Quality certificate")],
        personnel=[PersonnelItem(role="Welder")],
        equipment=[EquipmentItem(name="Welding machine")],
        technical_solution=[BidTextItem(text="Synthetic technical approach")],
        service_commitments=[BidTextItem(text="Synthetic support commitment")],
        warnings=["synthetic_warning"],
    )
    assert result.bidder == "Example Bidder"
    assert result.deviation_items[0].deviation_type is DeviationType.NONE


def test_default_lists_are_empty_and_independent() -> None:
    first = BidAnalysis()
    second = BidAnalysis()
    first.warnings.append("only_first")
    assert second.warnings == []
    assert second.qualification_materials == []


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BidAnalysis(unexpected="value")


def test_qualification_material() -> None:
    item = QualificationMaterial(name=" Business license ", description="Synthetic", evidence=[evidence()])
    assert item.name == "Business license"
    assert item.evidence[0].paragraph_index == 8


def test_technical_response() -> None:
    item = TechnicalResponse(subject="Parameter", response=" Accepted ", evidence=[evidence()])
    assert item.response == "Accepted"


def test_declared_status_enum() -> None:
    item = TechnicalResponse(response="Declared compliant", declared_status="stated_compliant")
    assert item.declared_status is DeclaredResponseStatus.STATED_COMPLIANT
    assert item.model_dump(mode="json")["declared_status"] == "stated_compliant"


def test_deviation_enum() -> None:
    item = DeviationItem(description="Synthetic deviation", deviation_type="negative")
    assert item.deviation_type is DeviationType.NEGATIVE


def test_commercial_response() -> None:
    item = CommercialResponse(subject="Payment", response="Net 30", evidence=[evidence()])
    assert item.subject == "Payment"


def test_submitted_document() -> None:
    item = SubmittedDocument(name="Audit report", description="Synthetic", evidence=[evidence()])
    assert item.name == "Audit report"


def test_experience_item() -> None:
    item = ExperienceItem(
        project_name="Prior Project",
        client="Example Client",
        contract_amount="CNY 50",
        date_or_period="2025",
        description="Synthetic experience",
        evidence=[evidence()],
    )
    assert item.contract_amount == "CNY 50"


def test_certification_item() -> None:
    item = CertificationItem(
        name="Quality certificate",
        certificate_number="CERT-1",
        validity="2026",
        description="Synthetic",
        evidence=[evidence()],
    )
    assert item.certificate_number == "CERT-1"


def test_personnel_allows_optional_name() -> None:
    item = PersonnelItem(name=None, role="Welder", qualification="Certified", description="12 people")
    assert item.name is None
    assert item.role == "Welder"


def test_equipment_item() -> None:
    item = EquipmentItem(name="Crane", quantity="2 units", specification="Synthetic", evidence=[evidence()])
    assert item.quantity == "2 units"


def test_technical_solution() -> None:
    result = BidAnalysis(technical_solution=[BidTextItem(text="Proposed method", evidence=[evidence()])])
    assert result.technical_solution[0].text == "Proposed method"


def test_service_commitment() -> None:
    result = BidAnalysis(service_commitments=[BidTextItem(text="24-hour response", evidence=[evidence()])])
    assert result.service_commitments[0].text == "24-hour response"


def test_evidence_reference_is_reused() -> None:
    item = QualificationMaterial(name="License", evidence=[evidence()])
    assert type(item.evidence[0]) is EvidenceReference


def test_serialization_round_trip() -> None:
    original = BidAnalysis(
        bidder="Example Bidder",
        technical_responses=[
            TechnicalResponse(
                response="No deviation",
                declared_status=DeclaredResponseStatus.STATED_NO_DEVIATION,
                evidence=[evidence()],
            )
        ],
    )
    restored = BidAnalysis.model_validate_json(original.model_dump_json())
    assert restored == original


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (BidTextItem, {"text": "  "}),
        (QualificationMaterial, {"name": "\t"}),
        (TechnicalResponse, {"response": "\n"}),
        (CommercialResponse, {"response": " "}),
        (DeviationItem, {"description": " ", "deviation_type": "unclear"}),
        (SubmittedDocument, {"name": " "}),
        (CertificationItem, {"name": " "}),
        (EquipmentItem, {"name": " "}),
    ],
)
def test_whitespace_only_required_field_is_rejected(model, data) -> None:
    with pytest.raises(ValidationError, match="string_too_short"):
        model.model_validate(data)


def test_json_schema_generation() -> None:
    schema = BidAnalysis.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "technical_responses" in schema["properties"]
    assert "DeclaredResponseStatus" in schema["$defs"]
    assert "EvidenceReference" in schema["$defs"]
