from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.app.analyzer.chunking import build_document_chunks
from backend.app.analyzer.models import DocumentChunk
from backend.app.bid_analyzer import BidAnalyzer, BidAnalyzerError
from backend.app.bid_analyzer.models import (
    BidChunkAnalysis,
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
from backend.app.llm.exceptions import LLMConnectionError
from backend.app.llm.fake import FakeLLMProvider
from backend.app.models import ContentBlockReference, EvidenceReference, Paragraph, ParsedDocument, Table


def paragraph_evidence(index: int, quote: str | None = None, page: int | None = None) -> EvidenceReference:
    return EvidenceReference(paragraph_index=index, page_number=page, quote=quote)


def document(*texts: str, content_order: bool = False) -> ParsedDocument:
    paragraphs = [Paragraph(text=text, index=index, page_number=index) for index, text in enumerate(texts, start=1)]
    order = [ContentBlockReference(type="paragraph", index=item.index) for item in paragraphs] if content_order else []
    return ParsedDocument(filename="synthetic-bid.docx", file_type="docx", paragraphs=paragraphs, content_order=order)


def one_paragraph_chunks(source: ParsedDocument) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_index=index,
            content=f"[P:{item.index} PAGE:{item.page_number}]\n{item.text}",
            paragraph_indices=[item.index],
            page_numbers=[item.page_number],
        )
        for index, item in enumerate(source.paragraphs)
    ]


class SequenceFakeLLMProvider(FakeLLMProvider):
    def __init__(self, responses: Sequence[BidChunkAnalysis | Exception]) -> None:
        super().__init__(BidChunkAnalysis())
        self.responses = list(responses)

    def generate(self, messages, response_model):
        self.received_messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response.model_dump())


def analyze_responses(source: ParsedDocument, *responses: BidChunkAnalysis) -> object:
    return BidAnalyzer(SequenceFakeLLMProvider(responses), chunk_builder=one_paragraph_chunks).analyze(source)


def test_empty_document_does_not_call_provider() -> None:
    provider = FakeLLMProvider(BidChunkAnalysis())
    assert BidAnalyzer(provider).analyze(document()).warnings == ["document_has_no_extractable_content"]
    assert provider.received_messages == []


def test_single_chunk_bid_analysis() -> None:
    source = document("Bidder: Example", content_order=True)
    result = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis(bidder="Example"))).analyze(source)
    assert result.bidder == "Example"
    assert result.warnings == []


def test_prompt_requires_bid_facts_and_valid_evidence() -> None:
    source = document("Bidder commits to delivery", content_order=True)
    provider = FakeLLMProvider(BidChunkAnalysis())
    BidAnalyzer(provider).analyze(source)
    system = provider.received_messages[0][0].content
    assert "actually provided, declared, committed" in system
    assert "template instructions" in system
    assert "Evidence must cite" in system
    assert "satisfaction" in system


def test_three_chunk_synthetic_merge() -> None:
    source = document("License and permit", "Technical response and deviation", "Price and service")
    first = BidChunkAnalysis(
        project_name="Synthetic Project",
        project_number="SYN-1",
        bidder="Example Bidder",
        qualification_materials=[QualificationMaterial(name="Business license", evidence=[paragraph_evidence(1)])],
        submitted_documents=[SubmittedDocument(name="Permit", evidence=[paragraph_evidence(1)])],
    )
    second = BidChunkAnalysis(
        technical_responses=[TechnicalResponse(response="Parameter accepted", evidence=[paragraph_evidence(2)])],
        deviation_items=[DeviationItem(description="No deviation", deviation_type="none", evidence=[paragraph_evidence(2)])],
        technical_solution=[BidTextItem(text="Synthetic solution", evidence=[paragraph_evidence(2)])],
        personnel=[PersonnelItem(role="Engineer", description="Two people", evidence=[paragraph_evidence(2)])],
        equipment=[EquipmentItem(name="Crane", quantity="2", evidence=[paragraph_evidence(2)])],
    )
    third = BidChunkAnalysis(
        bid_price="CNY 100",
        delivery_commitment="30 days",
        validity_period="90 days",
        commercial_responses=[CommercialResponse(response="Payment accepted", evidence=[paragraph_evidence(3)])],
        experience_items=[ExperienceItem(project_name="Prior work", evidence=[paragraph_evidence(3)])],
        certifications=[CertificationItem(name="ISO", evidence=[paragraph_evidence(3)])],
        service_commitments=[BidTextItem(text="24-hour support", evidence=[paragraph_evidence(3)])],
    )
    result = analyze_responses(source, first, second, third)
    assert (result.project_name, result.bid_price, result.delivery_commitment) == ("Synthetic Project", "CNY 100", "30 days")
    assert len(result.qualification_materials) == len(result.technical_responses) == len(result.service_commitments) == 1


def test_scalar_first_non_null_wins() -> None:
    result = BidAnalyzer._merge([BidChunkAnalysis(), BidChunkAnalysis(bidder="First"), BidChunkAnalysis(bidder="Second")], [])
    assert result.bidder == "First"


def test_scalar_formatting_equivalent_has_no_conflict() -> None:
    result = BidAnalyzer._merge([BidChunkAnalysis(bidder="Example, Ltd."), BidChunkAnalysis(bidder=" example ltd ")], [])
    assert "scalar_conflict:bidder" not in result.warnings


def test_scalar_real_conflict_warns() -> None:
    result = BidAnalyzer._merge([BidChunkAnalysis(bid_price="100"), BidChunkAnalysis(bid_price="200")], [])
    assert result.bid_price == "100"
    assert result.warnings == ["scalar_conflict:bid_price"]


def test_qualification_material_and_evidence_merge() -> None:
    first = QualificationMaterial(name=" Business  License ", description="Copy", evidence=[paragraph_evidence(1)])
    second = QualificationMaterial(name="business license", description="Copy", evidence=[paragraph_evidence(2)])
    result = BidAnalyzer._merge([BidChunkAnalysis(qualification_materials=[first, second])], [])
    assert result.qualification_materials[0].name == first.name
    assert [item.paragraph_index for item in result.qualification_materials[0].evidence] == [1, 2]


def test_technical_responses_with_different_response_are_not_merged() -> None:
    items = [
        TechnicalResponse(subject="Pressure", response="10 MPa", evidence=[paragraph_evidence(1)]),
        TechnicalResponse(subject="Pressure", response="20 MPa", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(technical_responses=items)], []).technical_responses) == 2


def test_declared_status_is_preserved_and_part_of_merge_key() -> None:
    items = [
        TechnicalResponse(response="Accepted", declared_status="stated_compliant", evidence=[paragraph_evidence(1)]),
        TechnicalResponse(response="Accepted", declared_status="unclear", evidence=[paragraph_evidence(2)]),
    ]
    result = BidAnalyzer._merge([BidChunkAnalysis(technical_responses=items)], [])
    assert [item.declared_status for item in result.technical_responses] == [DeclaredResponseStatus.STATED_COMPLIANT, DeclaredResponseStatus.UNCLEAR]


def test_different_deviation_types_are_not_merged() -> None:
    items = [
        DeviationItem(description="Term", deviation_type="positive", evidence=[paragraph_evidence(1)]),
        DeviationItem(description="Term", deviation_type="negative", evidence=[paragraph_evidence(2)]),
    ]
    result = BidAnalyzer._merge([BidChunkAnalysis(deviation_items=items)], [])
    assert [item.deviation_type for item in result.deviation_items] == [DeviationType.POSITIVE, DeviationType.NEGATIVE]


def test_commercial_response_merge_preserves_evidence() -> None:
    items = [
        CommercialResponse(subject="Payment", response="Net 30", evidence=[paragraph_evidence(1)]),
        CommercialResponse(subject=" payment ", response="Net 30", evidence=[paragraph_evidence(2)]),
    ]
    merged = BidAnalyzer._merge([BidChunkAnalysis(commercial_responses=items)], []).commercial_responses
    assert len(merged) == 1
    assert len(merged[0].evidence) == 2


def test_submitted_document_merge() -> None:
    items = [
        SubmittedDocument(name="Audit Report", evidence=[paragraph_evidence(1)]),
        SubmittedDocument(name=" audit  report ", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(submitted_documents=items)], []).submitted_documents) == 1


def test_experience_different_amount_or_date_is_preserved() -> None:
    items = [
        ExperienceItem(project_name="A", contract_amount="100", date_or_period="2024", evidence=[paragraph_evidence(1)]),
        ExperienceItem(project_name="A", contract_amount="200", date_or_period="2024", evidence=[paragraph_evidence(2)]),
        ExperienceItem(project_name="A", contract_amount="100", date_or_period="2025", evidence=[paragraph_evidence(3)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(experience_items=items)], []).experience_items) == 3


def test_certification_number_difference_is_preserved() -> None:
    items = [
        CertificationItem(name="ISO", certificate_number="1", evidence=[paragraph_evidence(1)]),
        CertificationItem(name="ISO", certificate_number="2", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(certifications=items)], []).certifications) == 2


def test_personnel_without_name_is_not_merged_on_role_alone() -> None:
    items = [
        PersonnelItem(role="Welder", description="Team A", evidence=[paragraph_evidence(1)]),
        PersonnelItem(role="Welder", description="Team B", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(personnel=items)], []).personnel) == 2


def test_equipment_quantity_or_specification_difference_is_preserved() -> None:
    items = [
        EquipmentItem(name="Crane", quantity="1", specification="10 t", evidence=[paragraph_evidence(1)]),
        EquipmentItem(name="Crane", quantity="2", specification="10 t", evidence=[paragraph_evidence(2)]),
        EquipmentItem(name="Crane", quantity="1", specification="20 t", evidence=[paragraph_evidence(3)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(equipment=items)], []).equipment) == 3


@pytest.mark.parametrize("field", ["technical_solution", "service_commitments"])
def test_text_collection_deduplicates_close_formatting(field: str) -> None:
    first = BidTextItem(text="1. Provide 24-hour technical support.", evidence=[paragraph_evidence(1)])
    second = BidTextItem(text="Provide 24 hour technical support", evidence=[paragraph_evidence(2)])
    result = BidAnalyzer._merge([BidChunkAnalysis(**{field: [first, second]})], [])
    assert len(getattr(result, field)) == 1
    assert len(getattr(result, field)[0].evidence) == 2


def test_text_collection_preserves_different_numbers() -> None:
    items = [
        BidTextItem(text="Provide support within 24 hours", evidence=[paragraph_evidence(1)]),
        BidTextItem(text="Provide support within 48 hours", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidChunkAnalysis(service_commitments=items)], []).service_commitments) == 2


def test_invalid_locator_drops_evidence_and_item() -> None:
    source = document("Business license", content_order=True)
    response = BidChunkAnalysis(
        qualification_materials=[QualificationMaterial(name="License", evidence=[paragraph_evidence(99)])]
    )
    result = BidAnalyzer(FakeLLMProvider(response)).analyze(source)
    assert result.qualification_materials == []
    assert result.warnings == ["item_without_valid_evidence_dropped:qualification_materials"]


def test_bad_quote_is_removed_but_valid_locator_is_retained() -> None:
    source = document("Business license", content_order=True)
    response = BidChunkAnalysis(
        qualification_materials=[QualificationMaterial(name="License", evidence=[paragraph_evidence(1, "invented", 1)])]
    )
    item = BidAnalyzer(FakeLLMProvider(response)).analyze(source).qualification_materials[0]
    assert item.evidence[0].quote is None
    assert item.evidence[0].paragraph_index == 1


def test_page_mismatch_drops_item_without_valid_evidence() -> None:
    source = document("Business license", content_order=True)
    response = BidChunkAnalysis(
        qualification_materials=[QualificationMaterial(name="License", evidence=[paragraph_evidence(1, page=2)])]
    )
    assert BidAnalyzer(FakeLLMProvider(response)).analyze(source).qualification_materials == []


def test_valid_table_evidence_is_accepted() -> None:
    source = ParsedDocument(
        filename="synthetic.pdf",
        file_type="pdf",
        tables=[Table(index=4, page_number=2, rows=[["Price", "100"]])],
        content_order=[ContentBlockReference(type="table", index=4)],
    )
    response = BidChunkAnalysis(
        commercial_responses=[
            CommercialResponse(response="Price 100", evidence=[EvidenceReference(table_index=4, page_number=2, quote="100")])
        ]
    )
    assert BidAnalyzer(FakeLLMProvider(response)).analyze(source).commercial_responses[0].evidence[0].table_index == 4


def test_one_chunk_failure_continues() -> None:
    source = document("first", "second")
    provider = SequenceFakeLLMProvider([LLMConnectionError("synthetic"), BidChunkAnalysis(bidder="Survives")])
    result = BidAnalyzer(provider, chunk_builder=one_paragraph_chunks).analyze(source)
    assert result.bidder == "Survives"
    assert "chunk_analysis_failed:0" in result.warnings


def test_all_chunks_failure_raises() -> None:
    source = document("only")
    with pytest.raises(BidAnalyzerError, match="All document chunks failed"):
        BidAnalyzer(SequenceFakeLLMProvider([LLMConnectionError("synthetic")])).analyze(source)


def test_warnings_are_deterministic_and_deduplicated() -> None:
    results = [
        BidChunkAnalysis(warnings=["safe_warning", "safe_warning"]),
        BidChunkAnalysis(warnings=["safe_warning", "second_warning"]),
    ]
    assert BidAnalyzer._merge(results, ["initial", "initial"]).warnings == ["initial", "safe_warning", "second_warning"]


def test_merge_order_is_deterministic() -> None:
    items = [
        SubmittedDocument(name="First", evidence=[paragraph_evidence(1)]),
        SubmittedDocument(name="Second", evidence=[paragraph_evidence(2)]),
        SubmittedDocument(name=" first ", evidence=[paragraph_evidence(3)]),
    ]
    first = BidAnalyzer._merge([BidChunkAnalysis(submitted_documents=items)], [])
    second = BidAnalyzer._merge([BidChunkAnalysis(submitted_documents=items)], [])
    assert first == second
    assert [item.name for item in first.submitted_documents] == ["First", "Second"]


def test_source_order_warning_when_content_order_unavailable() -> None:
    result = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis())).analyze(document("content"))
    assert result.warnings == ["source_order_unavailable"]


def test_content_order_chunking_compatibility() -> None:
    source = ParsedDocument(
        filename="synthetic.docx",
        file_type="docx",
        paragraphs=[Paragraph(text="paragraph", index=1)],
        tables=[Table(index=2, rows=[["table"]])],
        content_order=[ContentBlockReference(type="table", index=2), ContentBlockReference(type="paragraph", index=1)],
    )
    chunk = build_document_chunks(source, max_chars=100, overlap_chars=0)[0]
    assert chunk.content.index("[T:2") < chunk.content.index("[P:1")
    assert BidAnalyzer(FakeLLMProvider(BidChunkAnalysis())).analyze(source).warnings == []
