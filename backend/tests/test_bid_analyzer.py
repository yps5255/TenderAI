from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from backend.app.analyzer.chunking import build_document_chunks
from backend.app.analyzer.models import DocumentChunk
from backend.app.bid_analyzer import BidAnalyzer, BidAnalyzerError
from backend.app.bid_analyzer.models import (
    BidAnalysis,
    BidCapabilityChunkAnalysis,
    BidChunkAnalysis,
    BidChunkCommercialResponse,
    BidChunkDeviationItem,
    BidChunkEquipmentItem,
    BidChunkExperienceItem,
    BidChunkQualificationMaterial,
    BidChunkSubmittedDocument,
    BidChunkTechnicalResponse,
    BidChunkTextItem,
    BidChunkCertificationItem,
    BidChunkPersonnelItem,
    BidSourceEvidence,
    BidCommercialServiceChunkAnalysis,
    BidCoreDocumentsChunkAnalysis,
    BidTechnicalChunkAnalysis,
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


def source_evidence(source_ref: str, quote: str | None = None) -> BidSourceEvidence:
    return BidSourceEvidence(source_ref=source_ref, quote=quote)


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


class ScriptedLLMProvider:
    def __init__(self, responses: list[BaseModel | Exception] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[type[BaseModel], object]] = []

    def generate(self, messages, response_model):
        self.calls.append((response_model, messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response.model_dump())


def empty_groups() -> list[BaseModel]:
    return [
        BidCoreDocumentsChunkAnalysis(),
        BidTechnicalChunkAnalysis(),
        BidCapabilityChunkAnalysis(),
        BidCommercialServiceChunkAnalysis(),
    ]


def test_empty_document_does_not_call_provider() -> None:
    provider = ScriptedLLMProvider([])
    assert BidAnalyzer(provider).analyze(document()).warnings == ["document_has_no_extractable_content"]
    assert provider.calls == []


def test_single_chunk_bid_analysis() -> None:
    source = document("Bidder: Example", content_order=True)
    responses = empty_groups()
    responses[0] = BidCoreDocumentsChunkAnalysis(bidder="Example")
    provider = ScriptedLLMProvider(responses)
    result = BidAnalyzer(provider).analyze(source)
    assert result.bidder == "Example"
    assert result.warnings == []
    assert [model.__name__ for model, _ in provider.calls] == [
        "BidCoreDocumentsChunkAnalysis",
        "BidTechnicalChunkAnalysis",
        "BidCapabilityChunkAnalysis",
        "BidCommercialServiceChunkAnalysis",
    ]


def test_prompt_requires_bid_facts_and_valid_evidence() -> None:
    system = BidAnalyzer._messages(
        "technical", BidTechnicalChunkAnalysis, DocumentChunk(chunk_index=0, content="synthetic")
    )[0].content
    assert "bidder actual technical responses" in system
    assert "bidder states 120kW" in system
    assert "not 100kW" in system
    assert "source_ref copied from the chunk locator" in system
    assert "satisfaction" in system


@pytest.mark.parametrize(
    "expected_rule",
    [
        "[P:7 PAGE:null], return source_ref P:7",
        "[T:3 PAGE:5], return source_ref T:3",
        "Never invent a source_ref",
        "Do not return page_number, paragraph_index, or table_index",
    ],
)
def test_prompt_has_explicit_source_reference_rule(expected_rule: str) -> None:
    system = BidAnalyzer._messages(
        "technical", BidTechnicalChunkAnalysis, DocumentChunk(chunk_index=0, content="synthetic")
    )[0].content
    assert expected_rule in system


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (BidCoreDocumentsChunkAnalysis, {"project_name", "project_number", "bidder", "qualification_materials", "submitted_documents", "warnings"}),
        (BidTechnicalChunkAnalysis, {"technical_responses", "deviation_items", "technical_solution", "warnings"}),
        (BidCapabilityChunkAnalysis, {"experience_items", "certifications", "personnel", "equipment", "warnings"}),
        (BidCommercialServiceChunkAnalysis, {"bid_price", "delivery_commitment", "validity_period", "commercial_responses", "service_commitments", "warnings"}),
    ],
)
def test_group_schema_is_narrow_and_forbids_extra(model: type[BaseModel], fields: set[str]) -> None:
    assert set(model.model_fields) == fields
    assert model.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({"unrelated": "value"})


def test_group_prompts_are_distinct_and_share_safety_rules() -> None:
    prompts = {
        group: BidAnalyzer._messages(group, model, DocumentChunk(chunk_index=0, content="synthetic"))[0].content
        for group, model in [
            ("core_documents", BidCoreDocumentsChunkAnalysis),
            ("technical", BidTechnicalChunkAnalysis),
            ("capability", BidCapabilityChunkAnalysis),
            ("commercial_service", BidCommercialServiceChunkAnalysis),
        ]
    }
    assert len(set(prompts.values())) == 4
    assert "documents actually submitted" in prompts["core_documents"]
    assert "technical solution" in prompts["technical"]
    assert "experience, certifications, personnel, and equipment" in prompts["capability"]
    assert "bidder own price" in prompts["commercial_service"]
    assert all("Never invent a source_ref" in prompt for prompt in prompts.values())


def test_three_chunk_synthetic_merge() -> None:
    source = document("License and permit", "Technical response and deviation", "Price and service")
    first = BidCoreDocumentsChunkAnalysis(
        project_name="Synthetic Project",
        project_number="SYN-1",
        bidder="Example Bidder",
        qualification_materials=[BidChunkQualificationMaterial(name="Business license", evidence=[source_evidence("P:1")])],
        submitted_documents=[BidChunkSubmittedDocument(name="Permit", evidence=[source_evidence("P:1")])],
    )
    second_technical = BidTechnicalChunkAnalysis(
        technical_responses=[BidChunkTechnicalResponse(
            response="Rated power 120kW", declared_status="stated_compliant", evidence=[source_evidence("P:2")]
        )],
        deviation_items=[BidChunkDeviationItem(description="No deviation", deviation_type="none", evidence=[source_evidence("P:2")])],
        technical_solution=[BidChunkTextItem(text="Synthetic solution", evidence=[source_evidence("P:2")])],
    )
    second_capability = BidCapabilityChunkAnalysis(
        personnel=[BidChunkPersonnelItem(role="Engineer", description="Two people", evidence=[source_evidence("P:2")])],
        equipment=[BidChunkEquipmentItem(name="Crane", quantity="2", evidence=[source_evidence("P:2")])],
    )
    third_capability = BidCapabilityChunkAnalysis(
        experience_items=[BidChunkExperienceItem(project_name="Prior work", evidence=[source_evidence("P:3")])],
        certifications=[BidChunkCertificationItem(name="ISO", evidence=[source_evidence("P:3")])],
    )
    third_commercial = BidCommercialServiceChunkAnalysis(
        bid_price="CNY 100",
        delivery_commitment="30 days",
        validity_period="90 days",
        commercial_responses=[BidChunkCommercialResponse(response="Payment accepted", evidence=[source_evidence("P:3")])],
        service_commitments=[BidChunkTextItem(text="24-hour support", evidence=[source_evidence("P:3")])],
    )
    responses = [first, *empty_groups()[1:], empty_groups()[0], second_technical, second_capability,
                 empty_groups()[3], empty_groups()[0], empty_groups()[1], third_capability, third_commercial]
    provider = ScriptedLLMProvider(responses)
    result = BidAnalyzer(provider, chunk_builder=one_paragraph_chunks).analyze(source)
    assert (result.project_name, result.bid_price, result.delivery_commitment) == ("Synthetic Project", "CNY 100", "30 days")
    assert (result.project_number, result.bidder, result.validity_period) == ("SYN-1", "Example Bidder", "90 days")
    for field in (
        "qualification_materials", "submitted_documents", "technical_responses", "deviation_items",
        "technical_solution", "personnel", "equipment", "experience_items", "certifications",
        "commercial_responses", "service_commitments",
    ):
        assert len(getattr(result, field)) == 1
    assert result.technical_responses[0].response == "Rated power 120kW"
    assert result.technical_responses[0].declared_status == DeclaredResponseStatus.STATED_COMPLIANT
    assert result.deviation_items[0].deviation_type == DeviationType.NONE
    assert "100kW" not in result.technical_responses[0].response
    assert all(item.evidence[0].paragraph_index in {1, 2, 3} for field in (
        "qualification_materials", "submitted_documents", "technical_responses", "deviation_items",
        "technical_solution", "personnel", "equipment", "experience_items", "certifications",
        "commercial_responses", "service_commitments",
    ) for item in getattr(result, field))
    assert len(provider.calls) == 12
    assert [model.__name__ for model, _ in provider.calls] == [
        name for _ in range(3) for name in (
            "BidCoreDocumentsChunkAnalysis", "BidTechnicalChunkAnalysis",
            "BidCapabilityChunkAnalysis", "BidCommercialServiceChunkAnalysis",
        )
    ]


def test_scalar_first_non_null_wins() -> None:
    result = BidAnalyzer._merge([BidAnalysis(), BidAnalysis(bidder="First"), BidAnalysis(bidder="Second")], [])
    assert result.bidder == "First"


def test_scalar_formatting_equivalent_has_no_conflict() -> None:
    result = BidAnalyzer._merge([BidAnalysis(bidder="Example, Ltd."), BidAnalysis(bidder=" example ltd ")], [])
    assert "scalar_conflict:bidder" not in result.warnings


def test_scalar_real_conflict_warns() -> None:
    result = BidAnalyzer._merge([BidAnalysis(bid_price="100"), BidAnalysis(bid_price="200")], [])
    assert result.bid_price == "100"
    assert result.warnings == ["scalar_conflict:bid_price"]


def test_qualification_material_and_evidence_merge() -> None:
    first = QualificationMaterial(name=" Business  License ", description="Copy", evidence=[paragraph_evidence(1)])
    second = QualificationMaterial(name="business license", description="Copy", evidence=[paragraph_evidence(2)])
    result = BidAnalyzer._merge([BidAnalysis(qualification_materials=[first, second])], [])
    assert result.qualification_materials[0].name == first.name
    assert [item.paragraph_index for item in result.qualification_materials[0].evidence] == [1, 2]


def test_technical_responses_with_different_response_are_not_merged() -> None:
    items = [
        TechnicalResponse(subject="Pressure", response="10 MPa", evidence=[paragraph_evidence(1)]),
        TechnicalResponse(subject="Pressure", response="20 MPa", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(technical_responses=items)], []).technical_responses) == 2


def test_declared_status_is_preserved_and_part_of_merge_key() -> None:
    items = [
        TechnicalResponse(response="Accepted", declared_status="stated_compliant", evidence=[paragraph_evidence(1)]),
        TechnicalResponse(response="Accepted", declared_status="unclear", evidence=[paragraph_evidence(2)]),
    ]
    result = BidAnalyzer._merge([BidAnalysis(technical_responses=items)], [])
    assert [item.declared_status for item in result.technical_responses] == [DeclaredResponseStatus.STATED_COMPLIANT, DeclaredResponseStatus.UNCLEAR]


def test_different_deviation_types_are_not_merged() -> None:
    items = [
        DeviationItem(description="Term", deviation_type="positive", evidence=[paragraph_evidence(1)]),
        DeviationItem(description="Term", deviation_type="negative", evidence=[paragraph_evidence(2)]),
    ]
    result = BidAnalyzer._merge([BidAnalysis(deviation_items=items)], [])
    assert [item.deviation_type for item in result.deviation_items] == [DeviationType.POSITIVE, DeviationType.NEGATIVE]


def test_commercial_response_merge_preserves_evidence() -> None:
    items = [
        CommercialResponse(subject="Payment", response="Net 30", evidence=[paragraph_evidence(1)]),
        CommercialResponse(subject=" payment ", response="Net 30", evidence=[paragraph_evidence(2)]),
    ]
    merged = BidAnalyzer._merge([BidAnalysis(commercial_responses=items)], []).commercial_responses
    assert len(merged) == 1
    assert len(merged[0].evidence) == 2


def test_submitted_document_merge() -> None:
    items = [
        SubmittedDocument(name="Audit Report", evidence=[paragraph_evidence(1)]),
        SubmittedDocument(name=" audit  report ", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(submitted_documents=items)], []).submitted_documents) == 1


def test_experience_different_amount_or_date_is_preserved() -> None:
    items = [
        ExperienceItem(project_name="A", contract_amount="100", date_or_period="2024", evidence=[paragraph_evidence(1)]),
        ExperienceItem(project_name="A", contract_amount="200", date_or_period="2024", evidence=[paragraph_evidence(2)]),
        ExperienceItem(project_name="A", contract_amount="100", date_or_period="2025", evidence=[paragraph_evidence(3)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(experience_items=items)], []).experience_items) == 3


def test_certification_number_difference_is_preserved() -> None:
    items = [
        CertificationItem(name="ISO", certificate_number="1", evidence=[paragraph_evidence(1)]),
        CertificationItem(name="ISO", certificate_number="2", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(certifications=items)], []).certifications) == 2


def test_personnel_without_name_is_not_merged_on_role_alone() -> None:
    items = [
        PersonnelItem(role="Welder", description="Team A", evidence=[paragraph_evidence(1)]),
        PersonnelItem(role="Welder", description="Team B", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(personnel=items)], []).personnel) == 2


def test_equipment_quantity_or_specification_difference_is_preserved() -> None:
    items = [
        EquipmentItem(name="Crane", quantity="1", specification="10 t", evidence=[paragraph_evidence(1)]),
        EquipmentItem(name="Crane", quantity="2", specification="10 t", evidence=[paragraph_evidence(2)]),
        EquipmentItem(name="Crane", quantity="1", specification="20 t", evidence=[paragraph_evidence(3)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(equipment=items)], []).equipment) == 3


@pytest.mark.parametrize("field", ["technical_solution", "service_commitments"])
def test_text_collection_deduplicates_close_formatting(field: str) -> None:
    first = BidTextItem(text="1. Provide 24-hour technical support.", evidence=[paragraph_evidence(1)])
    second = BidTextItem(text="Provide 24 hour technical support", evidence=[paragraph_evidence(2)])
    result = BidAnalyzer._merge([BidAnalysis(**{field: [first, second]})], [])
    assert len(getattr(result, field)) == 1
    assert len(getattr(result, field)[0].evidence) == 2


def test_text_collection_preserves_different_numbers() -> None:
    items = [
        BidTextItem(text="Provide support within 24 hours", evidence=[paragraph_evidence(1)]),
        BidTextItem(text="Provide support within 48 hours", evidence=[paragraph_evidence(2)]),
    ]
    assert len(BidAnalyzer._merge([BidAnalysis(service_commitments=items)], []).service_commitments) == 2


def resolver_source(page_number: int | None = 4) -> tuple[ParsedDocument, DocumentChunk]:
    source = ParsedDocument(
        filename="synthetic.pdf",
        file_type="pdf",
        paragraphs=[Paragraph(text="Paragraph seven", index=7, page_number=page_number)],
        tables=[Table(index=3, page_number=5, rows=[["Table three"]])],
    )
    chunk = DocumentChunk(
        chunk_index=0,
        content="[P:7 PAGE:4]\nParagraph seven\n[T:3 PAGE:5]\nROW 1: Table three",
        paragraph_indices=[7],
        table_indices=[3],
        page_numbers=[page_number, 5],
    )
    return source, chunk


def test_paragraph_source_ref_resolves_with_authoritative_page() -> None:
    source, chunk = resolver_source()
    resolved = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:7", "Paragraph seven")], chunk, source
    )
    assert resolved == [EvidenceReference(paragraph_index=7, page_number=4, quote="Paragraph seven")]


def test_table_source_ref_resolves_with_authoritative_page() -> None:
    source, chunk = resolver_source()
    resolved = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("T:3", "Table three")], chunk, source
    )
    assert resolved == [EvidenceReference(table_index=3, page_number=5, quote="Table three")]


def test_source_page_none_remains_none() -> None:
    source, chunk = resolver_source(page_number=None)
    resolved = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:7")], chunk, source
    )
    assert resolved[0].page_number is None


@pytest.mark.parametrize("source_ref", ["P:999", "P:-1", "P:1.5", "X:7", "P: 7", "p:7"])
def test_invented_or_malformed_source_ref_is_rejected(source_ref: str) -> None:
    source, chunk = resolver_source()
    assert BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence(source_ref, "Paragraph seven")], chunk, source
    ) == []


def test_empty_source_ref_is_rejected_by_schema() -> None:
    with pytest.raises(ValidationError, match="string_too_short"):
        BidSourceEvidence(source_ref="  ")


@pytest.mark.parametrize("source_ref", ["P:3", "T:7"])
def test_source_ref_type_cannot_cross_match(source_ref: str) -> None:
    source, chunk = resolver_source()
    assert BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence(source_ref)], chunk, source
    ) == []


def test_source_ref_outside_current_chunk_is_rejected() -> None:
    source, chunk = resolver_source()
    chunk = chunk.model_copy(update={"paragraph_indices": []})
    assert BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:7", "Paragraph seven")], chunk, source
    ) == []


def test_bad_quote_is_removed_but_valid_source_ref_is_retained() -> None:
    source, chunk = resolver_source()
    resolved = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:7", "invented")], chunk, source
    )
    assert resolved[0].quote is None
    assert resolved[0].paragraph_index == 7


def test_invalid_source_ref_with_plausible_quote_is_rejected() -> None:
    source, chunk = resolver_source()
    assert BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:999", "Paragraph seven")], chunk, source
    ) == []


def test_item_with_all_source_refs_invalid_is_dropped() -> None:
    source = document("Business license", content_order=True)
    response = BidCoreDocumentsChunkAnalysis(qualification_materials=[
        BidChunkQualificationMaterial(name="License", evidence=[source_evidence("P:999", "Business license")])
    ])
    result = BidAnalyzer(ScriptedLLMProvider([response, *empty_groups()[1:]])).analyze(source)
    assert result.qualification_materials == []
    assert result.warnings == ["item_without_valid_evidence_dropped:qualification_materials"]


def test_multiple_source_refs_preserve_order_and_deduplicate() -> None:
    source, chunk = resolver_source()
    resolved = BidAnalyzer(FakeLLMProvider(BidChunkAnalysis()))._resolve_evidence(
        [source_evidence("P:7"), source_evidence("T:3"), source_evidence("P:7")], chunk, source
    )
    assert [(item.paragraph_index, item.table_index) for item in resolved] == [(7, None), (None, 3)]


def test_chunk_schema_uses_source_ref_without_domain_locator_fields() -> None:
    schema_text = str(BidChunkAnalysis.model_json_schema())
    assert "source_ref" in schema_text
    assert "paragraph_index" not in schema_text
    assert "table_index" not in schema_text
    assert "page_number" not in schema_text


def test_final_bid_analysis_still_uses_evidence_reference() -> None:
    source = document("Business license", content_order=True)
    response = BidCoreDocumentsChunkAnalysis(qualification_materials=[
        BidChunkQualificationMaterial(name="License", evidence=[source_evidence("P:1", "Business license")])
    ])
    item = BidAnalyzer(ScriptedLLMProvider([response, *empty_groups()[1:]])).analyze(source).qualification_materials[0]
    assert type(item.evidence[0]) is EvidenceReference
    assert item.evidence[0].paragraph_index == 1


def test_one_group_failure_continues_without_chunk_failure() -> None:
    provider = ScriptedLLMProvider([LLMConnectionError("provider raw detail"), *empty_groups()[1:]])
    result = BidAnalyzer(provider).analyze(document("first", content_order=True))
    assert result.warnings == ["group_analysis_failed:core_documents:0"]
    assert "provider raw detail" not in str(result.warnings)


def test_all_groups_in_one_chunk_add_chunk_failure_warning() -> None:
    failures = [LLMConnectionError("synthetic") for _ in range(4)]
    responses = [*failures, BidCoreDocumentsChunkAnalysis(bidder="Survives"), *empty_groups()[1:]]
    result = BidAnalyzer(ScriptedLLMProvider(responses), chunk_builder=one_paragraph_chunks).analyze(
        document("first", "second")
    )
    assert result.bidder == "Survives"
    assert result.warnings[1:6] == [
        "group_analysis_failed:core_documents:0",
        "group_analysis_failed:technical:0",
        "group_analysis_failed:capability:0",
        "group_analysis_failed:commercial_service:0",
        "chunk_analysis_failed:0",
    ]


def test_all_chunks_failure_raises() -> None:
    source = document("only")
    with pytest.raises(BidAnalyzerError, match="All document chunks failed"):
        BidAnalyzer(ScriptedLLMProvider([LLMConnectionError("synthetic") for _ in range(4)])).analyze(source)


def test_legal_empty_group_results_are_successful() -> None:
    result = BidAnalyzer(ScriptedLLMProvider(empty_groups())).analyze(document("none", content_order=True))
    assert result.warnings == []


def test_warnings_are_deterministic_and_deduplicated() -> None:
    results = [
        BidAnalysis(warnings=["safe_warning", "safe_warning"]),
        BidAnalysis(warnings=["safe_warning", "second_warning"]),
    ]
    assert BidAnalyzer._merge(results, ["initial", "initial"]).warnings == ["initial", "safe_warning", "second_warning"]


def test_merge_order_is_deterministic() -> None:
    items = [
        SubmittedDocument(name="First", evidence=[paragraph_evidence(1)]),
        SubmittedDocument(name="Second", evidence=[paragraph_evidence(2)]),
        SubmittedDocument(name=" first ", evidence=[paragraph_evidence(3)]),
    ]
    first = BidAnalyzer._merge([BidAnalysis(submitted_documents=items)], [])
    second = BidAnalyzer._merge([BidAnalysis(submitted_documents=items)], [])
    assert first == second
    assert [item.name for item in first.submitted_documents] == ["First", "Second"]


def test_source_order_warning_when_content_order_unavailable() -> None:
    result = BidAnalyzer(ScriptedLLMProvider(empty_groups())).analyze(document("content"))
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
    assert BidAnalyzer(ScriptedLLMProvider(empty_groups())).analyze(source).warnings == []
