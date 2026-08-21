from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.app.analyzer.chunking import build_document_chunks
from backend.app.analyzer.models import TenderChunkAnalysis
from backend.app.analyzer.tender_analyzer import TenderAnalyzer, TenderAnalyzerError, _merge_requirements
from backend.app.llm.exceptions import LLMConnectionError
from backend.app.llm.fake import FakeLLMProvider
from backend.app.models import EvidenceReference, ImportantDate, Paragraph, ParsedDocument, RequirementItem, ScoringItem, Table, TenderAnalysis
from backend.app.settings import Settings


def document(*, paragraphs: list[Paragraph] | None = None, tables: list[Table] | None = None) -> ParsedDocument:
    return ParsedDocument(filename="synthetic.docx", file_type="docx", paragraphs=paragraphs or [], tables=tables or [])


def requirement(text: str = "must satisfy", evidence: list[EvidenceReference] | None = None) -> RequirementItem:
    return RequirementItem(text=text, evidence=evidence or [EvidenceReference(paragraph_index=15, page_number=3, quote="must")])


class SequenceProvider:
    def __init__(self, responses: Sequence[TenderChunkAnalysis | Exception]) -> None:
        self.responses = list(responses)
        self.received_messages: list[list[object]] = []

    def generate(self, messages: list[object], response_model: type[TenderChunkAnalysis]) -> TenderChunkAnalysis:
        self.received_messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response.model_dump())


def test_empty_document_returns_warning_without_provider_call() -> None:
    fake = FakeLLMProvider(TenderChunkAnalysis())
    result = TenderAnalyzer(fake).analyze(document())
    assert result == TenderAnalysis(warnings=["document_has_no_extractable_content"])
    assert fake.received_messages == []


def test_paragraph_only_document_is_chunked_with_stable_locator() -> None:
    source = document(paragraphs=[Paragraph(text="Paragraph body", page_number=3, index=15)])
    chunks = build_document_chunks(source, max_chars=100, overlap_chars=0)
    assert chunks[0].paragraph_indices == [15]
    assert chunks[0].page_numbers == [3]
    assert "[P:15 PAGE:3]\nParagraph body" in chunks[0].content


def test_table_only_document_is_chunked_with_stable_locator() -> None:
    source = document(tables=[Table(index=4, page_number=5, rows=[["A", "B"], ["C", "D"]])])
    chunks = build_document_chunks(source, max_chars=100, overlap_chars=0)
    assert chunks[0].table_indices == [4]
    assert chunks[0].page_numbers == [5]
    assert "[T:4 PAGE:5]" in chunks[0].content
    assert "ROW 1: A | B" in chunks[0].content


def test_mixed_document_keeps_paragraphs_and_tables() -> None:
    source = document(
        paragraphs=[Paragraph(text="paragraph", page_number=None, index=2)],
        tables=[Table(index=7, page_number=2, rows=[["table"]])],
    )
    joined = "\n".join(chunk.content for chunk in build_document_chunks(source, max_chars=100, overlap_chars=0))
    assert "[P:2 PAGE:null]" in joined
    assert "[T:7 PAGE:2]" in joined


def test_chunk_boundaries_are_deterministic_and_finite() -> None:
    source = document(paragraphs=[Paragraph(text=f"text-{index}", index=index) for index in range(5)])
    first = build_document_chunks(source, max_chars=30, overlap_chars=10)
    second = build_document_chunks(source, max_chars=30, overlap_chars=10)
    assert first == second
    assert 1 < len(first) <= 5
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_oversized_paragraph_is_split_without_content_loss() -> None:
    text = "x" * 80
    chunks = build_document_chunks(document(paragraphs=[Paragraph(text=text, index=9)]), max_chars=30, overlap_chars=0)
    serialized = "".join(chunk.content.split("\n", 1)[1] for chunk in chunks)
    assert serialized == text
    assert all(chunk.paragraph_indices == [9] for chunk in chunks)


def test_oversized_table_is_split_without_row_loss() -> None:
    text = "y" * 80
    chunks = build_document_chunks(document(tables=[Table(index=6, rows=[[text]])]), max_chars=30, overlap_chars=0)
    parts = [chunk.content.split("\n", 1)[1] for chunk in chunks]
    assert "".join(parts) == "ROW 1: " + text
    assert all(chunk.table_indices == [6] for chunk in chunks)


def test_overlap_is_bounded_and_does_not_loop() -> None:
    source = document(paragraphs=[Paragraph(text="x" * 12, index=index) for index in range(4)])
    chunks = build_document_chunks(source, max_chars=40, overlap_chars=15)
    assert len(chunks) < 10
    assert all(len(chunk.content) <= 40 for chunk in chunks)


def test_settings_rejects_overlap_not_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Settings(analysis_chunk_max_chars=100, analysis_chunk_overlap_chars=100)


def test_provider_receives_locator_rich_prompt() -> None:
    source = document(paragraphs=[Paragraph(text="must qualify", page_number=3, index=15)])
    fake = FakeLLMProvider(TenderChunkAnalysis(qualification_requirements=[requirement()]))
    TenderAnalyzer(fake).analyze(source)
    prompt = fake.received_messages[0][1].content
    assert "[P:15 PAGE:3]" in prompt
    assert "paragraph_index" in prompt


def test_multiple_chunks_call_provider_multiple_times() -> None:
    source = document(paragraphs=[Paragraph(text="x" * 30, index=index) for index in range(3)])
    fake = FakeLLMProvider(TenderChunkAnalysis())
    analyzer = TenderAnalyzer(fake, chunk_builder=lambda value: build_document_chunks(value, max_chars=45, overlap_chars=0))
    analyzer.analyze(source)
    assert len(fake.received_messages) > 1


def test_valid_paragraph_evidence_is_accepted() -> None:
    source = document(paragraphs=[Paragraph(text="must qualify", page_number=3, index=15)])
    result = TenderAnalyzer(FakeLLMProvider(TenderChunkAnalysis(qualification_requirements=[requirement()]))).analyze(source)
    assert result.qualification_requirements[0].evidence[0].paragraph_index == 15


def test_invalid_paragraph_evidence_drops_item() -> None:
    source = document(paragraphs=[Paragraph(text="must qualify", page_number=3, index=15)])
    bad = RequirementItem(text="invented", evidence=[EvidenceReference(paragraph_index=999)])
    result = TenderAnalyzer(FakeLLMProvider(TenderChunkAnalysis(qualification_requirements=[bad]))).analyze(source)
    assert result.qualification_requirements == []
    assert "item_without_valid_evidence_dropped:qualification_requirements" in result.warnings


def test_valid_table_evidence_is_accepted_and_invalid_table_is_rejected() -> None:
    source = document(tables=[Table(index=4, page_number=5, rows=[["qualified"]])])
    valid = RequirementItem(text="valid", evidence=[EvidenceReference(table_index=4, page_number=5, quote="qualified")])
    invalid = RequirementItem(text="invalid", evidence=[EvidenceReference(table_index=99)])
    result = TenderAnalyzer(FakeLLMProvider(TenderChunkAnalysis(technical_requirements=[valid, invalid]))).analyze(source)
    assert [item.text for item in result.technical_requirements] == ["valid"]


def test_quote_validation_keeps_matching_quote_and_removes_nonmatching_quote() -> None:
    source = document(paragraphs=[Paragraph(text="must qualify now", page_number=3, index=15)])
    matching = RequirementItem(text="one", evidence=[EvidenceReference(paragraph_index=15, quote="qualify now")])
    nonmatching = RequirementItem(text="two", evidence=[EvidenceReference(paragraph_index=15, quote="invented quote")])
    result = TenderAnalyzer(FakeLLMProvider(TenderChunkAnalysis(technical_requirements=[matching, nonmatching]))).analyze(source)
    assert result.technical_requirements[0].evidence[0].quote == "qualify now"
    assert result.technical_requirements[1].evidence[0].quote is None


def test_duplicate_requirements_merge_evidence_and_deduplicate_evidence() -> None:
    source = document(paragraphs=[Paragraph(text="first", index=1), Paragraph(text="second", index=2)])
    one = TenderChunkAnalysis(technical_requirements=[RequirementItem(text="  Must   Do ", evidence=[EvidenceReference(paragraph_index=1), EvidenceReference(paragraph_index=1)])])
    two = TenderChunkAnalysis(technical_requirements=[RequirementItem(text="must do", evidence=[EvidenceReference(paragraph_index=2)])])
    result = TenderAnalyzer(SequenceProvider([one, two]), chunk_builder=lambda _value: [
        build_document_chunks(source, max_chars=100, overlap_chars=0)[0],
        build_document_chunks(source, max_chars=100, overlap_chars=0)[0].model_copy(update={"chunk_index": 1, "paragraph_indices": [2]}),
    ]).analyze(source)
    assert len(result.technical_requirements) == 1
    assert [item.paragraph_index for item in result.technical_requirements[0].evidence] == [1, 2]


def test_scoring_and_dates_are_conservatively_deduplicated() -> None:
    source = document(paragraphs=[Paragraph(text="score date", index=1)])
    evidence = [EvidenceReference(paragraph_index=1)]
    first = TenderChunkAnalysis(scoring_items=[ScoringItem(name="Price", description=" total ", score=10, evidence=evidence)], important_dates=[ImportantDate(name="Deadline", value="2026-01-01", evidence=evidence)])
    second = TenderChunkAnalysis(scoring_items=[ScoringItem(name="price", description="total", score=10, evidence=evidence), ScoringItem(name="Price", description="total", score=11, evidence=evidence)], important_dates=[ImportantDate(name=" deadline ", value="2026-01-01", evidence=evidence)])
    result = TenderAnalyzer(SequenceProvider([first, second]), chunk_builder=lambda value: [
        build_document_chunks(value, max_chars=100, overlap_chars=0)[0],
        build_document_chunks(value, max_chars=100, overlap_chars=0)[0].model_copy(update={"chunk_index": 1}),
    ]).analyze(source)
    assert len(result.scoring_items) == 2
    assert len(result.important_dates) == 1


def test_scalar_uses_first_non_null_and_reports_conflict() -> None:
    source = document(paragraphs=[Paragraph(text="value", index=1)])
    first = TenderChunkAnalysis(project_name="A")
    second = TenderChunkAnalysis(project_name="B", budget="100")
    chunk = build_document_chunks(source, max_chars=100, overlap_chars=0)[0]
    result = TenderAnalyzer(SequenceProvider([first, second]), chunk_builder=lambda _value: [chunk, chunk.model_copy(update={"chunk_index": 1})]).analyze(source)
    assert result.project_name == "A"
    assert result.budget == "100"
    assert "scalar_conflict:project_name" in result.warnings


def test_one_chunk_failure_does_not_abort_remaining_chunks() -> None:
    source = document(paragraphs=[Paragraph(text="content", index=1)])
    chunk = build_document_chunks(source, max_chars=100, overlap_chars=0)[0]
    provider = SequenceProvider([LLMConnectionError("synthetic"), TenderChunkAnalysis(project_name="survives")])
    result = TenderAnalyzer(provider, chunk_builder=lambda _value: [chunk, chunk.model_copy(update={"chunk_index": 1})]).analyze(source)
    assert result.project_name == "survives"
    assert "chunk_analysis_failed:0" in result.warnings


def test_all_chunk_failures_raise_analyzer_error() -> None:
    source = document(paragraphs=[Paragraph(text="content", index=1)])
    with pytest.raises(TenderAnalyzerError, match="All document chunks failed"):
        TenderAnalyzer(SequenceProvider([LLMConnectionError("synthetic")])).analyze(source)


def test_analyzer_output_is_tender_analysis() -> None:
    source = document(paragraphs=[Paragraph(text="content", index=15)])
    result = TenderAnalyzer(FakeLLMProvider(TenderChunkAnalysis())).analyze(source)
    assert isinstance(result, TenderAnalysis)


def test_requirement_merge_handles_exact_whitespace_and_punctuation_variants() -> None:
    merged = _merge_requirements([
        RequirementItem(text="投标人须提供营业执照。", evidence=[EvidenceReference(paragraph_index=1)]),
        RequirementItem(text=" 投标人 须 提供 营业执照 ", evidence=[EvidenceReference(paragraph_index=2)]),
    ])
    assert len(merged) == 1
    assert [item.paragraph_index for item in merged[0].evidence] == [1, 2]


def test_requirement_merge_removes_list_number_and_modal_format_difference() -> None:
    merged = _merge_requirements([
        RequirementItem(text="1. 投标人须提供营业执照", evidence=[EvidenceReference(paragraph_index=1)]),
        RequirementItem(text="（1）投标人应当提供营业执照。", evidence=[EvidenceReference(paragraph_index=2)]),
    ])
    assert len(merged) == 1


def test_requirement_merge_handles_only_very_close_wording() -> None:
    merged = _merge_requirements([
        RequirementItem(text="投标人须在投标文件中提供有效营业执照复印件", evidence=[EvidenceReference(paragraph_index=1)]),
        RequirementItem(text="投标人应在投标文件中提供有效营业执照复印件。", evidence=[EvidenceReference(paragraph_index=2)]),
    ])
    assert len(merged) == 1


def test_requirement_merge_does_not_merge_different_numeric_requirements() -> None:
    merged = _merge_requirements([
        RequirementItem(text="设备额定功率不低于100kW", evidence=[EvidenceReference(paragraph_index=1)]),
        RequirementItem(text="设备额定功率不低于200kW", evidence=[EvidenceReference(paragraph_index=2)]),
    ])
    assert len(merged) == 2


def test_requirement_merge_order_is_deterministic() -> None:
    items = [
        RequirementItem(text="投标人须提供营业执照。", evidence=[EvidenceReference(paragraph_index=1)]),
        RequirementItem(text="投标人应当提供营业执照", evidence=[EvidenceReference(paragraph_index=2)]),
        RequirementItem(text="设备额定功率不低于100kW", evidence=[EvidenceReference(paragraph_index=3)]),
    ]
    assert _merge_requirements(items) == _merge_requirements(items)
    assert [item.text for item in _merge_requirements(items)] == [items[0].text, items[2].text]


def test_scalar_formatting_difference_does_not_create_conflict() -> None:
    result = TenderAnalyzer._merge(
        [TenderChunkAnalysis(tenderer="Example, Ltd."), TenderChunkAnalysis(tenderer=" example ltd ")], []
    )
    assert result.tenderer == "Example, Ltd."
    assert "scalar_conflict:tenderer" not in result.warnings


def test_true_scalar_difference_still_creates_conflict() -> None:
    result = TenderAnalyzer._merge(
        [TenderChunkAnalysis(tenderer="First Entity"), TenderChunkAnalysis(tenderer="Second Entity")], []
    )
    assert "scalar_conflict:tenderer" in result.warnings


def test_synthetic_multichunk_business_license_duplicate_is_merged() -> None:
    source = document(paragraphs=[Paragraph(text="营业执照", index=1), Paragraph(text="营业执照", index=2)])
    first = TenderChunkAnalysis(qualification_requirements=[
        RequirementItem(text="投标人须提供营业执照。", evidence=[EvidenceReference(paragraph_index=1)])
    ])
    second = TenderChunkAnalysis(qualification_requirements=[
        RequirementItem(text="（1）投标人应当提供营业执照", evidence=[EvidenceReference(paragraph_index=2)])
    ])
    chunk = build_document_chunks(source, max_chars=100, overlap_chars=0)[0]
    result = TenderAnalyzer(SequenceProvider([first, second]), chunk_builder=lambda _value: [
        chunk.model_copy(update={"paragraph_indices": [1]}),
        chunk.model_copy(update={"chunk_index": 1, "paragraph_indices": [2]}),
    ]).analyze(source)
    assert len(result.qualification_requirements) == 1
    assert [evidence.paragraph_index for evidence in result.qualification_requirements[0].evidence] == [1, 2]
