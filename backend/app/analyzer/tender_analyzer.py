"""Synchronous tender analysis orchestration without provider-specific details."""

from __future__ import annotations

import re
from collections.abc import Callable

from ..llm.models import LLMMessage
from ..llm.provider import LLMProvider
from ..models import EvidenceReference, ImportantDate, ParsedDocument, RequirementItem, ScoringItem, TenderAnalysis
from .chunking import build_document_chunks
from .models import DocumentChunk, TenderChunkAnalysis

_WHITESPACE = re.compile(r"\s+")
_SCALARS = ("project_name", "project_number", "tenderer", "agency", "deadline", "budget")
_REQUIREMENT_FIELDS = (
    "qualification_requirements",
    "technical_requirements",
    "commercial_requirements",
    "rejection_conditions",
    "required_documents",
)


class TenderAnalyzerError(Exception):
    """Raised when no chunk can be successfully analyzed."""


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _normalized_quote(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


class TenderAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        chunk_builder: Callable[[ParsedDocument], list[DocumentChunk]] = build_document_chunks,
    ) -> None:
        self.provider = provider
        self.chunk_builder = chunk_builder

    @staticmethod
    def _messages(chunk: DocumentChunk) -> list[LLMMessage]:
        schema = TenderChunkAnalysis.model_json_schema()
        system = (
            "You extract tender facts only; never create, infer, or complete information. "
            "Return only JSON. Use null or [] when information is absent. Evidence must cite a locator "
            "that exists in this chunk, and any quote must be copied from chunk content. Do not treat advice, "
            "speculation, or explanation as a tender requirement. Classify requirements as qualification, scoring, "
            "technical, commercial, rejection conditions, required documents, or important dates."
        )
        user = (
            "Extract the following chunk. Locators use [P:<global paragraph_index> PAGE:<page or null>] "
            "and [T:<global table_index> PAGE:<page or null>]. Return JSON matching this schema exactly:\n"
            f"{schema}\n\nCHUNK {chunk.chunk_index}:\n{chunk.content}"
        )
        return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]

    @staticmethod
    def _source_text(document: ParsedDocument) -> tuple[dict[int, tuple[str, int | None]], dict[int, tuple[str, int | None]]]:
        paragraphs = {item.index: (item.text, item.page_number) for item in document.paragraphs}
        tables = {
            item.index: ("\n".join("\t".join(row) for row in item.rows), item.page_number)
            for item in document.tables
        }
        return paragraphs, tables

    def _validate_evidence(
        self, evidence: list[EvidenceReference], chunk: DocumentChunk, document: ParsedDocument
    ) -> list[EvidenceReference]:
        paragraphs, tables = self._source_text(document)
        valid: list[EvidenceReference] = []
        for reference in evidence:
            sources: list[tuple[str, int | None]] = []
            if reference.paragraph_index is not None:
                if reference.paragraph_index not in chunk.paragraph_indices:
                    continue
                source = paragraphs.get(reference.paragraph_index)
                if source is None:
                    continue
                sources.append(source)
            if reference.table_index is not None:
                if reference.table_index not in chunk.table_indices:
                    continue
                source = tables.get(reference.table_index)
                if source is None:
                    continue
                sources.append(source)
            if not sources:
                continue
            if reference.page_number is not None and any(page != reference.page_number for _, page in sources):
                continue
            quote = reference.quote
            if quote is not None and not any(_normalized_quote(quote) in _normalized_quote(text) for text, _ in sources):
                quote = None
            valid.append(reference.model_copy(update={"quote": quote}))
        return _dedupe_evidence(valid)

    def _validate_chunk_result(
        self, result: TenderChunkAnalysis, chunk: DocumentChunk, document: ParsedDocument, warnings: list[str]
    ) -> TenderChunkAnalysis:
        data = result.model_dump()
        for field in (*_REQUIREMENT_FIELDS, "scoring_items", "important_dates"):
            validated_items = []
            for item in data[field]:
                item["evidence"] = self._validate_evidence(
                    [EvidenceReference.model_validate(reference) for reference in item["evidence"]], chunk, document
                )
                if item["evidence"]:
                    validated_items.append(item)
                else:
                    warnings.append(f"item_without_valid_evidence_dropped:{field}")
            data[field] = validated_items
        return TenderChunkAnalysis.model_validate(data)

    def analyze(self, document: ParsedDocument) -> TenderAnalysis:
        chunks = self.chunk_builder(document)
        if not chunks:
            return TenderAnalysis(warnings=["document_has_no_extractable_content"])

        warnings: list[str] = [] if document.content_order else ["source_order_unavailable"]
        results: list[TenderChunkAnalysis] = []
        for chunk in chunks:
            try:
                result = self.provider.generate(self._messages(chunk), TenderChunkAnalysis)
                results.append(self._validate_chunk_result(result, chunk, document, warnings))
            except Exception:
                warnings.append(f"chunk_analysis_failed:{chunk.chunk_index}")
        if not results:
            raise TenderAnalyzerError("All document chunks failed analysis.")
        return self._merge(results, warnings)

    @staticmethod
    def _merge(results: list[TenderChunkAnalysis], warnings: list[str]) -> TenderAnalysis:
        data: dict[str, object] = {field: None for field in _SCALARS}
        for result in results:
            for field in _SCALARS:
                value = getattr(result, field)
                if value is None:
                    continue
                if data[field] is None:
                    data[field] = value
                elif data[field] != value:
                    warnings.append(f"scalar_conflict:{field}")
        for field in _REQUIREMENT_FIELDS:
            data[field] = _merge_requirements([item for result in results for item in getattr(result, field)])
        data["scoring_items"] = _merge_scoring([item for result in results for item in result.scoring_items])
        data["important_dates"] = _merge_dates([item for result in results for item in result.important_dates])
        data["warnings"] = list(dict.fromkeys(warnings))
        return TenderAnalysis.model_validate(data)


def _dedupe_evidence(evidence: list[EvidenceReference]) -> list[EvidenceReference]:
    seen: set[tuple[int | None, int | None, int | None, str | None]] = set()
    output: list[EvidenceReference] = []
    for item in evidence:
        key = (item.page_number, item.paragraph_index, item.table_index, item.quote)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _merge_requirements(items: list[RequirementItem]) -> list[RequirementItem]:
    merged: dict[str, RequirementItem] = {}
    for item in items:
        key = _normalize(item.text)
        if key in merged:
            merged[key] = merged[key].model_copy(update={"evidence": _dedupe_evidence([*merged[key].evidence, *item.evidence])})
        else:
            merged[key] = item
    return list(merged.values())


def _merge_scoring(items: list[ScoringItem]) -> list[ScoringItem]:
    merged: dict[tuple[str, str, float | None], ScoringItem] = {}
    for item in items:
        key = (_normalize(item.name), _normalize(item.description or ""), item.score)
        if key in merged:
            merged[key] = merged[key].model_copy(update={"evidence": _dedupe_evidence([*merged[key].evidence, *item.evidence])})
        else:
            merged[key] = item
    return list(merged.values())


def _merge_dates(items: list[ImportantDate]) -> list[ImportantDate]:
    merged: dict[tuple[str, str], ImportantDate] = {}
    for item in items:
        key = (_normalize(item.name), _normalize(item.value))
        if key in merged:
            merged[key] = merged[key].model_copy(update={"evidence": _dedupe_evidence([*merged[key].evidence, *item.evidence])})
        else:
            merged[key] = item
    return list(merged.values())
