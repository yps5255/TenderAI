"""Synchronous tender analysis orchestration without provider-specific details."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher

from ..llm.models import LLMMessage
from ..llm.provider import LLMProvider
from ..models import EvidenceReference, ImportantDate, ParsedDocument, RequirementItem, ScoringItem, TenderAnalysis
from .chunking import build_document_chunks
from .models import DocumentChunk, TenderChunkAnalysis

_WHITESPACE = re.compile(r"\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:(?:[（(]\s*)?\d{1,3}\s*(?:[）).、】【、]|[-:])|[一二三四五六七八九十]+[、.])\s*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[a-zA-Z%]+)?")
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


def _normalize_scalar(value: str) -> str:
    """Compare scalar formatting without changing the first observed display value."""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character.casefold() for character in normalized if not character.isspace() and not unicodedata.category(character).startswith("P"))


def _normalize_requirement(value: str) -> str:
    """Create a conservative comparison key for numbered requirement statements."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _LIST_PREFIX.sub("", normalized)
    normalized = _WHITESPACE.sub("", normalized).casefold()
    normalized = "".join(character for character in normalized if not unicodedata.category(character).startswith("P"))
    return re.sub(r"应当|应", "须", normalized)


def _requirements_are_near_duplicates(left: str, right: str) -> bool:
    """Merge only very close wording variants, never statements with different numeric terms."""
    left_key = _normalize_requirement(left)
    right_key = _normalize_requirement(right)
    if left_key == right_key:
        return True
    if len(left_key) < 12 or len(right_key) < 12:
        return False
    if _NUMBER.findall(left_key) != _NUMBER.findall(right_key):
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.94


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
        last_error: Exception | None = None
        for chunk in chunks:
            try:
                result = self.provider.generate(self._messages(chunk), TenderChunkAnalysis)
                results.append(self._validate_chunk_result(result, chunk, document, warnings))
            except Exception as exc:
                last_error = exc
                warnings.append(f"chunk_analysis_failed:{chunk.chunk_index}")
        if not results:
            raise TenderAnalyzerError("All document chunks failed analysis.") from last_error
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
                elif _normalize_scalar(str(data[field])) != _normalize_scalar(value):
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
    merged: list[RequirementItem] = []
    for item in items:
        for index, existing in enumerate(merged):
            if _requirements_are_near_duplicates(existing.text, item.text):
                merged[index] = existing.model_copy(
                    update={"evidence": _dedupe_evidence([*existing.evidence, *item.evidence])}
                )
                break
        else:
            merged.append(item)
    return merged


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
