"""Synchronous bid analysis orchestration without provider-specific details."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import TypeVar

from pydantic import BaseModel

from ..analyzer.chunking import build_document_chunks
from ..analyzer.models import DocumentChunk
from ..llm.models import LLMMessage
from ..llm.provider import LLMProvider
from ..models import EvidenceReference, ParsedDocument
from .models import (
    BidAnalysis,
    BidChunkAnalysis,
    BidTextItem,
    CertificationItem,
    CommercialResponse,
    DeviationItem,
    EquipmentItem,
    ExperienceItem,
    PersonnelItem,
    QualificationMaterial,
    SubmittedDocument,
    TechnicalResponse,
)

_WHITESPACE = re.compile(r"\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:(?:[（(]\s*)?\d{1,3}\s*(?:[）).】【、】【\u3001]|[-:])|[一二三四五六七八九十]+[、.])\s*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[a-zA-Z%]+)?")
_SCALARS = ("project_name", "project_number", "bidder", "bid_price", "delivery_commitment", "validity_period")
_COLLECTIONS = (
    "qualification_materials",
    "technical_responses",
    "commercial_responses",
    "deviation_items",
    "submitted_documents",
    "experience_items",
    "certifications",
    "personnel",
    "equipment",
    "technical_solution",
    "service_commitments",
)

ItemT = TypeVar("ItemT", bound=BaseModel)


class BidAnalyzerError(Exception):
    """Raised when no document chunk can be successfully analyzed."""


def _normalize_component(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _normalize_scalar(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character.casefold()
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _normalize_text(value: str) -> str:
    normalized = _LIST_PREFIX.sub("", unicodedata.normalize("NFKC", value))
    normalized = _WHITESPACE.sub("", normalized).casefold()
    return "".join(character for character in normalized if not unicodedata.category(character).startswith("P"))


def _text_items_are_near_duplicates(left: str, right: str) -> bool:
    left_key = _normalize_text(left)
    right_key = _normalize_text(right)
    if left_key == right_key:
        return True
    if len(left_key) < 12 or len(right_key) < 12:
        return False
    if _NUMBER.findall(left_key) != _NUMBER.findall(right_key):
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.96


def _normalized_quote(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def _dedupe_evidence(evidence: list[EvidenceReference]) -> list[EvidenceReference]:
    seen: set[tuple[int | None, int | None, int | None, str | None]] = set()
    output: list[EvidenceReference] = []
    for item in evidence:
        key = (item.page_number, item.paragraph_index, item.table_index, item.quote)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _merge_exact(items: list[ItemT], key_builder: Callable[[ItemT], tuple[object, ...]]) -> list[ItemT]:
    merged: dict[tuple[object, ...], ItemT] = {}
    for item in items:
        key = key_builder(item)
        if key in merged:
            existing = merged[key]
            merged[key] = existing.model_copy(
                update={"evidence": _dedupe_evidence([*existing.evidence, *item.evidence])}
            )
        else:
            merged[key] = item
    return list(merged.values())


def _merge_text_items(items: list[BidTextItem]) -> list[BidTextItem]:
    merged: list[BidTextItem] = []
    for item in items:
        for index, existing in enumerate(merged):
            if _text_items_are_near_duplicates(existing.text, item.text):
                merged[index] = existing.model_copy(
                    update={"evidence": _dedupe_evidence([*existing.evidence, *item.evidence])}
                )
                break
        else:
            merged.append(item)
    return merged


class BidAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        chunk_builder: Callable[[ParsedDocument], list[DocumentChunk]] = build_document_chunks,
    ) -> None:
        self.provider = provider
        self.chunk_builder = chunk_builder

    @staticmethod
    def _messages(chunk: DocumentChunk) -> list[LLMMessage]:
        schema = BidChunkAnalysis.model_json_schema()
        system = (
            "Extract only facts explicitly supported by the current bid-document chunk. Never infer or complete "
            "missing information. Return structured JSON only; absent scalars must be null and absent collections "
            "must be []. Evidence must cite a [P:n PAGE:x] or [T:n PAGE:x] locator present in this chunk, and quotes "
            "must be copied from the cited source. Distinguish tender requirements quoted in the bid from what the "
            "bidder actually provided, declared, committed to, or responded with. Do not treat template instructions "
            "as bidder commitments. declared_status is allowed only when explicitly stated by the bidder. "
            "deviation_type is allowed only for an explicit deviation table or declaration. Never perform tender-bid "
            "alignment, satisfaction, qualification, scoring, award, or rejection-risk judgments."
        )
        user = (
            "Extract this bid chunk using its global locators. Return JSON matching this schema exactly:\n"
            f"{schema}\n\nCHUNK {chunk.chunk_index}:\n{chunk.content}"
        )
        return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]

    @staticmethod
    def _source_text(
        document: ParsedDocument,
    ) -> tuple[dict[int, tuple[str, int | None]], dict[int, tuple[str, int | None]]]:
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
            if quote is not None and not any(
                _normalized_quote(quote) in _normalized_quote(text) for text, _ in sources
            ):
                quote = None
            valid.append(reference.model_copy(update={"quote": quote}))
        return _dedupe_evidence(valid)

    def _validate_chunk_result(
        self, result: BidChunkAnalysis, chunk: DocumentChunk, document: ParsedDocument, warnings: list[str]
    ) -> BidChunkAnalysis:
        data = result.model_dump()
        for field in _COLLECTIONS:
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
        return BidChunkAnalysis.model_validate(data)

    def analyze(self, document: ParsedDocument) -> BidAnalysis:
        chunks = self.chunk_builder(document)
        if not chunks:
            return BidAnalysis(warnings=["document_has_no_extractable_content"])

        warnings: list[str] = [] if document.content_order else ["source_order_unavailable"]
        results: list[BidChunkAnalysis] = []
        last_error: Exception | None = None
        for chunk in chunks:
            try:
                result = self.provider.generate(self._messages(chunk), BidChunkAnalysis)
                results.append(self._validate_chunk_result(result, chunk, document, warnings))
            except Exception as exc:
                last_error = exc
                warnings.append(f"chunk_analysis_failed:{chunk.chunk_index}")
        if not results:
            raise BidAnalyzerError("All document chunks failed analysis.") from last_error
        return self._merge(results, warnings)

    @staticmethod
    def _merge(results: list[BidChunkAnalysis], warnings: list[str]) -> BidAnalysis:
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

        qualifications = [item for result in results for item in result.qualification_materials]
        data["qualification_materials"] = _merge_exact(
            qualifications, lambda item: (_normalize_component(item.name), _normalize_component(item.description))
        )
        technical = [item for result in results for item in result.technical_responses]
        data["technical_responses"] = _merge_exact(
            technical,
            lambda item: (
                _normalize_component(item.subject),
                _normalize_component(item.response),
                item.declared_status,
            ),
        )
        commercial = [item for result in results for item in result.commercial_responses]
        data["commercial_responses"] = _merge_exact(
            commercial,
            lambda item: (_normalize_component(item.subject), _normalize_component(item.response)),
        )
        deviations = [item for result in results for item in result.deviation_items]
        data["deviation_items"] = _merge_exact(
            deviations,
            lambda item: (
                _normalize_component(item.subject),
                _normalize_component(item.description),
                item.deviation_type,
            ),
        )
        submitted = [item for result in results for item in result.submitted_documents]
        data["submitted_documents"] = _merge_exact(
            submitted, lambda item: (_normalize_component(item.name), _normalize_component(item.description))
        )
        experience = [item for result in results for item in result.experience_items]
        data["experience_items"] = _merge_exact(
            experience,
            lambda item: (
                _normalize_component(item.project_name),
                _normalize_component(item.client),
                _normalize_component(item.contract_amount),
                _normalize_component(item.date_or_period),
                _normalize_component(item.description),
            ),
        )
        certifications = [item for result in results for item in result.certifications]
        data["certifications"] = _merge_exact(
            certifications,
            lambda item: (
                _normalize_component(item.name),
                _normalize_component(item.certificate_number),
                _normalize_component(item.validity),
                _normalize_component(item.description),
            ),
        )
        personnel = [item for result in results for item in result.personnel]
        data["personnel"] = _merge_exact(
            personnel,
            lambda item: (
                _normalize_component(item.name),
                _normalize_component(item.role),
                _normalize_component(item.qualification),
                _normalize_component(item.description),
            ),
        )
        equipment = [item for result in results for item in result.equipment]
        data["equipment"] = _merge_exact(
            equipment,
            lambda item: (
                _normalize_component(item.name),
                _normalize_component(item.quantity),
                _normalize_component(item.specification),
                _normalize_component(item.description),
            ),
        )
        data["technical_solution"] = _merge_text_items(
            [item for result in results for item in result.technical_solution]
        )
        data["service_commitments"] = _merge_text_items(
            [item for result in results for item in result.service_commitments]
        )
        warnings.extend(warning for result in results for warning in result.warnings)
        data["warnings"] = list(dict.fromkeys(warnings))
        return BidAnalysis.model_validate(data)
