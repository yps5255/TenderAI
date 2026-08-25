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
from ..llm.exceptions import LLMStructuredOutputError
from ..models import EvidenceReference, ParsedDocument
from .models import (
    BidAnalysis,
    BidCapabilityChunkAnalysis,
    BidCommercialServiceChunkAnalysis,
    BidCoreDocumentsChunkAnalysis,
    BidSourceEvidence,
    BidTechnicalChunkAnalysis,
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
_SOURCE_REF = re.compile(r"([PT]):(0|[1-9]\d*)")
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
_GROUPS: tuple[tuple[str, type[BaseModel], tuple[str, ...]], ...] = (
    ("core_documents", BidCoreDocumentsChunkAnalysis, ("qualification_materials", "submitted_documents")),
    ("technical", BidTechnicalChunkAnalysis, ("technical_responses", "deviation_items", "technical_solution")),
    ("capability", BidCapabilityChunkAnalysis, ("experience_items", "certifications", "personnel", "equipment")),
    ("commercial_service", BidCommercialServiceChunkAnalysis, ("commercial_responses", "service_commitments")),
)
_GROUP_INSTRUCTIONS = {
    "core_documents": (
        "Extract only project identity, bidder identity, qualification materials actually provided by the bidder, "
        "and documents actually submitted. A tender requirement to provide a document is not evidence that the "
        "bidder submitted it."
    ),
    "technical": (
        "Extract only the bidder actual technical responses, explicit deviation statements, and technical solution. "
        "If a tender minimum is 100kW and the bidder states 120kW, extract the bidder response 120kW, not 100kW. "
        "declared_status and deviation_type must reflect only explicit bidder statements; no-deviation is not your "
        "compliance judgment."
    ),
    "capability": (
        "Extract only experience, certifications, personnel, and equipment explicitly claimed by the bidder. "
        "Do not turn minimum staffing, equipment, certification, or experience requirements into bidder capability."
    ),
    "commercial_service": (
        "Extract only the bidder own price, delivery commitment, validity period, commercial responses, and service "
        "commitments. Do not use a price ceiling or tender-required delivery or validity terms as bidder declarations."
    ),
}
_GROUP_SIGNALS: dict[str, tuple[str, ...]] = {
    "core_documents": (
        "项目名称", "项目编号", "招标编号", "投标编号", "投标人", "投标单位", "供应商",
        "资格材料", "资格证明", "资格文件", "营业执照", "许可证", "资质证书", "提交文件",
        "已提交", "随投标文件提交", "投标文件组成", "附件清单",
    ),
    "technical": (
        "技术响应", "技术要求", "技术参数", "技术规格", "规格参数", "性能参数", "额定", "功率",
        "压力参数", "压力要求", "压力等级", "流量", "温度", "控制系统", "plc", "触摸屏", "远程监控", "技术方案", "工艺方案",
        "偏离表", "技术偏离", "无偏离", "正偏离", "负偏离",
    ),
    "capability": (
        "类似业绩", "项目业绩", "合同业绩", "客户", "合同金额", "完成时间", "认证", "iso9001",
        "iso14001", "iso45001", "人员配置", "项目人员", "工程师", "技术人员", "焊工", "持证人员",
        "设备能力", "生产设备", "检测设备", "试验设备", "数控切割", "设备台数",
    ),
    "commercial_service": (
        "投标报价", "报价", "总价", "单价", "人民币", "万元", "交货期", "交货时间", "供货期", "工期",
        "投标有效期", "有效期", "付款", "付款条件", "商务响应", "商务偏离", "售后", "售后服务",
        "服务承诺", "质量保证", "质保", "保修", "维保",
    ),
}

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
        structured_output_retries: int = 1,
    ) -> None:
        if structured_output_retries < 0:
            raise ValueError("structured_output_retries must be non-negative")
        self.provider = provider
        self.chunk_builder = chunk_builder
        self.structured_output_retries = structured_output_retries

    @staticmethod
    def _relevance_signals(chunk: DocumentChunk, group_id: str) -> tuple[str, ...]:
        normalized = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", chunk.content)).casefold()
        return tuple(signal for signal in _GROUP_SIGNALS[group_id] if signal.casefold() in normalized)

    @classmethod
    def _should_run_group(cls, chunk: DocumentChunk, group_id: str) -> bool:
        target_signals = cls._relevance_signals(chunk, group_id)
        if target_signals:
            return True
        return not any(cls._relevance_signals(chunk, other_id) for other_id, _, _ in _GROUPS if other_id != group_id)

    @staticmethod
    def _messages(group_id: str, response_model: type[BaseModel], chunk: DocumentChunk) -> list[LLMMessage]:
        schema = response_model.model_json_schema()
        system = (
            "Extract only facts explicitly supported by the current bid-document chunk. Never infer or complete "
            "missing information. Return structured JSON only; absent scalars must be null and absent collections "
            "must be []. EVIDENCE SOURCE REFERENCE RULES: Every emitted collection item must contain evidence with "
            "a source_ref copied from the chunk locator. For [P:7 PAGE:null], return source_ref P:7. For "
            "[T:3 PAGE:5], return source_ref T:3. Never invent a source_ref. If no source_ref can be identified, do "
            "not emit the item. The quote must come from the same source_ref. Do not return page_number, "
            "paragraph_index, or table_index; TenderAI derives them from source_ref. Never perform tender-bid "
            "satisfaction or alignment judgment, scoring, award prediction, or rejection-risk judgment. "
            f"{_GROUP_INSTRUCTIONS[group_id]}"
        )
        user = (
            f"Extract group {group_id} from this bid chunk using its global locators. "
            "Return JSON matching this schema exactly:\n"
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

    def _resolve_evidence(
        self, evidence: list[BidSourceEvidence], chunk: DocumentChunk, document: ParsedDocument
    ) -> list[EvidenceReference]:
        paragraphs, tables = self._source_text(document)
        valid: list[EvidenceReference] = []
        for reference in evidence:
            match = _SOURCE_REF.fullmatch(reference.source_ref)
            if match is None:
                continue
            locator_type, raw_index = match.groups()
            index = int(raw_index)
            if locator_type == "P":
                if index not in chunk.paragraph_indices or index not in paragraphs:
                    continue
                source_text, page_number = paragraphs[index]
                paragraph_index, table_index = index, None
            else:
                if index not in chunk.table_indices or index not in tables:
                    continue
                source_text, page_number = tables[index]
                paragraph_index, table_index = None, index
            quote = reference.quote
            if quote is not None and _normalized_quote(quote) not in _normalized_quote(source_text):
                quote = None
            valid.append(EvidenceReference(
                page_number=page_number,
                paragraph_index=paragraph_index,
                table_index=table_index,
                quote=quote,
            ))
        return _dedupe_evidence(valid)

    def _validate_group_result(
        self,
        result: BaseModel,
        collection_fields: tuple[str, ...],
        chunk: DocumentChunk,
        document: ParsedDocument,
        warnings: list[str],
    ) -> BidAnalysis:
        data = result.model_dump(exclude=set(collection_fields))
        for field in collection_fields:
            validated_items = []
            for item in getattr(result, field):
                item_data = item.model_dump(exclude={"evidence"})
                item_data["evidence"] = self._resolve_evidence(item.evidence, chunk, document)
                if item_data["evidence"]:
                    validated_items.append(item_data)
                else:
                    warnings.append(f"item_without_valid_evidence_dropped:{field}")
            data[field] = validated_items
        return BidAnalysis.model_validate(data)

    def analyze(self, document: ParsedDocument) -> BidAnalysis:
        chunks = self.chunk_builder(document)
        if not chunks:
            return BidAnalysis(warnings=["document_has_no_extractable_content"])

        warnings: list[str] = [] if document.content_order else ["source_order_unavailable"]
        results: list[BidAnalysis] = []
        last_error: Exception | None = None
        for chunk in chunks:
            failed_groups = 0
            attempted_groups = 0
            for group_id, response_model, collection_fields in _GROUPS:
                if not self._should_run_group(chunk, group_id):
                    continue
                attempted_groups += 1
                messages = self._messages(group_id, response_model, chunk)
                try:
                    result = None
                    for attempt in range(self.structured_output_retries + 1):
                        try:
                            result = self.provider.generate(messages, response_model)
                            break
                        except LLMStructuredOutputError:
                            if attempt >= self.structured_output_retries:
                                raise
                    assert result is not None
                    results.append(
                        self._validate_group_result(result, collection_fields, chunk, document, warnings)
                    )
                except Exception as exc:
                    last_error = exc
                    failed_groups += 1
                    warnings.append(f"group_analysis_failed:{group_id}:{chunk.chunk_index}")
            if attempted_groups > 0 and failed_groups == attempted_groups:
                warnings.append(f"chunk_analysis_failed:{chunk.chunk_index}")
        if not results:
            raise BidAnalyzerError("All document chunks failed analysis.") from last_error
        return self._merge(results, warnings)

    @staticmethod
    def _merge(results: list[BidAnalysis], warnings: list[str]) -> BidAnalysis:
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
