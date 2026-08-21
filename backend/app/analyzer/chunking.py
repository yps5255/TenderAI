"""Deterministic, evidence-preserving document chunking."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ParsedDocument, Table
from .models import DocumentChunk

ANALYSIS_CHUNK_MAX_CHARS = 12000
ANALYSIS_CHUNK_OVERLAP_CHARS = 1000


@dataclass(frozen=True)
class _Fragment:
    content: str
    paragraph_index: int | None
    table_index: int | None
    page_number: int | None


def _page_label(page_number: int | None) -> str:
    return "null" if page_number is None else str(page_number)


def _split_text(text: str, limit: int) -> list[str]:
    """Split only when necessary, retaining every character in source order."""
    return [text[index : index + limit] for index in range(0, len(text), limit)] or [""]


def _paragraph_fragments(document: ParsedDocument, max_chars: int) -> list[_Fragment]:
    fragments: list[_Fragment] = []
    for paragraph in document.paragraphs:
        prefix = f"[P:{paragraph.index} PAGE:{_page_label(paragraph.page_number)}]\n"
        limit = max(1, max_chars - len(prefix))
        for part in _split_text(paragraph.text, limit):
            fragments.append(_Fragment(prefix + part, paragraph.index, None, paragraph.page_number))
    return fragments


def _table_fragments(table: Table, max_chars: int) -> list[_Fragment]:
    prefix = f"[T:{table.index} PAGE:{_page_label(table.page_number)}]\n"
    limit = max(1, max_chars - len(prefix))
    fragments: list[_Fragment] = []
    for row_number, row in enumerate(table.rows, start=1):
        row_text = f"ROW {row_number}: " + " | ".join(row)
        for part in _split_text(row_text, limit):
            fragments.append(_Fragment(prefix + part, None, table.index, table.page_number))
    if not table.rows:
        fragments.append(_Fragment(prefix + "ROW 0: ", None, table.index, table.page_number))
    return fragments


def _fragments(document: ParsedDocument, max_chars: int) -> list[_Fragment]:
    """Use parsed body content only; pages duplicate paragraph text for PDFs."""
    if document.content_order:
        paragraph_fragments = {paragraph.index: [] for paragraph in document.paragraphs}
        for fragment in _paragraph_fragments(document, max_chars):
            assert fragment.paragraph_index is not None
            paragraph_fragments[fragment.paragraph_index].append(fragment)
        table_fragments = {table.index: _table_fragments(table, max_chars) for table in document.tables}
        ordered: list[_Fragment] = []
        for reference in document.content_order:
            if reference.type == "paragraph":
                ordered.extend(paragraph_fragments.get(reference.index, []))
            else:
                ordered.extend(table_fragments.get(reference.index, []))
        return ordered
    fragments = _paragraph_fragments(document, max_chars)
    for table in document.tables:
        fragments.extend(_table_fragments(table, max_chars))
    return fragments


def _make_chunk(chunk_index: int, fragments: list[_Fragment]) -> DocumentChunk:
    return DocumentChunk(
        chunk_index=chunk_index,
        content="\n".join(fragment.content for fragment in fragments),
        paragraph_indices=list(dict.fromkeys(fragment.paragraph_index for fragment in fragments if fragment.paragraph_index is not None)),
        table_indices=list(dict.fromkeys(fragment.table_index for fragment in fragments if fragment.table_index is not None)),
        page_numbers=list(dict.fromkeys(fragment.page_number for fragment in fragments)),
    )


def build_document_chunks(
    document: ParsedDocument,
    max_chars: int = ANALYSIS_CHUNK_MAX_CHARS,
    overlap_chars: int = ANALYSIS_CHUNK_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Pack stable source fragments into finite, deterministic overlapping chunks."""
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("chunk overlap must be non-negative and smaller than max_chars")
    fragments = _fragments(document, max_chars)
    if not fragments:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(fragments):
        selected: list[_Fragment] = []
        size = 0
        cursor = start
        while cursor < len(fragments):
            additional = len(fragments[cursor].content) + (1 if selected else 0)
            if selected and size + additional > max_chars:
                break
            selected.append(fragments[cursor])
            size += additional
            cursor += 1
        chunks.append(_make_chunk(len(chunks), selected))
        if cursor >= len(fragments):
            break

        overlap_size = 0
        overlap_start = cursor
        while overlap_start > start:
            candidate = fragments[overlap_start - 1]
            candidate_size = len(candidate.content) + (1 if overlap_start - 1 < cursor - 1 else 0)
            if overlap_size + candidate_size > overlap_chars:
                break
            overlap_start -= 1
            overlap_size += candidate_size
        # Always advance by at least one fragment, even when a whole chunk fits
        # inside the requested overlap budget.
        start = max(start + 1, overlap_start)
    return chunks
