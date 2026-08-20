from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from ..models import Page, Paragraph, ParsedDocument, Table


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed safely."""


def parse_pdf(path: Path, filename: str) -> ParsedDocument:
    """Extract text and detectable tables from a text-based PDF."""
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError, OSError) as exc:
        raise DocumentParseError("The PDF file is invalid or unreadable.") from exc

    metadata: dict[str, Any] = {
        key: value for key, value in document.metadata.items() if value is not None
    }

    paragraphs: list[Paragraph] = []
    tables: list[Table] = []
    pages: list[Page] = []
    warnings: list[str] = []

    try:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append(Page(page_number=page_index, text=text))
            if not text:
                warnings.append(f"Page {page_index} contains no extractable text.")
            else:
                for block in page.get_text("blocks"):
                    block_text = str(block[4]).strip()
                    if block_text:
                        paragraphs.append(
                            Paragraph(
                                text=block_text,
                                page_number=page_index,
                                index=len(paragraphs),
                            )
                        )

            try:
                found_tables = page.find_tables()
                for found_table in found_tables.tables:
                    rows = [
                        ["" if cell is None else str(cell) for cell in row]
                        for row in found_table.extract()
                    ]
                    if rows:
                        tables.append(
                            Table(
                                page_number=page_index,
                                index=len(tables),
                                rows=rows,
                            )
                        )
            except (AttributeError, RuntimeError, ValueError):
                warnings.append(
                    f"Table extraction was unavailable for page {page_index}."
                )
    finally:
        document.close()

    if not paragraphs:
        warnings.extend(["pdf_has_little_or_no_extractable_text", "possible_scanned_or_drawing_pdf"])

    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        page_count=len(pages),
        paragraphs=paragraphs,
        tables=tables,
        pages=pages,
        metadata=metadata,
        warnings=warnings,
    )
