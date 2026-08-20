from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from ..models import Page, Paragraph, ParsedDocument, Table
from .pdf_parser import DocumentParseError


def _iter_body_items(document: DocumentType):
    """Yield paragraphs and tables in their original document-body order."""
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)


def parse_docx(path: Path, filename: str) -> ParsedDocument:
    """Extract paragraphs and tables from a DOCX document."""
    try:
        document = Document(path)
    except (BadZipFile, OSError, ValueError, KeyError, TypeError) as exc:
        raise DocumentParseError("The DOCX file is invalid or unreadable.") from exc

    paragraphs: list[Paragraph] = []
    tables: list[Table] = []
    body_text: list[str] = []

    for item in _iter_body_items(document):
        if isinstance(item, DocxParagraph):
            text = item.text.strip()
            if text:
                paragraphs.append(Paragraph(text=text, page_number=None, index=len(paragraphs)))
                body_text.append(text)
        else:
            rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            tables.append(Table(page_number=None, index=len(tables), rows=rows))
            body_text.extend("\t".join(row) for row in rows)

    metadata: dict[str, Any] = {
        "author": document.core_properties.author,
        "title": document.core_properties.title,
        "subject": document.core_properties.subject,
        "keywords": document.core_properties.keywords,
    }
    metadata = {key: value for key, value in metadata.items() if value}
    warnings: list[str] = []
    if not paragraphs and not tables:
        warnings.append("No non-empty paragraphs or tables were found in this DOCX file.")

    return ParsedDocument(
        filename=filename,
        file_type="docx",
        page_count=None,
        paragraphs=paragraphs,
        tables=tables,
        pages=[Page(page_number=None, text="\n".join(body_text))],
        metadata=metadata,
        warnings=warnings,
    )
