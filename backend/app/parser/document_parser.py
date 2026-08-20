from __future__ import annotations

from pathlib import Path

from ..models import ParsedDocument
from .docx_parser import parse_docx
from .pdf_parser import DocumentParseError, parse_pdf

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def parse_document(path: Path, filename: str | None = None) -> ParsedDocument:
    """Parse a supported local document into TenderAI's common model."""
    source_name = filename or path.name
    extension = Path(source_name).suffix.lower()
    if extension == ".pdf":
        return parse_pdf(path, source_name)
    if extension == ".docx":
        return parse_docx(path, source_name)
    raise DocumentParseError("Only .pdf and .docx files are supported.")
