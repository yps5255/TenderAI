from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    """A non-empty textual paragraph extracted from a document."""

    text: str
    page_number: int | None = None
    index: int


class Table(BaseModel):
    """A table represented as rows of cell strings."""

    page_number: int | None = None
    index: int
    rows: list[list[str]]


class Page(BaseModel):
    """Text extracted from one source page (or the DOCX document body)."""

    page_number: int | None = None
    text: str


class ParsedDocument(BaseModel):
    """Format-independent result returned by TenderAI document parsers."""

    filename: str
    file_type: str
    page_count: int | None = None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
