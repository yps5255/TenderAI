from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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


class ContentBlockReference(BaseModel):
    """Reference to an existing paragraph or table in original body order."""

    type: Literal["paragraph", "table"]
    index: int


class ParsedDocument(BaseModel):
    """Format-independent result returned by TenderAI document parsers."""

    filename: str
    file_type: str
    page_count: int | None = None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    content_order: list[ContentBlockReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DocumentRole(str, Enum):
    TENDER = "tender"
    BID = "bid"
    TECHNICAL_DRAWING = "technical_drawing"
    ATTACHMENT = "attachment"
    UNKNOWN = "unknown"


class ProjectDocument(BaseModel):
    filename: str
    relative_path: str
    file_type: str
    role: DocumentRole
    parse_success: bool
    paragraph_count: int = 0
    table_count: int = 0
    warnings_count: int = 0
    error_type: str | None = None
    error_message_short: str | None = None


class ProjectScan(BaseModel):
    project_id: str
    source_folder: str
    files: list[ProjectDocument] = Field(default_factory=list)
