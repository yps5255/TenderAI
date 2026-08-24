from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from sqlalchemy import BigInteger, CheckConstraint, Enum as SqlEnum, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base
from ..types import UTCDateTime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectFileRole(str, Enum):
    TENDER = "tender"
    BID = "bid"
    ATTACHMENT = "attachment"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    DOCUMENT = "document"
    TECHNICAL_DRAWING = "technical_drawing"
    SCANNED_DOCUMENT = "scanned_document"
    OTHER = "other"


class ParseStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _enum_values(enum_class):
    return [member.value for member in enum_class]


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (CheckConstraint("length(trim(name)) > 0", name="ck_projects_name_nonempty"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_number: Mapped[str | None] = mapped_column(String(100))
    source_root: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now, nullable=False)

    files: Mapped[list[ProjectFile]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)


class ProjectFile(Base):
    __tablename__ = "project_files"
    __table_args__ = (UniqueConstraint("project_id", "relative_path", name="uq_project_files_project_relative_path"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    extension: Mapped[str] = mapped_column(String(50), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[ProjectFileRole] = mapped_column(SqlEnum(ProjectFileRole, values_callable=_enum_values, native_enum=False), default=ProjectFileRole.UNKNOWN, nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(SqlEnum(AssetType, values_callable=_enum_values, native_enum=False), default=AssetType.DOCUMENT, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    parse_status: Mapped[ParseStatus] = mapped_column(SqlEnum(ParseStatus, values_callable=_enum_values, native_enum=False), default=ParseStatus.PENDING, nullable=False)
    parse_error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now, nullable=False)

    project: Mapped[Project] = relationship(back_populates="files")
