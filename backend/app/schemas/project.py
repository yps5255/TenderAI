from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..db.models import AssetType, ParseStatus, ProjectFileRole


class ProjectCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    project_number: str | None = Field(default=None, max_length=100)
    source_root: str | None = Field(default=None, max_length=2048)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    project_number: str | None
    source_root: str | None
    created_at: datetime
    updated_at: datetime


class ProjectFileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    filename: str
    relative_path: str
    extension: str
    size_bytes: int
    role: ProjectFileRole
    asset_type: AssetType
    sha256: str | None
    parse_status: ParseStatus
    parse_error_code: str | None
    created_at: datetime
    updated_at: datetime


class ProjectImportRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    source_root: str = Field(min_length=1, max_length=2048)


class ProjectImportError(BaseModel):
    relative_path: str | None = None
    error_code: str


class ProjectImportResult(BaseModel):
    project_id: str
    scanned_files: int
    supported_files: int
    created: int
    updated: int
    unchanged: int
    ignored: int
    unsupported: int
    missing_existing: int
    errors: list[ProjectImportError] = Field(default_factory=list)
