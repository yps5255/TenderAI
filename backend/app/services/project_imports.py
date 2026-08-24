from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from ..db.models import ParseStatus, ProjectFile
from ..project_import import scan_project_directory
from ..repositories import ProjectRepository
from ..schemas import ProjectImportResult
from .projects import ProjectNotFoundError


class SourceRootError(ValueError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class ProjectImportService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = ProjectRepository(session)

    def import_directory(self, project_id: str, source_root: str) -> ProjectImportResult:
        project = self.repository.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)

        try:
            requested_root = Path(source_root).expanduser()
            if not requested_root.exists():
                raise SourceRootError("source_root_not_found")
            if not requested_root.is_dir():
                raise SourceRootError("source_root_not_directory")
            if requested_root.is_symlink() or (
                hasattr(requested_root, "is_junction") and requested_root.is_junction()
            ):
                raise SourceRootError("source_root_link_not_allowed")
            root = requested_root.resolve(strict=True)
        except SourceRootError:
            raise
        except (OSError, ValueError) as exc:
            raise SourceRootError("source_root_invalid") from exc
        scan = scan_project_directory(root)
        existing = self.repository.get_files_by_relative_path(project_id)
        seen_paths = set(scan.supported_paths)
        created = updated = unchanged = 0

        for scanned in scan.files:
            stored = existing.get(scanned.relative_path)
            if stored is None:
                self.repository.add_file(ProjectFile(
                    project_id=project_id,
                    filename=scanned.filename,
                    relative_path=scanned.relative_path,
                    extension=scanned.extension,
                    size_bytes=scanned.size_bytes,
                    role=scanned.role,
                    asset_type=scanned.asset_type,
                    sha256=scanned.sha256,
                ))
                created += 1
            elif stored.size_bytes == scanned.size_bytes and stored.sha256 == scanned.sha256:
                unchanged += 1
            else:
                stored.filename = scanned.filename
                stored.extension = scanned.extension
                stored.size_bytes = scanned.size_bytes
                stored.role = scanned.role
                stored.asset_type = scanned.asset_type
                stored.sha256 = scanned.sha256
                stored.parse_status = ParseStatus.PENDING
                stored.parse_error_code = None
                updated += 1

        project.source_root = str(root)
        missing_existing = len(set(existing) - seen_paths)
        self.session.commit()
        return ProjectImportResult(
            project_id=project_id,
            scanned_files=scan.scanned_files,
            supported_files=len(scan.supported_paths),
            created=created,
            updated=updated,
            unchanged=unchanged,
            ignored=scan.ignored,
            unsupported=scan.unsupported,
            missing_existing=missing_existing,
            errors=[{"relative_path": relative_path, "error_code": error_code} for relative_path, error_code in scan.errors],
        )

    def list_files(self, project_id: str) -> list[ProjectFile]:
        if self.repository.get(project_id) is None:
            raise ProjectNotFoundError(project_id)
        return self.repository.list_files(project_id)
