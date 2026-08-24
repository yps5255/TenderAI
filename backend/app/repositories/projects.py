from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Project, ProjectFile


class ProjectRepository:
    """Persistence operations for projects; transaction ownership stays with the service."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, project: Project) -> Project:
        self.session.add(project)
        self.session.flush()
        return project

    def get(self, project_id: str) -> Project | None:
        return self.session.get(Project, project_id)

    def list(self) -> list[Project]:
        return list(self.session.scalars(select(Project).order_by(Project.created_at, Project.id)))

    def get_files_by_relative_path(self, project_id: str) -> dict[str, ProjectFile]:
        files = self.session.scalars(select(ProjectFile).where(ProjectFile.project_id == project_id))
        return {project_file.relative_path: project_file for project_file in files}

    def add_file(self, project_file: ProjectFile) -> ProjectFile:
        self.session.add(project_file)
        return project_file

    def list_files(self, project_id: str) -> list[ProjectFile]:
        statement = select(ProjectFile).where(ProjectFile.project_id == project_id).order_by(ProjectFile.relative_path)
        return list(self.session.scalars(statement))
