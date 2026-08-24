from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Project


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
