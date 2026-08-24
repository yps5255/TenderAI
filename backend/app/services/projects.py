from sqlalchemy.orm import Session

from ..db.models import Project
from ..repositories import ProjectRepository
from ..schemas import ProjectCreate


class ProjectNotFoundError(Exception):
    pass


class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = ProjectRepository(session)

    def create(self, data: ProjectCreate) -> Project:
        project = Project(**data.model_dump())
        self.repository.add(project)
        self.session.commit()
        return project

    def get(self, project_id: str) -> Project:
        project = self.repository.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        return project

    def list(self) -> list[Project]:
        return self.repository.list()
