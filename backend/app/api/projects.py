from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.dependencies import get_database_session
from ..schemas import ProjectCreate, ProjectRead
from ..services import ProjectNotFoundError, ProjectService

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(data: ProjectCreate, session: Annotated[Session, Depends(get_database_session)]) -> ProjectRead:
    return ProjectService(session).create(data)


@router.get("", response_model=list[ProjectRead])
def list_projects(session: Annotated[Session, Depends(get_database_session)]) -> list[ProjectRead]:
    return ProjectService(session).list()


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: str, session: Annotated[Session, Depends(get_database_session)]) -> ProjectRead:
    try:
        return ProjectService(session).get(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.") from exc
