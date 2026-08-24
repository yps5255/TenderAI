from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.dependencies import get_database_session
from ..schemas import ProjectCreate, ProjectFileRead, ProjectImportRequest, ProjectImportResult, ProjectRead
from ..services import ProjectImportService, ProjectNotFoundError, ProjectService, SourceRootError

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


@router.post("/{project_id}/import", response_model=ProjectImportResult)
def import_project_directory(
    project_id: str,
    data: ProjectImportRequest,
    session: Annotated[Session, Depends(get_database_session)],
) -> ProjectImportResult:
    try:
        return ProjectImportService(session).import_directory(project_id, data.source_root)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.") from exc
    except SourceRootError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.error_code) from exc


@router.get("/{project_id}/files", response_model=list[ProjectFileRead])
def list_project_files(project_id: str, session: Annotated[Session, Depends(get_database_session)]) -> list[ProjectFileRead]:
    try:
        return ProjectImportService(session).list_files(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.") from exc
