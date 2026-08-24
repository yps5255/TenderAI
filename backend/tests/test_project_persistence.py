from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import AssetType, ParseStatus, Project, ProjectFile, ProjectFileRole
from backend.app.db.session import create_session_factory, get_engine
from backend.app.repositories import ProjectRepository


@pytest.fixture
def session(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    with create_session_factory(database_url)() as value:
        yield value
    engine.dispose()
    get_engine.cache_clear()


def test_repository_create_get_and_list_projects(session: Session) -> None:
    repository = ProjectRepository(session)
    alpha = repository.add(Project(name="Project Alpha", project_number="TEST-001"))
    beta = repository.add(Project(name="Project Beta"))
    session.commit()

    assert repository.get(alpha.id) is alpha
    assert {project.id for project in repository.list()} == {alpha.id, beta.id}
    assert alpha.created_at.tzinfo is not None


def test_repository_returns_none_for_unknown_project(session: Session) -> None:
    assert ProjectRepository(session).get("00000000-0000-0000-0000-000000000000") is None


def test_project_file_fk_is_enforced(session: Session) -> None:
    session.add(ProjectFile(project_id="missing", filename="bid.pdf", relative_path="bid.pdf", extension=".pdf", size_bytes=10))
    with pytest.raises(IntegrityError):
        session.commit()


def test_project_file_relative_path_is_unique_per_project(session: Session) -> None:
    project = Project(name="Project Alpha")
    session.add(project)
    session.flush()
    session.add_all([
        ProjectFile(project_id=project.id, filename="one.pdf", relative_path="docs/file.pdf", extension=".pdf", size_bytes=10),
        ProjectFile(project_id=project.id, filename="two.pdf", relative_path="docs/file.pdf", extension=".pdf", size_bytes=20),
    ])
    with pytest.raises(IntegrityError):
        session.commit()


def test_project_delete_cascades_to_files(session: Session) -> None:
    project = Project(name="Project Alpha")
    project.files.append(ProjectFile(filename="tender.pdf", relative_path="tender.pdf", extension=".pdf", size_bytes=10))
    session.add(project)
    session.commit()

    session.delete(project)
    session.commit()

    assert session.scalar(select(func.count()).select_from(ProjectFile)) == 0


def test_project_file_enum_round_trip_and_parse_status_default(session: Session) -> None:
    project = Project(name="Project Alpha")
    project.files.append(ProjectFile(
        filename="drawing.dwg",
        relative_path="drawings/drawing.dwg",
        extension=".dwg",
        size_bytes=42,
        role=ProjectFileRole.ATTACHMENT,
        asset_type=AssetType.TECHNICAL_DRAWING,
    ))
    session.add(project)
    session.commit()
    session.expire_all()

    stored = session.scalar(select(ProjectFile))
    assert stored is not None
    assert stored.role is ProjectFileRole.ATTACHMENT
    assert stored.asset_type is AssetType.TECHNICAL_DRAWING
    assert stored.parse_status is ParseStatus.PENDING


def test_temp_sqlite_is_isolated(tmp_path: Path) -> None:
    first_url = f"sqlite:///{(tmp_path / 'first.db').as_posix()}"
    second_url = f"sqlite:///{(tmp_path / 'second.db').as_posix()}"
    engines = [get_engine(url) for url in (first_url, second_url)]
    for engine in engines:
        Base.metadata.create_all(engine)
    with create_session_factory(first_url)() as first:
        first.add(Project(name="Project Alpha"))
        first.commit()
    with create_session_factory(second_url)() as second:
        assert second.scalar(select(func.count()).select_from(Project)) == 0
    for engine in engines:
        engine.dispose()
    get_engine.cache_clear()
