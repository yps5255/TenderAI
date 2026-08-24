from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.db.base import Base
from backend.app.db.dependencies import get_database_session
from backend.app.db.session import create_session_factory, get_engine
from backend.app.main import app


@pytest.fixture
def client(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    factory = create_session_factory(database_url)

    def override_session():
        with factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = override_session
    with TestClient(app) as value:
        yield value
    app.dependency_overrides.pop(get_database_session, None)
    engine.dispose()
    get_engine.cache_clear()


def test_create_get_and_list_project(client: TestClient) -> None:
    response = client.post("/api/v1/projects", json={"name": "Project Alpha", "project_number": "TEST-001"})
    assert response.status_code == 201
    created = response.json()
    assert created["name"] == "Project Alpha"
    assert created["project_number"] == "TEST-001"
    assert created["created_at"].endswith("Z")

    response = client.get(f"/api/v1/projects/{created['id']}")
    assert response.status_code == 200
    assert response.json() == created

    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    assert response.json() == [created]


def test_unknown_project_returns_safe_404(client: TestClient) -> None:
    response = client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"detail": "Project not found."}


def test_project_name_must_not_be_empty(client: TestClient) -> None:
    assert client.post("/api/v1/projects", json={"name": ""}).status_code == 422
    assert client.post("/api/v1/projects", json={"name": "   "}).status_code == 422
