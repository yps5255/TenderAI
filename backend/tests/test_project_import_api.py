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
    database_url = f"sqlite:///{(tmp_path / 'api-import.db').as_posix()}"
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


def _create_project(client: TestClient) -> str:
    return client.post("/api/v1/projects", json={"name": "Project Alpha"}).json()["id"]


def test_import_and_get_project_files(client: TestClient, tmp_path: Path) -> None:
    project_id = _create_project(client)
    source_root = tmp_path / "source"
    path = source_root / "招标资料" / "设备招标图.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"drawing")

    response = client.post(f"/api/v1/projects/{project_id}/import", json={"source_root": str(source_root)})

    assert response.status_code == 200
    assert response.json()["errors"] == []
    assert response.json()["created"] == 1
    files = client.get(f"/api/v1/projects/{project_id}/files")
    assert files.status_code == 200
    assert files.json()[0]["relative_path"] == "招标资料/设备招标图.pdf"
    assert files.json()[0]["role"] == "tender"
    assert files.json()[0]["asset_type"] == "technical_drawing"


@pytest.mark.parametrize("suffix", ["missing", "file.pdf"])
def test_import_rejects_invalid_source_root(client: TestClient, tmp_path: Path, suffix: str) -> None:
    project_id = _create_project(client)
    source_root = tmp_path / suffix
    if suffix.endswith(".pdf"):
        source_root.write_bytes(b"file")

    response = client.post(f"/api/v1/projects/{project_id}/import", json={"source_root": str(source_root)})

    assert response.status_code == 422
    assert response.json()["detail"] in {"source_root_not_found", "source_root_not_directory"}


def test_import_and_file_list_return_404_for_unknown_project(client: TestClient, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    imported = client.post("/api/v1/projects/missing/import", json={"source_root": str(source_root)})
    listed = client.get("/api/v1/projects/missing/files")

    assert imported.status_code == 404
    assert listed.status_code == 404
