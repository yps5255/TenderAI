from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import AssetType, ParseStatus, Project, ProjectFile, ProjectFileRole
from backend.app.db.session import create_session_factory, get_engine
from backend.app.project_import import hash_file_sha256, scan_project_directory
from backend.app.project_import import scanner as scanner_module
from backend.app.services import ProjectImportService, ProjectNotFoundError, SourceRootError


@pytest.fixture
def session(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'import.db').as_posix()}"
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    with create_session_factory(database_url)() as value:
        yield value
    engine.dispose()
    get_engine.cache_clear()


@pytest.fixture
def project(session: Session) -> Project:
    value = Project(name="Project Alpha", project_number="TEST-001")
    session.add(value)
    session.commit()
    return value


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Project Alpha"
    files = {
        "招标资料/招标文件.docx": b"tender-docx",
        "招标资料/技术要求.doc": b"technical-requirements",
        "招标资料/设备招标图.pdf": b"drawing",
        "投标资料/商务投标文件.docx": b"commercial-bid",
        "投标资料/技术方案.docx": b"technical-plan",
        "投标资料/响应文件.pdf": b"response",
        "附件/附件1.pdf": b"attachment",
        "其他/一般资料.pdf": b"unknown",
        "~$招标文件.docx": b"temporary",
        "README.txt": b"unsupported",
        ".hidden.pdf": b"hidden",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_recursive_scan_counts_ignored_and_unsupported(synthetic_tree: Path) -> None:
    scan = scan_project_directory(synthetic_tree)

    assert scan.scanned_files == 10
    assert len(scan.files) == 8
    assert scan.ignored == 2
    assert scan.unsupported == 1
    assert scan.errors == []


def test_role_and_asset_classification_is_contextual_and_independent(synthetic_tree: Path) -> None:
    files = {item.relative_path: item for item in scan_project_directory(synthetic_tree).files}

    assert files["招标资料/技术要求.doc"].role is ProjectFileRole.TENDER
    assert files["投标资料/技术方案.docx"].role is ProjectFileRole.BID
    assert files["投标资料/技术方案.docx"].asset_type is AssetType.DOCUMENT
    assert files["附件/附件1.pdf"].role is ProjectFileRole.ATTACHMENT
    assert files["其他/一般资料.pdf"].role is ProjectFileRole.UNKNOWN
    drawing = files["招标资料/设备招标图.pdf"]
    assert drawing.role is ProjectFileRole.TENDER
    assert drawing.asset_type is AssetType.TECHNICAL_DRAWING


def test_scan_preserves_relative_metadata_and_streaming_hash(synthetic_tree: Path) -> None:
    stored = next(item for item in scan_project_directory(synthetic_tree).files if item.filename == "招标文件.docx")
    expected = (synthetic_tree / "招标资料" / "招标文件.docx").read_bytes()

    assert stored.relative_path == "招标资料/招标文件.docx"
    assert stored.extension == ".docx"
    assert stored.size_bytes == len(expected)
    assert stored.sha256 == hashlib.sha256(expected).hexdigest()


def test_hash_file_sha256_handles_multiple_chunks(tmp_path: Path) -> None:
    content = b"a" * (1024 * 1024 + 17)
    path = tmp_path / "large.pdf"
    path.write_bytes(content)

    assert hash_file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_first_import_creates_records_and_updates_source_root(
    session: Session, project: Project, synthetic_tree: Path
) -> None:
    result = ProjectImportService(session).import_directory(project.id, str(synthetic_tree))

    assert result.created == 8
    assert result.updated == result.unchanged == result.missing_existing == 0
    assert session.scalar(select(func.count()).select_from(ProjectFile)) == 8
    assert project.source_root == str(synthetic_tree.resolve())


def test_second_identical_import_is_unchanged_without_unique_error(
    session: Session, project: Project, synthetic_tree: Path
) -> None:
    service = ProjectImportService(session)
    service.import_directory(project.id, str(synthetic_tree))
    first_ids = {item.relative_path: item.id for item in service.list_files(project.id)}

    result = service.import_directory(project.id, str(synthetic_tree))

    assert result.created == result.updated == 0
    assert result.unchanged == 8
    assert {item.relative_path: item.id for item in service.list_files(project.id)} == first_ids


def test_changed_file_updates_same_record_and_resets_parse_state(
    session: Session, project: Project, synthetic_tree: Path
) -> None:
    service = ProjectImportService(session)
    service.import_directory(project.id, str(synthetic_tree))
    relative_path = "招标资料/招标文件.docx"
    stored = {item.relative_path: item for item in service.list_files(project.id)}[relative_path]
    original_id = stored.id
    stored.parse_status = ParseStatus.FAILED
    stored.parse_error_code = "old_error"
    session.commit()
    (synthetic_tree / relative_path).write_bytes(b"changed-content")

    result = service.import_directory(project.id, str(synthetic_tree))
    updated = {item.relative_path: item for item in service.list_files(project.id)}[relative_path]

    assert result.updated == 1
    assert updated.id == original_id
    assert updated.sha256 == hashlib.sha256(b"changed-content").hexdigest()
    assert updated.parse_status is ParseStatus.PENDING
    assert updated.parse_error_code is None


def test_new_file_is_created_and_missing_existing_is_not_deleted(
    session: Session, project: Project, synthetic_tree: Path
) -> None:
    service = ProjectImportService(session)
    service.import_directory(project.id, str(synthetic_tree))
    missing_path = "其他/一般资料.pdf"
    missing_id = {item.relative_path: item.id for item in service.list_files(project.id)}[missing_path]
    (synthetic_tree / missing_path).unlink()
    new_path = synthetic_tree / "新增" / "采购文件.pdf"
    new_path.parent.mkdir()
    new_path.write_bytes(b"new")

    result = service.import_directory(project.id, str(synthetic_tree))

    assert result.created == 1
    assert result.missing_existing == 1
    assert session.get(ProjectFile, missing_id) is not None


def test_unreadable_supported_file_reports_error_without_false_missing(
    session: Session, project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    path = root / "招标文件.pdf"
    path.write_bytes(b"content")
    existing = ProjectFile(
        project_id=project.id,
        filename=path.name,
        relative_path=path.name,
        extension=".pdf",
        size_bytes=7,
    )
    session.add(existing)
    session.commit()
    monkeypatch.setattr(scanner_module, "hash_file_sha256", lambda _path: (_ for _ in ()).throw(OSError()))

    result = ProjectImportService(session).import_directory(project.id, str(root))

    assert result.supported_files == 1
    assert result.missing_existing == 0
    assert result.errors[0].error_code == "file_unreadable"


@pytest.mark.parametrize(
    ("root_factory", "error_code"),
    [
        (lambda path: path / "missing", "source_root_not_found"),
        (lambda path: _create_file(path / "source.pdf"), "source_root_not_directory"),
    ],
)
def test_invalid_source_root(session: Session, project: Project, tmp_path: Path, root_factory, error_code: str) -> None:
    source_root = root_factory(tmp_path)
    with pytest.raises(SourceRootError, match=error_code):
        ProjectImportService(session).import_directory(project.id, str(source_root))


def _create_file(path: Path) -> Path:
    path.write_bytes(b"file")
    return path


def test_unknown_project_is_rejected(session: Session, synthetic_tree: Path) -> None:
    with pytest.raises(ProjectNotFoundError):
        ProjectImportService(session).import_directory("missing", str(synthetic_tree))


def test_symlink_or_junction_is_not_followed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    link = root / "linked"
    link.mkdir(parents=True)
    (link / "招标文件.pdf").write_bytes(b"outside")
    monkeypatch.setattr(scanner_module, "_is_link_or_junction", lambda path, _entry: path.name == "linked")

    scan = scan_project_directory(root)

    assert scan.files == []
    assert scan.ignored == 1
