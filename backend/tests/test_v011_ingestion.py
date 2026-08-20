from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pymupdf
import pytest
from docx import Document
from fastapi import HTTPException

from backend.app.main import MAX_UPLOAD_SIZE, stream_upload_to_path
from backend.app.models import DocumentRole
from backend.app.parser import document_parser
from backend.app.parser.legacy_doc_parser import LegacyDocConversionProcessFailed, LegacyDocConversionTimeout, LegacyDocOutputMissing, LegacyDocConverterUnavailable, LibreOfficeDocConverter, _cleanup_directory, parse_legacy_doc
from backend.app.parser.project_scanner import classify_document_role, scan_project


def make_docx(path: Path, text: str = "synthetic paragraph") -> Path:
    document = Document()
    document.add_paragraph(text)
    document.save(path)
    return path


def test_upload_limit_is_500_mib() -> None:
    assert MAX_UPLOAD_SIZE == 500 * 1024 * 1024 == 524288000


def test_stream_upload_allows_exact_limit_and_reads_in_chunks(tmp_path: Path) -> None:
    class FakeUpload:
        def __init__(self) -> None:
            self.chunks = [b"abc", b"de"]
            self.read_calls = 0

        async def read(self, _size: int) -> bytes:
            self.read_calls += 1
            return self.chunks.pop(0) if self.chunks else b""

    upload = FakeUpload()
    assert asyncio.run(stream_upload_to_path(upload, tmp_path / "upload.bin", max_size=5)) == 5
    assert (tmp_path / "upload.bin").read_bytes() == b"abcde"
    assert upload.read_calls == 3


def test_stream_upload_rejects_over_limit_without_reading_remaining_chunks(tmp_path: Path) -> None:
    class FakeUpload:
        def __init__(self) -> None:
            self.chunks = [b"abc", b"def", b"not-read"]
            self.read_calls = 0

        async def read(self, _size: int) -> bytes:
            self.read_calls += 1
            return self.chunks.pop(0)

    upload = FakeUpload()
    with pytest.raises(HTTPException) as error:
        asyncio.run(stream_upload_to_path(upload, tmp_path / "upload.bin", max_size=5))
    assert error.value.status_code == 413
    assert upload.read_calls == 2


def test_doc_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(document_parser, "parse_legacy_doc", lambda path, filename: "legacy-result")
    assert document_parser.parse_document(source) == "legacy-result"


def test_libreoffice_unavailable_is_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda _: None)
    monkeypatch.setattr(LibreOfficeDocConverter, "default_windows_paths", ())
    with pytest.raises(LegacyDocConverterUnavailable, match="legacy_doc_converter_unavailable"):
        LibreOfficeDocConverter().convert(tmp_path / "source.doc", tmp_path, tmp_path / "profile")


def test_configured_libreoffice_path_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "soffice.exe"
    executable.touch()
    monkeypatch.setenv("LIBREOFFICE_PATH", f'  "{executable}"  ')
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda _: "wrong.exe")
    assert LibreOfficeDocConverter()._find_executable() == str(executable)


def test_invalid_configured_path_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBREOFFICE_PATH", "missing-soffice.exe")
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda _: "fallback.exe")
    with pytest.raises(LegacyDocConverterUnavailable, match="configured_libreoffice_not_found"):
        LibreOfficeDocConverter()._find_executable()


def test_path_lookup_uses_which_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda name: "found.exe" if name == "soffice" else None)
    assert LibreOfficeDocConverter()._find_executable() == "found.exe"


def test_windows_default_location_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "soffice.exe"
    executable.touch()
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda _: None)
    monkeypatch.setattr(LibreOfficeDocConverter, "default_windows_paths", (executable,))
    assert LibreOfficeDocConverter()._find_executable() == str(executable)


def test_legacy_conversion_uses_and_cleans_temp_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.doc"
    source.write_bytes(b"synthetic")
    seen: list[Path] = []

    class FakeConverter:
        def convert(self, _source: Path, output_dir: Path, profile_dir: Path) -> Path:
            seen.append(output_dir)
            assert profile_dir.parent == output_dir.parent
            return make_docx(output_dir / "source.docx")

    parsed = parse_legacy_doc(source, source.name, FakeConverter())
    assert parsed.file_type == "doc"
    assert seen and not seen[0].exists()


def test_each_legacy_conversion_uses_unique_profile(tmp_path: Path) -> None:
    source = tmp_path / "source.doc"
    source.write_bytes(b"synthetic")
    profiles: list[Path] = []

    class FakeConverter:
        def convert(self, _source: Path, output_dir: Path, profile_dir: Path) -> Path:
            profiles.append(profile_dir)
            return make_docx(output_dir / "source.docx")

    parse_legacy_doc(source, source.name, FakeConverter())
    parse_legacy_doc(source, source.name, FakeConverter())
    assert len(profiles) == 2 and profiles[0] != profiles[1]
    assert all(profile.name == "profile" and not profile.exists() for profile in profiles)


def test_converter_passes_profile_file_uri_and_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.doc"; source.write_bytes(b"x")
    output = tmp_path / "output"; output.mkdir()
    profile = tmp_path / "profile"; profile.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setenv("LIBREOFFICE_PATH", str(tmp_path / "soffice.exe")); (tmp_path / "soffice.exe").touch()
    def fake_run(args: list[str], **kwargs: object) -> object:
        calls.append(args); (output / "source.docx").touch()
        assert kwargs["timeout"] == 180
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.subprocess.run", fake_run)
    LibreOfficeDocConverter().convert(source, output, profile)
    assert f"-env:UserInstallation={profile.resolve().as_uri()}" in calls[0]


def test_explicit_soffice_exe_is_not_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "soffice.exe"; executable.touch()
    monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda _: "soffice.com")
    assert LibreOfficeDocConverter()._find_executable() == str(executable)


def test_auto_discovery_prefers_soffice_com(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.which", lambda name: name if name in {"soffice.com", "soffice"} else None)
    assert LibreOfficeDocConverter()._find_executable() == "soffice.com"


def test_conversion_timeout_has_distinct_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "a.doc"; source.write_bytes(b"x"); output = tmp_path / "out"; output.mkdir(); profile = tmp_path / "p"; profile.mkdir()
    executable = tmp_path / "soffice.exe"; executable.touch(); monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    def timeout(*_args: object, **kwargs: object) -> object:
        assert kwargs["timeout"] == 180; raise subprocess.TimeoutExpired("soffice", 180)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.subprocess.run", timeout)
    with pytest.raises(LegacyDocConversionTimeout, match="legacy_doc_conversion_timeout"):
        LibreOfficeDocConverter().convert(source, output, profile)


def test_nonzero_process_does_not_poll_for_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "a.doc"; source.write_bytes(b"x"); output = tmp_path / "out"; output.mkdir(); profile = tmp_path / "p"; profile.mkdir(); executable = tmp_path / "soffice.exe"; executable.touch(); monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.subprocess.run", lambda *_a, **_k: type("Result", (), {"returncode": 7})())
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.sleep", lambda _x: pytest.fail("should not poll"))
    with pytest.raises(LegacyDocConversionProcessFailed, match="code 7"):
        LibreOfficeDocConverter().convert(source, output, profile)


def test_missing_output_has_bounded_polling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "a.doc"; source.write_bytes(b"x"); output = tmp_path / "out"; output.mkdir(); profile = tmp_path / "p"; profile.mkdir(); executable = tmp_path / "soffice.exe"; executable.touch(); monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    converter = LibreOfficeDocConverter(); converter.OUTPUT_POLL_TIMEOUT_SECONDS = 1; converter.OUTPUT_POLL_INTERVAL_SECONDS = 0.5
    values = iter([0, 0, 2]); sleeps: list[float] = []
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.subprocess.run", lambda *_a, **_k: type("Result", (), {"returncode": 0})())
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.monotonic", lambda: next(values))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.sleep", lambda value: sleeps.append(value))
    with pytest.raises(LegacyDocOutputMissing): converter.convert(source, output, profile)
    assert sleeps == [0.5]


def test_delayed_output_is_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "a.doc"; source.write_bytes(b"x"); output = tmp_path / "out"; output.mkdir(); profile = tmp_path / "p"; profile.mkdir(); executable = tmp_path / "soffice.exe"; executable.touch(); monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.subprocess.run", lambda *_a, **_k: type("Result", (), {"returncode": 0})())
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.monotonic", lambda: 0)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.sleep", lambda _x: (output / "a.docx").touch())
    assert LibreOfficeDocConverter().convert(source, output, profile).is_file()


def test_cleanup_retries_and_original_error_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts = [PermissionError(), PermissionError(), None]
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.rmtree", lambda _p: (_ for _ in ()).throw(attempts.pop(0)) if attempts[0] is not None else None)
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.sleep", lambda _x: None)
    assert _cleanup_directory(tmp_path) is True


def test_cleanup_failure_does_not_mask_original_conversion_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.doc"
    source.write_bytes(b"synthetic")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    sleep_calls: list[float] = []

    class TimeoutConverter:
        def convert(self, _source: Path, _output: Path, _profile: Path) -> Path:
            raise LegacyDocConversionTimeout("legacy_doc_conversion_timeout: synthetic timeout")

    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.mkdtemp", lambda prefix: str(work_dir))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.shutil.rmtree", lambda _path: (_ for _ in ()).throw(PermissionError("locked")))
    monkeypatch.setattr("backend.app.parser.legacy_doc_parser.time.sleep", lambda delay: sleep_calls.append(delay))
    with pytest.raises(LegacyDocConversionTimeout, match="legacy_doc_conversion_timeout"):
        parse_legacy_doc(source, source.name, TimeoutConverter())
    assert sleep_calls == [0.1, 0.2, 0.4, 0.8, 1.6]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("招标文件.docx", DocumentRole.TENDER),
        ("投标响应文件.docx", DocumentRole.BID),
        ("工艺图.pdf", DocumentRole.TECHNICAL_DRAWING),
        ("notes.pdf", DocumentRole.UNKNOWN),
    ],
)
def test_document_role_classification(filename: str, expected: DocumentRole) -> None:
    assert classify_document_role(filename) == expected


def test_recursive_scan_ignores_word_temp_and_keeps_relative_paths(tmp_path: Path) -> None:
    project = tmp_path / "ProjectA"
    nested = project / "nested" / "drawings"
    nested.mkdir(parents=True)
    make_docx(project / "招标文件.docx")
    make_docx(nested / "投标文件.docx")
    make_docx(project / "~$投标文件.docx")
    result = scan_project(project)
    assert [item.relative_path for item in result.files] == ["nested/drawings/投标文件.docx", "招标文件.docx"]
    assert {item.role for item in result.files} == {DocumentRole.TENDER, DocumentRole.BID}
    assert result.source_folder == "ProjectA"


def test_blank_drawing_pdf_is_not_parse_failure(tmp_path: Path) -> None:
    path = tmp_path / "技术图纸.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    result = document_parser.parse_document(path)
    assert result.page_count == 1
    assert "pdf_has_little_or_no_extractable_text" in result.warnings
