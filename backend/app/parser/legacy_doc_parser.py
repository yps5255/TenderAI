from __future__ import annotations

import shutil
import subprocess
import os
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Protocol

from ..models import ParsedDocument
from .docx_parser import parse_docx
from .pdf_parser import DocumentParseError


class LegacyDocConverterUnavailable(DocumentParseError):
    pass


class LegacyDocConversionError(DocumentParseError):
    pass


class LegacyDocConversionTimeout(LegacyDocConversionError):
    pass


class LegacyDocConversionProcessFailed(LegacyDocConversionError):
    pass


class LegacyDocOutputMissing(LegacyDocConversionError):
    pass


class LegacyDocConverter(Protocol):
    def convert(self, source: Path, output_dir: Path, profile_dir: Path) -> Path: ...


class LibreOfficeDocConverter:
    executable_names = ("soffice.com", "soffice", "libreoffice")

    default_windows_paths = (
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    )

    LEGACY_DOC_CONVERSION_TIMEOUT_SECONDS = 180
    OUTPUT_POLL_INTERVAL_SECONDS = 0.5
    OUTPUT_POLL_TIMEOUT_SECONDS = 15

    def _find_executable(self) -> str:
        configured = os.environ.get("LIBREOFFICE_PATH")
        if configured is not None:
            candidate = Path(configured.strip().strip("\"'"))
            if candidate.is_file():
                return str(candidate)
            raise LegacyDocConverterUnavailable(
                "configured_libreoffice_not_found: LIBREOFFICE_PATH must reference an executable file."
            )

        for name in self.executable_names:
            if executable := shutil.which(name):
                return executable
        for candidate in self.default_windows_paths:
            if candidate.is_file():
                return str(candidate)
        raise LegacyDocConverterUnavailable(
            "legacy_doc_converter_unavailable: LibreOffice was not found."
        )

    def convert(self, source: Path, output_dir: Path, profile_dir: Path) -> Path:
        executable = self._find_executable()
        try:
            result = subprocess.run([executable, f"-env:UserInstallation={profile_dir.resolve().as_uri()}", "--headless", "--convert-to", "docx", "--outdir", str(output_dir), str(source)], check=False, capture_output=True, text=True, timeout=self.LEGACY_DOC_CONVERSION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise LegacyDocConversionTimeout("legacy_doc_conversion_timeout: LibreOffice conversion exceeded the time limit.") from exc
        except OSError as exc:
            raise LegacyDocConversionProcessFailed("legacy_doc_conversion_process_failed: LibreOffice could not start.") from exc
        if result.returncode != 0:
            raise LegacyDocConversionProcessFailed(f"legacy_doc_conversion_process_failed: LibreOffice exited with code {result.returncode}.")
        converted = output_dir / f"{source.stem}.docx"
        deadline = time.monotonic() + self.OUTPUT_POLL_TIMEOUT_SECONDS
        while not converted.is_file() and time.monotonic() < deadline:
            time.sleep(self.OUTPUT_POLL_INTERVAL_SECONDS)
        if not converted.is_file():
            raise LegacyDocOutputMissing("legacy_doc_output_missing: LibreOffice did not produce a DOCX file.")
        return converted


def _cleanup_directory(directory: Path) -> bool:
    for delay in (0.1, 0.2, 0.4, 0.8, 1.6):
        try:
            shutil.rmtree(directory)
            return True
        except OSError:
            time.sleep(delay)
    return not directory.exists()


def parse_legacy_doc(path: Path, filename: str, converter: LegacyDocConverter | None = None) -> ParsedDocument:
    """Convert only in a unique system TEMP directory, then reuse the DOCX parser."""
    root = Path(mkdtemp(prefix="tenderai-doc-"))
    parsed: ParsedDocument | None = None
    failure: Exception | None = None
    try:
        output_dir = root / "output"
        profile_dir = root / "profile"
        output_dir.mkdir()
        profile_dir.mkdir()
        parsed = parse_docx((converter or LibreOfficeDocConverter()).convert(path, output_dir, profile_dir), filename)
    except Exception as exc:
        failure = exc
    cleaned = _cleanup_directory(root)
    if failure is not None:
        raise failure
    assert parsed is not None
    warnings = [*parsed.warnings, *( ["legacy_doc_temp_cleanup_failed"] if not cleaned else [])]
    return parsed.model_copy(update={"file_type": "doc", "filename": filename, "warnings": warnings})
