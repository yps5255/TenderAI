from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .api.projects import router as projects_router
from .analyzer.chunking import build_document_chunks
from .analyzer.tender_analyzer import TenderAnalyzer, TenderAnalyzerError
from .llm.exceptions import LLMConfigurationError, LLMConnectionError, LLMError, LLMHTTPError, LLMResponseError, LLMStructuredOutputError, LLMTimeoutError
from .llm.factory import create_llm_provider
from .llm.provider import LLMProvider
from .models import ParsedDocument, TenderAnalysis
from .parser.document_parser import SUPPORTED_EXTENSIONS, parse_document
from .parser.pdf_parser import DocumentParseError
from .settings import Settings

app = FastAPI(title="TenderAI", version=__version__)
app.include_router(projects_router)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.exception_handler(LLMError)
async def llm_error_handler(_request, exc: LLMError) -> JSONResponse:
    """Keep dependency-construction errors safe and consistent for API clients."""
    error = _provider_http_exception(exc)
    return JSONResponse(status_code=error.status_code, content={"detail": error.detail})


async def stream_upload_to_path(file: UploadFile, destination: Path, max_size: int = MAX_UPLOAD_SIZE) -> int:
    """Write an upload incrementally, rejecting immediately after its size limit."""
    uploaded_size = 0
    with destination.open("wb") as uploaded_file:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            uploaded_size += len(chunk)
            if uploaded_size > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="File size must not exceed 500 MiB.",
                )
            uploaded_file.write(chunk)
    return uploaded_size


async def stream_supported_upload_to_temp(file: UploadFile, directory: Path) -> tuple[Path, str]:
    """Validate and stream one supported upload into the caller-owned TEMP directory."""
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .doc, .docx, and .pdf files are supported.",
        )
    destination = directory / f"upload{extension}"
    uploaded_size = await stream_upload_to_path(file, destination)
    if uploaded_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded file is empty.",
        )
    return destination, filename


def get_settings() -> Settings:
    return Settings()


def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    return create_llm_provider(settings)


def _analyze_parsed_document(document: ParsedDocument, provider: LLMProvider, settings: Settings):
    analyzer = TenderAnalyzer(
        provider,
        chunk_builder=lambda value: build_document_chunks(
            value,
            max_chars=settings.analysis_chunk_max_chars,
            overlap_chars=settings.analysis_chunk_overlap_chars,
        ),
    )
    return analyzer.analyze(document)


def _provider_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, LLMConfigurationError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM provider configuration is unavailable.")
    if isinstance(error, LLMTimeoutError):
        return HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM provider request timed out.")
    if isinstance(error, LLMConnectionError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM provider connection failed.")
    if isinstance(error, (LLMHTTPError, LLMStructuredOutputError, LLMResponseError)):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM provider returned an unusable response.")
    if isinstance(error, TenderAnalyzerError):
        cause = error.__cause__
        if isinstance(cause, Exception) and isinstance(cause, LLMError):
            return _provider_http_exception(cause)
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Tender analysis failed.")
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Tender analysis failed.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/documents/parse", response_model=ParsedDocument)
async def parse_uploaded_document(file: UploadFile = File(...)) -> ParsedDocument:
    """Parse a supported upload without persisting it in the project."""
    try:
        with TemporaryDirectory(prefix="tenderai-") as directory:
            destination, filename = await stream_supported_upload_to_temp(file, Path(directory))
            return parse_document(destination, filename)
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()


@app.post("/api/v1/tenders/analyze", response_model=TenderAnalysis)
async def analyze_uploaded_tender(
    file: UploadFile = File(...),
    settings: Annotated[Settings, Depends(get_settings)] = None,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)] = None,
):
    """Stream, parse, and analyze a tender upload while always releasing TEMP files."""
    assert settings is not None
    assert provider is not None
    try:
        with TemporaryDirectory(prefix="tenderai-") as directory:
            destination, filename = await stream_supported_upload_to_temp(file, Path(directory))
            try:
                document = await run_in_threadpool(parse_document, destination, filename)
            except DocumentParseError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Document could not be parsed.",
                ) from exc
            try:
                return await run_in_threadpool(_analyze_parsed_document, document, provider, settings)
            except (LLMError, TenderAnalyzerError) as exc:
                raise _provider_http_exception(exc) from exc
    finally:
        await file.close()
