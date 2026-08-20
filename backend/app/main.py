from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from .models import ParsedDocument
from .parser.document_parser import SUPPORTED_EXTENSIONS, parse_document
from .parser.pdf_parser import DocumentParseError

app = FastAPI(title="TenderAI")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/documents/parse", response_model=ParsedDocument)
async def parse_uploaded_document(file: UploadFile = File(...)) -> ParsedDocument:
    """Parse a PDF or DOCX upload without persisting it in the project."""
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf and .docx files are supported.",
        )

    try:
        with TemporaryDirectory(prefix="tenderai-") as directory:
            destination = Path(directory) / f"upload{extension}"
            uploaded_size = 0
            with destination.open("wb") as uploaded_file:
                while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                    uploaded_size += len(chunk)
                    if uploaded_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="File size must not exceed 50 MB.",
                        )
                    uploaded_file.write(chunk)
            if uploaded_size == 0:
                raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="The uploaded file is empty.",
                )
            return parse_document(destination, filename)
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()
