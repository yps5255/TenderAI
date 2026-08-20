from __future__ import annotations

from pathlib import Path

from ..models import DocumentRole, ProjectDocument, ProjectScan
from .document_parser import SUPPORTED_EXTENSIONS, parse_document

_KEYWORDS: dict[DocumentRole, tuple[str, ...]] = {
    DocumentRole.TENDER: ("招标", "采购文件", "磋商文件", "询价文件", "比选文件", "tender", "rfp"),
    DocumentRole.BID: ("投标", "响应文件", "响应书", "标书", "bid", "proposal", "response"),
    DocumentRole.TECHNICAL_DRAWING: ("图纸", "技术图", "工艺图", "设计图", "设备图", "施工图", "总图", "平面图", "流程图", "drawing", "blueprint"),
    DocumentRole.ATTACHMENT: ("附件", "attachment", "appendix"),
}


def classify_document_role(filename: str, parent_directory_names: tuple[str, ...] = ()) -> DocumentRole:
    def matches_for(text: str) -> list[DocumentRole]:
        normalized = text.casefold()
        return [role for role, keywords in _KEYWORDS.items() if any(keyword.casefold() in normalized for keyword in keywords)]

    matches = matches_for(filename)
    if not matches and parent_directory_names:
        matches = matches_for(" ".join(parent_directory_names))
    return matches[0] if len(matches) == 1 else DocumentRole.UNKNOWN


def scan_project(project_dir: Path) -> ProjectScan:
    """Recursively discover supported project files without exposing absolute paths."""
    files: list[ProjectDocument] = []
    for path in sorted(project_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.name.startswith("~$") or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(project_dir)
        record = ProjectDocument(filename=path.name, relative_path=relative.as_posix(), file_type=path.suffix.lower().lstrip("."), role=classify_document_role(path.name, relative.parts[:-1]), parse_success=False)
        try:
            parsed = parse_document(path, path.name)
            record.parse_success = True
            record.paragraph_count = len(parsed.paragraphs)
            record.table_count = len(parsed.tables)
            record.warnings_count = len(parsed.warnings)
        except Exception as exc:
            record.error_type = type(exc).__name__
            record.error_message_short = str(exc).replace("\n", " ")[:200]
        files.append(record)
    return ProjectScan(project_id=project_dir.name, source_folder=project_dir.name, files=files)


def scan_projects(root: Path) -> list[ProjectScan]:
    return [scan_project(path) for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()) if path.is_dir()]
