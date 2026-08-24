from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..db.models import AssetType, ProjectFileRole
from ..parser.document_parser import SUPPORTED_EXTENSIONS
from .classifier import classify_asset_type, classify_project_file_role

HASH_CHUNK_SIZE = 1024 * 1024
_SYSTEM_JUNK_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}


@dataclass(frozen=True)
class ScannedProjectFile:
    path: Path
    filename: str
    relative_path: str
    extension: str
    size_bytes: int
    sha256: str
    role: ProjectFileRole
    asset_type: AssetType


@dataclass
class DirectoryScan:
    files: list[ScannedProjectFile]
    scanned_files: int = 0
    ignored: int = 0
    unsupported: int = 0
    supported_paths: set[str] = field(default_factory=set)
    errors: list[tuple[str | None, str]] = field(default_factory=list)


def hash_file_sha256(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hidden_or_system(entry: os.DirEntry[str]) -> bool:
    if entry.name.startswith(".") or entry.name.casefold() in _SYSTEM_JUNK_NAMES:
        return True
    try:
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
        hidden_or_system = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0) | getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 0)
        return bool(attributes & hidden_or_system)
    except OSError:
        return False


def _is_link_or_junction(path: Path, entry: os.DirEntry[str]) -> bool:
    return entry.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _walk_readonly(root: Path, scan: DirectoryScan) -> Iterator[Path]:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            resolved_directory = directory.resolve(strict=True)
            if not resolved_directory.is_relative_to(root) or directory.is_symlink() or (
                hasattr(directory, "is_junction") and directory.is_junction()
            ):
                scan.ignored += 1
                continue
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name.casefold())
        except OSError:
            relative = directory.relative_to(root).as_posix() if directory != root else None
            scan.errors.append((relative, "directory_unreadable"))
            continue
        for entry in entries:
            path = Path(entry.path)
            if _is_hidden_or_system(entry) or _is_link_or_junction(path, entry):
                scan.ignored += 1
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path
            except OSError:
                scan.errors.append((path.relative_to(root).as_posix(), "entry_unreadable"))


def scan_project_directory(root: Path) -> DirectoryScan:
    """Discover and hash supported regular files without parsing or following links."""
    scan = DirectoryScan(files=[])
    for path in _walk_readonly(root, scan):
        scan.scanned_files += 1
        relative = path.relative_to(root)
        relative_path = relative.as_posix()
        if path.name.startswith("~$"):
            scan.ignored += 1
            continue
        extension = path.suffix.casefold()
        if extension not in SUPPORTED_EXTENSIONS:
            scan.unsupported += 1
            continue
        scan.supported_paths.add(relative_path)
        try:
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_relative_to(root) or path.is_symlink() or (
                hasattr(path, "is_junction") and path.is_junction()
            ):
                scan.ignored += 1
                scan.supported_paths.discard(relative_path)
                continue
            before = path.stat()
            sha256 = hash_file_sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                scan.errors.append((relative_path, "file_changed_during_import"))
                continue
        except OSError:
            scan.errors.append((relative_path, "file_unreadable"))
            continue
        parent_parts = relative.parts[:-1]
        scan.files.append(ScannedProjectFile(
            path=path,
            filename=path.name,
            relative_path=relative_path,
            extension=extension,
            size_bytes=after.st_size,
            sha256=sha256,
            role=classify_project_file_role(path.name, parent_parts),
            asset_type=classify_asset_type(path.name, parent_parts),
        ))
    scan.files.sort(key=lambda item: item.relative_path.casefold())
    return scan
