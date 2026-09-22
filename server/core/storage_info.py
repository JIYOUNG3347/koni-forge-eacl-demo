"""Pure module: NO fastapi / celery / redis imports — so it stays unit-testable
without the web stack.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

PathLike = Union[str, Path]

# Mirrors server/routers/data.py ALLOWED_EXTENSIONS — the file types a dataset
# folder can legitimately contain (raw source docs + generated QA payloads).
DEFAULT_SUPPORTED_EXTS: Set[str] = {
    ".pdf",
    ".hwp",
    ".hwpx",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

# Subdirectories that are never data — excluded from size + count.
META_DIRNAMES: Set[str] = {".koni_meta"}

# Metadata/log sidecars excluded from file_count (but not disk size).
META_FILENAMES: Set[str] = {
    "generation_params.json",
    "provenance.json",
    "preprocessing_stats.txt",
    "training_params.json",
    ".DS_Store",
}


def _is_under_meta_dir(rel_parts: tuple) -> bool:
    """True if any path component is a meta directory (e.g. ``.koni_meta``)."""
    return any(part in META_DIRNAMES for part in rel_parts)


def compute_storage_info(
    folder: PathLike,
    supported_exts: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Return storage facts for ``folder``.

    Always returns a dict with the four keys, even for a missing/empty folder
    (``disk_size_bytes=0``, ``file_count=0``, ``last_modified=None``) so callers
    never have to branch on existence.

    Returns:
        ``{folder_path: str, disk_size_bytes: int, file_count: int,
        last_modified: str | None}`` where ``last_modified`` is ISO 8601.
    """
    folder = Path(folder)
    exts = supported_exts if supported_exts is not None else DEFAULT_SUPPORTED_EXTS

    info: Dict[str, Any] = {
        "folder_path": str(folder),
        "disk_size_bytes": 0,
        "file_count": 0,
        "last_modified": None,
    }
    if not folder.exists() or not folder.is_dir():
        return info

    total_size = 0
    file_count = 0
    latest_mtime = 0.0

    for p in folder.rglob("*"):
        rel_parts = p.relative_to(folder).parts
        if _is_under_meta_dir(rel_parts):
            continue
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        total_size += st.st_size
        if st.st_mtime > latest_mtime:
            latest_mtime = st.st_mtime
        # Count only user-facing data files (supported ext, not a meta sidecar).
        if p.name in META_FILENAMES or p.name.endswith(".log"):
            continue
        if p.suffix.lower() in exts:
            file_count += 1

    info["disk_size_bytes"] = total_size
    info["file_count"] = file_count
    if latest_mtime > 0:
        info["last_modified"] = datetime.fromtimestamp(latest_mtime).isoformat()
    return info
