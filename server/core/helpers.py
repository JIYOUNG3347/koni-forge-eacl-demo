"""Small helpers with no dependencies on the rest of the app."""

from pathlib import Path
from typing import Any, Optional, Union


def format_size(size_bytes: Union[int, float]) -> str:
    """
    Format byte count as human-readable string.

    Examples:
        format_size(1024) → "1.00 KB"
        format_size(1048576) → "1.00 MB"
        format_size(0) → "0 B"
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)

    for unit in units:
        if abs(size) < 1024.0:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} EB"


def get_dir_size(path: Union[str, Path]) -> int:
    """
    Get total size of a directory in bytes (recursive).

    Args:
        path: Directory path

    Returns:
        Total size in bytes. Returns 0 if path doesn't exist.
    """
    total = 0
    p = Path(path)
    if not p.exists():
        return 0

    if p.is_file():
        return p.stat().st_size

    for entry in p.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except (OSError, PermissionError):
                continue

    return total


def safe_upload_basename(filename: Optional[str]) -> Optional[str]:
    """Folder uploads (``<input webkitdirectory>``) send the path-prefixed name
    ``"subdir/file.pdf"`` via ``webkitRelativePath``; joining that onto the
    dataset dir fails because the subdir does not exist. ``Path(...).name``
    strips directory components (incl. ``../`` traversal). Returns the safe
    basename, or ``None`` for empty/invalid names.

    Lives here (fastapi-free) so it is unit-testable without importing the
    FastAPI router, which is absent in the CI unit-test environment.

    """
    if not filename:
        return None
    base = Path(filename).name.strip()
    if not base or base in (".", ".."):
        return None
    return base


def safe_json_serialize(obj: Any) -> Any:
    """
    Make an object JSON-serializable by converting special types.
    """

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    return str(obj)
