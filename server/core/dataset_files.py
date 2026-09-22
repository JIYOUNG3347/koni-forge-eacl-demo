"""Pure module: no fastapi, celery or redis imports.

This lists real files rather than estimating: only source document extensions
present in the folder, sorted by name. QA payloads and metadata are excluded.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

PathLike = Union[str, Path]

# Source document extensions. Unlike storage_info.DEFAULT_SUPPORTED_EXTS this
# excludes the QA payloads .json and .jsonl, since these are the uploaded sources.
SOURCE_DOC_EXTS: Set[str] = {".pdf", ".hwp", ".hwpx", ".docx", ".pptx", ".txt", ".md"}

# Non-data subdirectories excluded from the listing (.koni_meta/upload.json).
META_DIRNAMES: Set[str] = {".koni_meta"}


def list_dataset_files(
    folder: PathLike,
    exts: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Source document files in ``folder``, sorted by name.

    Returns:
        ``[{name, size, mtime}]``. An empty or missing folder gives an empty list.

    """
    folder = Path(folder)
    allowed = exts if exts is not None else SOURCE_DOC_EXTS
    if not folder.exists() or not folder.is_dir():
        return []

    out: List[Dict[str, Any]] = []
    for p in folder.rglob("*"):
        rel_parts = p.relative_to(folder).parts
        if any(part in META_DIRNAMES for part in rel_parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            {
                "name": p.name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            }
        )

    out.sort(key=lambda f: f["name"].lower())
    return out


def read_dataset_format(folder: PathLike) -> Optional[str]:
    """The ``data_format`` in ``<folder>/.koni_meta/format.json``."""
    from server.core.classify_dataset_format import (
        FORMAT_META_DIRNAME,
        FORMAT_META_FILENAME,
    )

    p = Path(folder) / FORMAT_META_DIRNAME / FORMAT_META_FILENAME
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        fmt = data.get("data_format")
        return str(fmt) if fmt else None
    return None
