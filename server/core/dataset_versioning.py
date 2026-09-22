r"""Dataset folder versioning: a trailing ``_v{N}`` suffix.

Parsing is one regex, ``_v(\d+)$``. Raw ids are ``dataset_{timestamp}``, which
never end in ``_v<digits>``, so false positives are not a concern.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]

# Trailing _v{N} suffix, N >= 1.
_VERSION_RE = re.compile(r"^(?P<base>.+)_v(?P<ver>\d+)$")


def parse_version(folder_name: str) -> Tuple[str, int]:
    """Folder name to ``(base_id, version)``.

    ``"dataset_xxx"`` → ``("dataset_xxx", 1)`` (implicit v1, carry-over)
    ``"dataset_xxx_v2"`` → ``("dataset_xxx", 2)``
    ``"dataset_xxx_v10"`` → ``("dataset_xxx", 10)``
    Anything else (no suffix) gives ``(folder_name, 1)``.
    """
    m = _VERSION_RE.match(folder_name)
    if not m:
        return folder_name, 1
    ver = int(m.group("ver"))
    if ver < 1:  # _v0 and similar are not versions
        return folder_name, 1
    return m.group("base"), ver


def list_versions(base_id: str, root_dir: PathLike) -> List[int]:
    """Version numbers belonging to ``base_id`` under ``root_dir``, sorted."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    versions: List[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        base, ver = parse_version(entry.name)
        if base == base_id:
            versions.append(ver)
    return sorted(versions)


