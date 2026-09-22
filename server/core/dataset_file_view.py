"""What each file in a dataset folder is, and whether it can be deleted::

    corpus/{X}/qa_dataset.json              original
               qa_dataset_preprocessed.json preprocessed
               qa_dataset_twist.json        augmented    <- what training reads
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from server.core.corpus_files import is_data_file, is_refined, select_data_files

#: File role. The UI uses it for icons and sorting; the values are machine-read.
KIND_DATA = "data"
KIND_META = "meta"


def file_kind(name: str) -> str:
    """Whether this file is training data or metadata."""
    return KIND_DATA if is_data_file(name) else KIND_META


def build_file_entries(
    file_names: Sequence[str],
    sizes: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Display entries for **every** file in a folder, sorted by name.

    Nothing is hidden: a hidden file cannot be deleted, and the user cannot see
    why it is there. ``active`` is True only for files training actually reads,
    """
    names = [n for n in file_names if isinstance(n, str) and n]
    active = set(select_data_files(names))
    sizes = sizes or {}
    return [
        {
            "name": name,
            "kind": file_kind(name),
            "active": name in active,
            "size": int(sizes.get(name, 0)),
        }
        for name in sorted(names)
    ]


def removable_reason(file_names: Sequence[str], target: str) -> Optional[str]:
    """Why ``target`` cannot be deleted, or ``None`` when it can.

    **Only a missing file is refused.** Deleting the last training file is
    allowed: the opposite of "undo the augmentation" is "undo the
    preprocessing too", and that is a legitimate intent. The outcome is
    reported through :func:`training_ready` instead of blocked silently.
    """
    if not target or target != target.strip():
        return "invalid_name"
    if target not in set(file_names or ()):
        return "not_found"
    return None


def training_ready(file_names: Sequence[str]) -> bool:
    """Whether the folder still has data to train on."""
    return bool(select_data_files([n for n in file_names if isinstance(n, str)]))


def remaining_after(file_names: Sequence[str], removed: str) -> List[str]:
    """Remaining file names after ``removed``, for the response state."""
    return [n for n in file_names if n != removed]


__all__ = [
    "KIND_DATA",
    "KIND_META",
    "build_file_entries",
    "file_kind",
    "is_refined",
    "remaining_after",
    "removable_reason",
    "training_ready",
]
