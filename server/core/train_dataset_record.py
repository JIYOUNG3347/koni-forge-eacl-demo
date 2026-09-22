"""Record which dataset a training run actually read.

The training task knows the path exactly — it scans the candidates and picks the
folder that exists — so it writes that into ``training_params.json`` rather than
leaving the report to guess. The candidate order lives here too, so every caller
resolves a name the same way.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Suffix of a TWIST-augmented folder.
TWIST_SUFFIX = "_twist"

#: Search order when looking a dataset up by name (root, whether to add the suffix).
#: `corpus/{name}` → `corpus/{name}_twist` → `generated_corpus/{name}` → `…_twist`.
_ROOTS = ("corpus",)


def candidate_paths(name: str, storage_root: str) -> List[str]:
    """Folder candidates to try for one name, in order."""
    out: List[str] = []
    for root in _ROOTS:
        out.append(os.path.join(storage_root, root, name))
        out.append(os.path.join(storage_root, root, f"{name}{TWIST_SUFFIX}"))
    return out


def resolve_dataset_path(
    name: str,
    storage_root: str,
    explicit_path: Optional[str] = None,
    exists: Optional[Callable[[str], bool]] = None,
) -> str:
    """The folder the training run will read. An explicit path is used as given.

    With no candidate present, the **first** candidate is returned. Training then
    fails with "path not found", which is better than silently training on a
    different dataset.
    """
    if explicit_path:
        return explicit_path
    if not name:
        return os.path.join(storage_root, "corpus", "")
    _exists = exists or os.path.exists
    candidates = candidate_paths(name, storage_root)
    for path in candidates:
        if _exists(path):
            return path
    return candidates[0]


def dataset_record(
    dataset_path: str,
    file_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The record to write into the run output. The report reads only this.

    ``dataset`` is the name of the folder actually read, not the one requested.
    A request for ``agent-gen-X`` that read ``agent-gen-X_twist`` records the latter.
    """
    from server.core.corpus_files import select_data_files

    path = (dataset_path or "").rstrip("/")
    record: Dict[str, Any] = {"dataset": os.path.basename(path), "dataset_path": path}
    if file_names is not None:
        record["data_files"] = select_data_files(list(file_names))
    return record


