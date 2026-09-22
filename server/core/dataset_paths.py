from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

#: Storage next to the checkout, used when ``STORAGE_BASE_PATH`` is unset.
#: ``server.core.config`` exports that variable from host.toml at import time.
DEFAULT_STORAGE_ROOT = str(Path(__file__).resolve().parents[2] / "storage")

RAW_DIRNAME = "raw_corpus"
CORPUS_DIRNAME = "corpus"


def storage_root() -> Path:
    """Storage root, from ``STORAGE_BASE_PATH`` or the local default."""
    return Path(os.getenv("STORAGE_BASE_PATH", DEFAULT_STORAGE_ROOT))


def chroma_dir() -> Path:
    """ChromaDB store. ``server.core.config`` exports the resolved path."""
    return Path(os.getenv("CHROMA_PERSIST_DIR", str(storage_root() / "chroma")))


def raw_root() -> Path:
    """Root of the source documents (PDF, HWP, DOCX). Kept separate from datasets."""
    return storage_root() / RAW_DIRNAME


def corpus_root() -> Path:
    return storage_root() / CORPUS_DIRNAME


def dataset_output_root() -> Path:
    """Where a new dataset is created."""
    return corpus_root()


def dataset_roots() -> Tuple[Path, ...]:
    """The single root to search for dataset folders."""
    return (corpus_root(),)


def iter_dataset_dirs() -> Iterator[Path]:
    """Iterate every dataset folder. A missing root is skipped."""
    for root in dataset_roots():
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                yield entry


def find_dataset_dir(name: str) -> Optional[Path]:
    """Find a dataset folder by name, or ``None``.

    Nothing is invented: on a miss it does not return the first candidate, which
    the caller would take for a real folder and only discover at read time.
    """
    folder = (name or "").strip()
    if not folder or "/" in folder or folder in (".", ".."):
        return None
    for root in dataset_roots():
        candidate = root / folder
        if candidate.is_dir():
            return candidate
    return None


def find_dataset_file(name: str, *, prefer: Sequence[str] = ()) -> Optional[Path]:
    """One file to read inside a dataset folder, or ``None``.

    ``prefer`` names are tried first, in order, preserving the caller's choice.
    With none of them present, the file is picked by the **same rule** as
    training (:func:`corpus_files.select_data_files`), so a preview and the run
    read the same data.
    """
    directory = find_dataset_dir(name)
    if directory is None:
        return None
    for filename in prefer:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return select_file_in(directory)


def select_file_in(directory: Path) -> Optional[Path]:
    from server.core.corpus_files import select_data_files

    if not directory.is_dir():
        return None
    names = [f.name for f in directory.iterdir() if f.is_file()]
    selected = select_data_files(names)
    return directory / selected[0] if selected else None
