"""Storage shape (``provenance.json``)::"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PROVENANCE_FILENAME = "provenance.json"
SCHEMA_VERSION = 1

UPLOAD_META_DIRNAME = ".koni_meta"
UPLOAD_META_FILENAME = "upload.json"

PathLike = Union[str, Path]


@dataclass
class SourceFile:
    """A single original source document a dataset was generated from."""

    name: str
    size: int = 0
    mtime: Optional[str] = None

    @classmethod
    def from_path(cls, path: PathLike) -> "SourceFile":
        p = Path(path)
        try:
            st = p.stat()
            size = int(st.st_size)
            mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
        except OSError:
            size, mtime = 0, None
        return cls(name=p.name, size=size, mtime=mtime)


@dataclass
class ProvenanceRecord:
    """Tier-1 provenance for one generated dataset folder."""

    source_folder: str
    source_files: List[SourceFile] = field(default_factory=list)
    created_at: Optional[str] = None
    generator: Dict[str, Any] = field(default_factory=dict)
    pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)
    source_folder_label: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_folder": self.source_folder,
            "source_folder_label": self.source_folder_label,
            "source_files": [asdict(sf) for sf in self.source_files],
            "created_at": self.created_at,
            "generator": dict(self.generator),
            "pipeline_steps": list(self.pipeline_steps),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        files: List[SourceFile] = []
        for f in data.get("source_files") or []:
            if isinstance(f, dict):
                files.append(
                    SourceFile(
                        name=str(f.get("name", "")),
                        size=int(f.get("size", 0) or 0),
                        mtime=f.get("mtime"),
                    )
                )
            elif isinstance(f, str):  # tolerate a bare filename list
                files.append(SourceFile(name=f))
        return cls(
            source_folder=str(data.get("source_folder", "")),
            source_files=files,
            created_at=data.get("created_at"),
            generator=dict(data.get("generator") or {}),
            pipeline_steps=list(data.get("pipeline_steps") or []),
            source_folder_label=data.get("source_folder_label"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
        )


def collect_source_files(source_dir: PathLike, *, skip_names: Optional[Iterable[str]] = None) -> List[SourceFile]:
    """List files in ``source_dir`` as :class:`SourceFile` records (non-recursive).

    Returns an empty list if the directory is missing.
    """
    skip = set(skip_names or ())
    d = Path(source_dir)
    if not d.is_dir():
        return []
    return [
        SourceFile.from_path(f) for f in sorted(d.iterdir(), key=lambda p: p.name) if f.is_file() and f.name not in skip
    ]


def write_upload_label(dataset_dir: PathLike, folder_name: str) -> Path:
    """Stored under ``<dataset_dir>/.koni_meta/upload.json`` — a subdirectory so it
    is invisible to top-level ``is_file()`` scans (RAG indexer, generator,
    dataset listing). Returns the written path.

    """
    p = Path(dataset_dir) / UPLOAD_META_DIRNAME / UPLOAD_META_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"folder_name": folder_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def provenance_path(dataset_dir: PathLike) -> Path:
    return Path(dataset_dir) / PROVENANCE_FILENAME


def read_provenance(dataset_dir: PathLike) -> Optional[ProvenanceRecord]:
    """Read ``provenance.json`` from ``dataset_dir``.

    Returns ``None`` when the sidecar is missing or unreadable (e.g. pre-v2
    datasets that predate provenance tracking) — callers treat this as
    "unavailable", not an error.
    """
    p = provenance_path(dataset_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return ProvenanceRecord.from_dict(data)
