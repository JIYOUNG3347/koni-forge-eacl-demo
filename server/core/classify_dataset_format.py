"""Classify an uploaded QA dataset from the columns of its first record:
  * all of ``{"prompt", "chosen", "rejected"}``  -> ``"preference"`` (DPO/ORPO pairs)
  * all of ``{"question", "answer"}``            -> ``"sft"`` (supervised QA)
  * anything else, a parse failure or an empty file -> ``"unknown"`` (rejected)

The format marker is written to ``<dataset_dir>/.koni_meta/format.json``, in
the same ``.koni_meta`` directory as ``provenance.UPLOAD_META_DIRNAME``:
``upload.json`` holds the user's folder label, ``format.json`` the system's
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]

FORMAT_META_DIRNAME = ".koni_meta"
FORMAT_META_FILENAME = "format.json"

SFT = "sft"
PREFERENCE = "preference"
UNKNOWN = "unknown"

_SFT_KEYS = frozenset({"question", "answer"})
_PREFERENCE_KEYS = frozenset({"prompt", "chosen", "rejected"})
_RECORD_KEYS = _SFT_KEYS | _PREFERENCE_KEYS
# List keys to descend into when the .json is a wrapper dict ({"qa_pairs": [...]}).
# The same set as the QA counter in data.py, preserving existing uploads.
_WRAPPER_KEYS = ("qa_pairs", "items", "data", "records")

# Cap on the .jsonl first-line scan, protecting against one huge record.
_JSONL_HEAD_BYTES = 1024 * 1024


def _classify_record(record: Any) -> str:
    """One record dict to a format string, by which required keys are present."""
    if not isinstance(record, dict):
        return UNKNOWN
    keys = record.keys()
    # Check preference first (more specific), so mixed keys prefer the pair form.
    if _PREFERENCE_KEYS <= keys:
        return PREFERENCE
    if _SFT_KEYS <= keys:
        return SFT
    return UNKNOWN


def _candidate_record(obj: Any) -> Optional[dict]:
    """Take the first record dict to classify from parsed JSON.

    * ``list``          -> the first dict element
    * a record ``dict``  -> itself (it holds the required keys)
    * a wrapper ``dict`` -> the first dict under a known list key (``qa_pairs``, ...)
    """
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                return el
        return None
    if isinstance(obj, dict):
        if obj.keys() & _RECORD_KEYS:
            return obj
        for wk in _WRAPPER_KEYS:
            v = obj.get(wk)
            if isinstance(v, list):
                for el in v:
                    if isinstance(el, dict):
                        return el
        return obj
    return None


def _first_jsonl_record(content_bytes: bytes) -> Optional[dict]:
    """Parse only the first non-empty .jsonl line (within 1MB) into a record dict."""
    head = content_bytes[:_JSONL_HEAD_BYTES]
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            return _candidate_record(json.loads(line))
        except ValueError:
            return None
    return None


def classify_dataset_format(content_bytes: bytes, ext: str) -> str:
    """Uploaded file content to ``"sft"``, ``"preference"`` or ``"unknown"``.

    A parse failure, empty file, encoding error or unsupported extension is ``"unknown"``.
    """
    ext = (ext or "").lower()
    if ext == ".jsonl":
        record = _first_jsonl_record(content_bytes)
    elif ext == ".json":
        try:
            record = _candidate_record(json.loads(content_bytes.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            return UNKNOWN
    else:
        return UNKNOWN
    if record is None:
        return UNKNOWN
    return _classify_record(record)


def resolve_upload_target(requested_target: str, data_format: Optional[str]) -> str:
    """Destination root key for the upload."""
    if data_format == PREFERENCE:
        return "training"
    return requested_target


def format_marker_path(dataset_dir: PathLike) -> Path:
    return Path(dataset_dir) / FORMAT_META_DIRNAME / FORMAT_META_FILENAME


def write_dataset_format(
    dataset_dir: PathLike,
    data_format: str,
    *,
    source_filename: Optional[str] = None,
    classified_at: Optional[str] = None,
) -> Path:
    """Write the format marker to ``<dataset_dir>/.koni_meta/format.json``.

    The ``.koni_meta`` subdirectory keeps it out of the top-level ``is_file()``
    scans the RAG indexer and the dataset listing run, so the marker is never
    mistaken for data.
    """
    p = format_marker_path(dataset_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_format": data_format,
        "classified_at": classified_at or datetime.now().isoformat(timespec="seconds"),
    }
    if source_filename:
        payload["source_filename"] = source_filename
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
