"""Pick the documents a corpus tool should list.

``.koni_meta`` is defined as a directory that top-level ``is_file()`` scans do
not see, but ``rglob`` is recursive and breaks that convention, which is how
metadata files leaked into the document list.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from server.core.auto_intent import find_folder_in_text
from server.core.provenance import UPLOAD_META_DIRNAME
from server.core.rag_index_plan import INDEXABLE_EXTS


def is_document(rel_path: str) -> bool:
    """Whether this relative path is a user document."""
    if not rel_path:
        return False
    parts = rel_path.replace("\\", "/").split("/")
    if UPLOAD_META_DIRNAME in parts:
        return False
    name = parts[-1]
    dot = name.rfind(".")
    return dot > 0 and name[dot:].lower() in INDEXABLE_EXTS


def folder_of(rel_path: str) -> str:
    """Top-level folder of a relative path. Empty when there is none."""
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else ""


def select_folder(
    folders: Sequence[str],
    user_text: str = "",
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Target folder, or ``None`` to show everything.

    Order: an explicit tool argument, then a match against the user's own words.
    An argument naming a folder that does not exist is ignored, so a hallucinated
    name falls back to the user's text rather than scoping to nothing.
    """
    known = [f for f in folders if f]
    if explicit and explicit in known:
        return explicit
    return find_folder_in_text(user_text or "", known)


def scope_documents(
    docs: Sequence[Mapping[str, Any]],
    target: Optional[str],
) -> List[dict]:
    """Keep only the target folder's documents. Without a target, keep all."""
    if not target:
        return [dict(d) for d in docs]
    return [dict(d) for d in docs if folder_of(str(d.get("name", ""))) == target]


_HINT_EN = {
    "pinned": (
        "Call detect_document_type(raw_dataset_folder='{target}'). "
        "The target folder is '{target}' — do not pick a different folder."
    ),
    "none": "No uploaded folder was found. Ask the user to upload documents.",
    "single": "Call detect_document_type(raw_dataset_folder='{folder}').",
    "ambiguous": (
        "The target folder is not determined. Ask the user which folder to use, then call "
        "detect_document_type(raw_dataset_folder=...). "
        "Candidates: {candidates}"
    ),
}


def target_hint(target: Optional[str], folders: Sequence[str], lang: object = None) -> str:
    """Next-step guidance."""
    t = _HINT_EN
    if target:
        return t["pinned"].format(target=target)
    known = [f for f in folders if f]
    if not known:
        return t["none"]
    if len(known) == 1:
        return t["single"].format(folder=known[0])
    return t["ambiguous"].format(candidates=", ".join(known))


