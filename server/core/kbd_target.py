"""Order the KBD target list, honouring what the user asked for.

``boundary_specialist.list_indexed_documents`` sorted only by chunk count
descending, and the LLM picks the first entry, so whatever the user named, the
largest collection always won. Nothing compared the request against the list.

Policy: **the user's choice wins; size only breaks ties.**
Matching compares against the real collection list rather than enumerating
phrases, so a new wording needs no code change (the same principle as
``auto_intent.find_folder_in_text``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from server.core.auto_intent import find_folder_in_text


def select_kbd_target(
    documents: Sequence[Mapping[str, Any]],
    user_text: str = "",
) -> tuple[list, Optional[str]]:
    """Sort the KBD target list.

    Args:
        documents: collections shaped ``{"id"/"name", "chunks"}``.
        user_text: the user's request; a real collection name in it wins.

    Returns:
        ``(sorted list, the collection the user named or None)``. Sorted by
        (1) what the user asked for, (2) chunk count descending. The chosen
        entry gets ``requested: True`` so the caller can say "using X as you asked".
    """
    docs = [dict(d) for d in documents]
    names = [str(d.get("id") or d.get("name") or "") for d in docs]
    requested = find_folder_in_text(user_text or "", [n for n in names if n])

    def sort_key(doc: Mapping[str, Any]) -> tuple:
        name = str(doc.get("id") or doc.get("name") or "")
        is_requested = requested is not None and name == requested
        try:
            chunks = int(doc.get("chunks") or 0)
        except (TypeError, ValueError):
            chunks = 0
        # The requested one first (0), then chunk count descending.
        return (0 if is_requested else 1, -chunks)

    docs.sort(key=sort_key)
    if requested:
        for doc in docs:
            if str(doc.get("id") or doc.get("name") or "") == requested:
                doc["requested"] = True
                break
    return docs, requested


_HINT_EN = {
    "pinned": (
        "The user specified '{requested}'. Run KBD on this document only "
        "(do not pick a different collection). Then call get_gpu_info."
    ),
    "free": "Call get_gpu_info",
}


def target_hint(requested: Optional[str], lang: object = None) -> str:
    """Next-step guidance attached to the tool result.

    When the user named a target, this pins it so the LLM cannot drift elsewhere.
    """

    t = _HINT_EN
    return t["pinned"].format(requested=requested) if requested else t["free"]
