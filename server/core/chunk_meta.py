"""Pure helpers for page-aware chunking and chunk listings.
No fastapi, celery or chromadb dependency, so it imports in a plain CI venv.

1. build_page_chunks — split per-page text into chunks and tag each with its
   page_num. PDFs carry a real page number; formats with no page concept
   (MD, HWP, DOCX, TXT) get page_num=None.
2. build_chunks_listing — shape chunk records read from Chroma into a listing
   response (sort by chunk_index, paginate, truncate to the preview length).
   Older chunks have no page_num key and fall back to None.

chunk_text comes from modules.rag.vectorstore, whose top-level import is light
(standard library only; chromadb is imported lazily inside methods).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Defaults; this module does not change them.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PREVIEW_CHARS = 200

# (page_num | None, text) pairs
Page = Tuple[Optional[int], str]


def build_page_chunks(
    pages: Sequence[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split per-page text into chunks, tagging each with its page_num.

    Args:
        pages: ``(page_num | None, text)`` entries. A PDF has one per page;
            other formats have a single ``[(None, whole_text)]``.
    Returns:
        ``[{"chunk_index": int, "page_num": int|None, "text": str}]`` —
        chunk_index counts up from 0 across the whole document.
    """
    from modules.rag.vectorstore import chunk_text  # light import, no chromadb

    out: List[Dict[str, Any]] = []
    idx = 0
    for page_num, text in pages:
        for piece in chunk_text(text or "", chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            out.append({"chunk_index": idx, "page_num": page_num, "text": piece})
            idx += 1
    return out


def build_chunks_listing(
    records: Sequence[Dict[str, Any]],
    limit: Optional[int] = 20,
    offset: int = 0,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> Dict[str, Any]:
    """Shape Chroma chunk records into a listing response.

    Args:
        records: ``{"chunk_index", "page_num"?, "doc_name"?, "text"}`` entries.
            A missing page_num key (an older chunk) is treated as None.
        limit: page size. None means everything after offset.
        offset: 0-based start index.
        preview_chars: truncation length for the text preview.
    Returns:
        ``{"chunks": [...], "total": int}`` — total counts before pagination.
    """
    ordered = sorted(
        records,
        key=lambda r: r.get("chunk_index") if isinstance(r.get("chunk_index"), int) else 0,
    )
    total = len(ordered)
    start = max(0, offset)
    window = ordered[start:] if limit is None else ordered[start : start + max(0, limit)]

    chunks: List[Dict[str, Any]] = []
    for r in window:
        text = r.get("text") or ""
        chunks.append(
            {
                "chunk_index": r.get("chunk_index"),
                "page_num": r.get("page_num"),  # None when the key is absent
                "doc_name": r.get("doc_name"),
                "text": text[:preview_chars],
                "truncated": len(text) > preview_chars,
            }
        )
    return {"chunks": chunks, "total": total}
