"""Reuse an existing index instead of rebuilding it.

Two pure functions:
  * :func:`collection_for` — make the pipeline use the **folder collection**, so
    it reuses what the upload already indexed. Both sides looking at the same
    place is the whole point.
  * :func:`plan_index` — skip the stage when every target document is indexed.

Conservative rule: **when in doubt, index.** A failed collection lookup or a
missing listing never skips, because an empty index would make KBD read the
wrong documents entirely, and a wasted rebuild is cheaper than that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from server.core.pipeline_stage_texts import stage_text as _stage_text

# Legacy shared collection, used when the folder cannot be determined.
LEGACY_COLLECTION = "pipeline_docs"

# Extensions to index. ``index_documents_task`` collects files with this same
# set; if they differ, the "already indexed" decision is wrong.
INDEXABLE_EXTS = frozenset({".txt", ".md", ".json", ".jsonl", ".pdf", ".docx", ".pptx", ".hwp", ".hwpx"})


def collection_for(raw_folder: Optional[str], fallback: str = LEGACY_COLLECTION) -> str:
    """Collection name the pipeline should use.

    The upload API indexes into the ``{folder_name}`` collection, so using the
    same name reuses that index directly. Without a folder, fall back to the
    legacy name to preserve the existing behaviour.
    """
    if isinstance(raw_folder, str) and raw_folder.strip():
        return raw_folder.strip()
    return fallback


@dataclass(frozen=True)
class IndexPlan:
    """Indexing plan: what is left and what can be skipped."""

    pending: tuple[str, ...]
    already_indexed: int
    skip_all: bool

    @property
    def total(self) -> int:
        return len(self.pending) + self.already_indexed

    def summary(self, lang: object = None) -> str:
        if self.skip_all:
            return _stage_text("index_reuse_all", lang, reused=self.already_indexed)
        if self.already_indexed:
            return _stage_text("index_mixed", lang, new=len(self.pending), reused=self.already_indexed)
        return _stage_text("index_new", lang, new=len(self.pending))


def plan_index(
    source_files: Sequence[str],
    indexed_docs: Optional[Iterable[str]],
) -> IndexPlan:
    """Work left after subtracting documents that are already indexed.

    Args:
        source_files: file names to index (names, not paths — the same unit as
            ``doc_name`` in the collection metadata).
        indexed_docs: ``doc_name`` values already in the collection. ``None``
            means "lookup failed" and skips nothing (conservative).

    Returns:
        An :class:`IndexPlan`. ``skip_all`` is True only when there is at least
        one target and all of them are indexed, so an empty folder is never
    """
    files = [f for f in source_files if f]
    if indexed_docs is None:
        return IndexPlan(pending=tuple(files), already_indexed=0, skip_all=False)

    have = {d for d in indexed_docs if d}
    pending = tuple(f for f in files if f not in have)
    already = len(files) - len(pending)
    return IndexPlan(
        pending=pending,
        already_indexed=already,
        skip_all=bool(files) and not pending,
    )
