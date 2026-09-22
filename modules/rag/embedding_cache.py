"""Process-wide cache of embedding models, keyed by normalised model name.

The key matters: the indexing path uses whatever ``embedding_model`` the request
carries, so an instance built for one model must not be reused for another —
two models' vectors in one collection cannot be compared.

A cached model stays **resident** for the life of the process (5.2GB for the
multilingual-e5 family), and every distinct name adds another copy, so memory
does not come back between requests.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover — typing only, no runtime import
    from modules.rag.embeddings import EmbeddingManager

logger = logging.getLogger("rag.embedding_cache")

#: Normalised model name to manager. Lives for the process lifetime.
_CACHE: Dict[str, "EmbeddingManager"] = {}
_LOCK = threading.Lock()


def resolve_model_name(model_name: Optional[str] = None) -> str:
    """Cache key: the model name after alias normalisation."""
    from modules.rag.embeddings import ALIAS_MAP, DEFAULT_MODEL

    raw = model_name or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)
    return ALIAS_MAP.get(raw, raw)


def get_embedding_manager(model_name: Optional[str] = None) -> "EmbeddingManager":
    """Per-model singleton `EmbeddingManager`.

    The same model returns the same instance. `_load_model()` already caches
    within an instance via ``if self._model is not None: return``, so all this
    does is extend that lifetime to the whole process. Results are unchanged.

    """
    from modules.rag.embeddings import EmbeddingManager

    key = resolve_model_name(model_name)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        # Check again inside the lock, or two threads load 5.2GB twice.
        cached = _CACHE.get(key)
        if cached is None:
            logger.info("Embedding manager cached for %s", key)
            cached = _CACHE[key] = EmbeddingManager(model_name=key)
    return cached


