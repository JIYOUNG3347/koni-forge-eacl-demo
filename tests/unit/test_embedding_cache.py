"""What this file protects is correctness, not speed.

A cache is easy to add, but **the wrong key fails silently**: the indexing path
uses whatever `embedding_model` the request carries, so an instance built for
model A could be reused for a model-B request and mix both models' vectors into
one collection.
"""

from __future__ import annotations

import ast
from pathlib import Path

from modules.rag.embedding_cache import (
    get_embedding_manager,
    resolve_model_name,
)
from modules.rag.embeddings import ALIAS_MAP, DEFAULT_MODEL, SUPPORTED_MODELS

REPO = Path(__file__).resolve().parents[2]
RAG_ROUTER = REPO / "server" / "routers" / "rag.py"


# ── the key: every risk in this change lives here ──────────────────────────


def test_alias_and_full_id_share_one_instance():
    """Keying **before** normalisation loads the same model twice (5.2GB x 2)."""
    a = get_embedding_manager("multilingual-e5-large")
    b = get_embedding_manager("intfloat/multilingual-e5-large")
    assert a is b


def test_different_models_are_never_shared():
    """The most important guard.

    Sharing one instance would mix vectors from model A and model B in one
    collection. Differing dimensions make Chroma reject it; matching ones fail silently.
    """
    large = get_embedding_manager("intfloat/multilingual-e5-large")
    base = get_embedding_manager("intfloat/multilingual-e5-base")
    assert large is not base
    assert large.model_name != base.model_name


def test_dimensions_actually_differ_across_supported_models():
    """Pin the guard above to a real pair, so it is not a hypothetical risk."""
    dims = {m: SUPPORTED_MODELS[m]["dim"] for m in SUPPORTED_MODELS}
    assert len(set(dims.values())) > 1, dims


def test_same_model_returns_the_same_instance():
    assert get_embedding_manager("bge-m3") is get_embedding_manager("bge-m3")


def test_none_resolves_to_the_env_default():
    assert resolve_model_name(None) == ALIAS_MAP.get(DEFAULT_MODEL, DEFAULT_MODEL)


def test_resolve_matches_the_managers_own_rule():
    """Redefining the normalisation rule here would split the two decisions."""
    from modules.rag.embeddings import EmbeddingManager

    for raw in list(ALIAS_MAP) + ["intfloat/multilingual-e5-large"]:
        assert resolve_model_name(raw) == EmbeddingManager(model_name=raw).model_name




# ── wiring ──────────────────────────────────────────────────────────────────


def test_rag_router_never_constructs_the_manager_directly():
    """A direct construction creeping back would silently reload on every call."""
    body = RAG_ROUTER.read_text(encoding="utf-8")
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    assert "EmbeddingManager(" not in code, "rag.py builds the manager directly again"
    assert "get_embedding_manager(" in code


def test_index_path_still_honours_the_requested_model():
    """Dropping the request's `embedding_model` for the cache would kill model choice."""
    body = RAG_ROUTER.read_text(encoding="utf-8")
    assert "get_embedding_manager(model_name=req.embedding_model)" in body


# ── deliberately left alone ─────────────────────────────────────────────────


def test_embedding_manager_signature_is_untouched():
    """Changing the constructor breaks the existing tests that build it directly."""
    src = (REPO / "modules" / "rag" / "embeddings.py").read_text(encoding="utf-8")
    assert "def __init__(self, model_name: Optional[str] = None):" in src
    assert "_CACHE" not in src, "the cache belongs in its own module"


def test_dialog_keeps_its_own_singleton():
    """The chat path is already fast and is not touched."""
    src = (REPO / "pipelines" / "dialog.py").read_text(encoding="utf-8")
    assert '_rag_cache["em"] = EmbeddingManager()' in src


def test_module_does_not_import_torch_at_module_scope():
    """A CI venv has no sentence-transformers or torch — the import must be lazy."""
    tree = ast.parse((REPO / "modules" / "rag" / "embedding_cache.py").read_text(encoding="utf-8"))
    top = " ".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert "torch" not in top and "sentence_transformers" not in top, top
    assert "modules.rag.embeddings" not in top, "embeddings must be imported inside the function too"


def test_residency_is_documented():
    """A resident model changes the assumption that memory frees up between requests."""
    doc = (REPO / "modules" / "rag" / "embedding_cache.py").read_text(encoding="utf-8")
    assert "resident" in doc and "5.2GB" in doc
