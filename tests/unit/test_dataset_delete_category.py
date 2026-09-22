"""Static source checks: the router and the two UI files must agree on the
delete category, and importing a router here would drag in the whole app.
"""

from pathlib import Path

DATA_PY = Path("server/routers/data.py")
DATASTORE_TS = Path("UI/src/stores/dataStore.ts")
DATAVIEW_TSX = Path("UI/src/views/guided/DataView.tsx")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _slice_function(src: str, marker: str) -> str:
    idx = src.find(marker)
    assert idx != -1, f"marker {marker!r} not found"
    end_candidates = [src.find(t, idx + len(marker)) for t in ("\n@router.", "\nasync def ", "\ndef ")]
    end_candidates = [e for e in end_candidates if e != -1]
    end = min(end_candidates) if end_candidates else len(src)
    return src[idx:end]


class TestEndpointAcceptsCategory:
    def test_category_query_param_present(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        # Must accept a `category` query parameter constrained to the
        # three storage categories
        assert "category" in body, "delete_dataset must accept a category parameter"
        assert "Query(" in body, (
            "category must be a FastAPI Query parameter so the regex constraint and OpenAPI doc apply"
        )
        assert 'regex="^(raw|training)$"' in body or "regex='^(raw|training)$'" in body, (
            "category Query must enforce regex ^(raw|training)$ so the backend doesn't silently accept typos"
        )

    def test_query_imported(self):
        src = _read(DATA_PY)
        # FastAPI's Query must be importable in this module
        assert "Query" in src.split("from fastapi import")[1].split("\n")[0], (
            "Query must be imported from fastapi alongside Request etc."
        )


class TestNoBuggyLoopInCategoryPath:
    """The category-aware path must NOT use the legacy for-loop+break
    pattern. The legacy fallback may still contain it (preserved for
    backward compatibility), but the new path must pin to the requested
    category's base only."""

    def test_category_path_uses_direct_base_lookup(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        # The category path should select a base by key, not iterate.
        # _CATEGORY_BASES dict makes this explicit and grep-able.
        assert "_CATEGORY_BASES" in body, (
            "category-aware path should select a base from a single dict lookup, not iterate"
        )


class TestRagCleanupHelper:
    def test_helper_calls_chromadb_delete_collection(self):
        body = _slice_function(_read(DATA_PY), "def _cleanup_rag_state")
        assert "client.delete_collection" in body, (
            "_cleanup_rag_state must call client.delete_collection so "
            "per-folder collections don't accumulate as orphans"
        )
        assert "dataset_id" in body, (
            "delete_collection must target the dataset_id (= collection name per data_tasks.py:203's 1:1 mapping)"
        )

    def test_helper_clears_rag_redis_key(self):
        body = _slice_function(_read(DATA_PY), "def _cleanup_rag_state")
        assert "_RAG_JOB_KEY_PREFIX" in body, (
            "_cleanup_rag_state must clear the rag_idx:{folder} key "
            "alongside the chroma collection — both belong to the same "
            "raw-folder lifecycle"
        )

    def test_helper_swallows_errors(self):
        """Missing collection / Redis outage must not fail the parent
        request. Cleanup is best-effort."""
        body = _slice_function(_read(DATA_PY), "def _cleanup_rag_state")
        # Two try/except blocks (one for chroma, one for Redis)
        assert body.count("try:") >= 2 and body.count("except") >= 2, (
            "_cleanup_rag_state must wrap both ChromaDB and Redis ops in try/except — neither failure should propagate"
        )


class TestRawOnlyTriggersRagCleanup:
    """processed / training deletes must NOT touch ChromaDB. RAG state
    is tied to the raw folder's lifecycle only."""

    def test_rag_cleanup_gated_on_category_raw(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        # The category-aware path should set skip_rag_cleanup based on
        # category, and only call _cleanup_rag_state when not skipping.
        assert "skip_rag_cleanup" in body, (
            "category-aware path must explicitly gate RAG cleanup so "
            "processed / training deletes don't drop a same-named raw "
            "folder's chroma collection"
        )
        # _cleanup_rag_state should be reachable from the handler
        assert "_cleanup_rag_state(" in body


class TestIndexingInProgressGuard:
    def test_handler_checks_indexing_status(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        assert "_get_indexing_status" in body, (
            "raw deletion must consult _get_indexing_status — racing the "
            "worker's chroma writes risks half-deleted state"
        )
        # If indexing/queued, skip RAG cleanup and surface a warning.
        assert "warnings" in body, (
            "the response must include a warnings field so the UI can tell the user to cancel the indexing job first"
        )


class TestGracefulRmtree:
    def test_rmtree_wrapped_in_try_except(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        # rmtree must not bubble up an unhandled OSError into 500.
        # Look for shutil.rmtree inside a try block.
        assert "shutil.rmtree" in body
        rmtree_idx = body.find("shutil.rmtree")
        # Walk back to find the nearest `try:` — it should be within a
        # reasonable window (50 lines is generous)
        before = body[max(0, rmtree_idx - 1500) : rmtree_idx]
        assert "try:" in before, (
            "shutil.rmtree must be inside a try block so a permission / "
            "NFS error is reported as a structured 500 instead of an "
            "uncaught exception"
        )


class TestLegacyFallbackPreserved:
    """Backward compat: callers that don't pass category still work
    (with the old buggy semantics + a warning log). Internal scripts /
    older clients shouldn't break atomically with this PR."""

    def test_legacy_path_logs_warning(self):
        body = _slice_function(_read(DATA_PY), "async def delete_dataset")
        # Look for the warning log distinguishing the legacy fallback
        assert "legacy" in body.lower(), (
            "legacy fallback should be clearly labelled in logs so a grep shows which callers haven't migrated yet"
        )


class TestFrontendPassesCategory:
    """A delete call must pass the category the item actually belongs to."""

    def test_delete_passes_item_own_category(self):
        src = _read(DATAVIEW_TSX)
        assert "handleDeleteDataset(item.id, item.type" in src, (
            "the delete wiring must pass the item id together with the item type — "
            "writing the category separately at the call site can disagree with the list"
        )

    def test_delete_category_is_narrowed_to_the_backend_values(self):
        src = _read(DATAVIEW_TSX)
        msg = "the category passed must be narrowed to the same values as the backend regex"
        assert "handleDeleteDataset(item.id, item.type)" in src, msg

    def test_handler_signature_takes_category(self):
        src = _read(DATAVIEW_TSX)
        # The handler must accept a category arg with the typed union
        assert 'category: "raw" | "training"' in src, (
            "handleDeleteDataset must declare a typed category union so "
            "the three CategoryCard call sites can't pass a wrong value"
        )


class TestDataStorePassesCategory:
    def test_signature_accepts_category(self):
        src = _read(DATASTORE_TS)
        assert 'deleteDataset: (datasetId: string, category?: "raw" | "training")' in src, (
            "dataStore.deleteDataset signature must accept the optional "
            "category union — the backend regex matches these exact values"
        )

    def test_url_includes_category_query_param(self):
        src = _read(DATASTORE_TS)
        # Look for the URL builder using category as ?category=
        assert "?category=" in src, (
            "dataStore.deleteDataset must encode category as a query "
            "parameter (?category=...) so the FastAPI Query picks it up"
        )
