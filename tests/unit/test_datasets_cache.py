"""Static source checks on the dataset listing cache.

The handler itself is exercised by running the app; this only pins that the
cache is read before the expensive scan and invalidated by the mutating routes.
"""

import re
from pathlib import Path

DATA_PY = Path("server/routers/data.py")


def _read() -> str:
    return DATA_PY.read_text(encoding="utf-8")


class TestCacheConstants:
    def test_key_prefix_defined(self):
        src = _read()
        assert "_DATASETS_CACHE_KEY_PREFIX" in src
        assert "datasets_listing_cache:" in src, (
            "the cache key prefix must be a distinct, namespaced string so "
            "it can't collide with other Redis keys in this codebase"
        )

    def test_ttl_is_short(self):
        """TTL must stay short (1-3s) — polling cadence is 5-15s, so a
        longer TTL would let mutation-invalidated state linger past the
        next poll, defeating the invalidation."""
        src = _read()
        m = re.search(r"_DATASETS_CACHE_TTL\s*=\s*(\d+)", src)
        assert m, "_DATASETS_CACHE_TTL constant missing"
        ttl = int(m.group(1))
        assert 1 <= ttl <= 3, f"TTL {ttl}s is out of safe range [1, 3]"

    def test_cache_key_includes_user_and_role(self):
        """Per-user keying defends against admin/user response divergence
        even though they currently return identical data — the moment any
        per-user filtering lands in list_datasets, the cache stays correct
        without a follow-up patch."""
        src = _read()
        key_fn_idx = src.find("def _datasets_cache_key")
        assert key_fn_idx != -1
        body = src[key_fn_idx : src.find("\n\n", key_fn_idx)]
        assert "user_id" in body
        assert "role" in body


class TestHandlerWiring:
    def test_handler_checks_cache_before_running_helpers(self):
        """Cache GET must come BEFORE the expensive helpers (ChromaDB
        client init, scan_dir, r.keys), or a hit doesn't actually skip
        the work."""
        src = _read()
        handler_idx = src.find("async def list_datasets")
        assert handler_idx != -1
        chroma_idx = src.find("chromadb.PersistentClient", handler_idx)
        cache_get_idx = src.find("cache_key", handler_idx)
        assert cache_get_idx != -1, "list_datasets must reference cache_key"
        assert cache_get_idx < chroma_idx, (
            "cache GET must precede ChromaDB init / scan_dir / r.keys; "
            "otherwise the cache only saves the JSON serialization step"
        )

    def test_handler_writes_to_cache_before_return(self):
        src = _read()
        handler_idx = src.find("async def list_datasets")
        next_route_idx = src.find("@router.", handler_idx + 1)
        body = src[handler_idx : next_route_idx if next_route_idx != -1 else len(src)]
        assert "setex(cache_key" in body, (
            "list_datasets must SETEX the computed response so the next poll within TTL hits the cache"
        )

    def test_handler_pulls_user_id_and_role_from_request_state(self):
        src = _read()
        handler_idx = src.find("async def list_datasets")
        end = src.find("@router.", handler_idx + 1)
        body = src[handler_idx : end if end != -1 else len(src)]
        assert "request.state" in body, (
            "handler must read user_id / user_role from request.state to "
            "key the cache; otherwise all users share one entry"
        )
        # Both attributes should be referenced
        assert "user_id" in body
        assert "user_role" in body


class TestInvalidationHelperShape:
    def test_invalidate_helper_swallows_redis_errors(self):
        """If Redis is down, mutation routes must still succeed —
        invalidation is best-effort. The next listing will recompute when
        the cache GET fails anyway."""
        src = _read()
        idx = src.find("def _invalidate_datasets_cache(")
        assert idx != -1
        end = src.find("\ndef ", idx + 1)
        body = src[idx : end if end != -1 else len(src)]
        assert "try:" in body and "except" in body, (
            "_invalidate_datasets_cache must wrap Redis access so a broker outage doesn't fail user-facing mutations"
        )
