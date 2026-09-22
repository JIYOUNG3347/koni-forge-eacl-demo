"""Verifies that all backend files reference the canonical /storage/chroma
path (via CHROMA_PERSIST_DIR env var) and not the legacy /storage/chroma_db.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── helpers ──────────────────────────────────────────────────────────────────


def _source(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


# ── A. static grep — no /storage/chroma_db hardcoding in changed files ──────


class TestNoChromaDbHardcoding:
    CHANGED_FILES = [
        "server/routers/eval.py",
        "server/routers/rag.py",
        "modules/agents/foundation.py",
        "modules/agents/dispatcher.py",
        "modules/agents/specialists/assessment_specialist.py",
        "modules/agents/specialists/corpus_specialist.py",
        "server/routers/system.py",
    ]

    def test_rag_no_chroma_db_literal(self):
        src = _source("server/routers/rag.py")
        # _rebuild_datasets_from_chroma must not fall back to chroma_db
        assert '"chroma_db"' not in src and "chroma_db" not in src, "rag.py still references /storage/chroma_db"

    def test_foundation_no_chroma_db_literal(self):
        src = _source("modules/agents/foundation.py")
        assert "chroma_db" not in src, "foundation.py still references /storage/chroma_db"

    def test_dispatcher_no_chroma_db_literal(self):
        src = _source("modules/agents/dispatcher.py")
        assert "chroma_db" not in src, "dispatcher.py still references /storage/chroma_db"

    def test_system_no_chroma_db_literal(self):
        src = _source("server/routers/system.py")
        assert "chroma_db" not in src, "system.py still references /storage/chroma_db"


# ── B. eval.py uses CHROMA_PERSIST_DIR-based path for gate ──────────────────


# ── C. system.py storage scan uses 'chroma' key (not 'chroma_db') ───────────


class TestSystemStorageKey:
    def test_system_storage_scan_key_not_chroma_db(self):
        src = _source("server/routers/system.py")
        assert '"chroma_db"' not in src, "system.py still uses old 'chroma_db' key"


# ── D. config.py CHROMA_DIR defaults to /storage/chroma (not chroma_db) ─────


class TestConfigChromaDir:
    def test_config_chroma_dir_default_is_chroma(self):
        src = _source("server/core/config.py")
        assert (
            '"chroma"' in src or '/ "chroma"' in src or "/ 'chroma'" in src or 'chroma")' in src or "chroma')" in src
        ), "config.py CHROMA_DIR default does not point to /storage/chroma"
        assert "chroma_db" not in src, "config.py CHROMA_DIR default still references chroma_db"

    def test_config_chroma_dir_uses_chroma_persist_dir_env(self):
        src = _source("server/core/config.py")
        # CHROMA_DIR must be derived from CHROMA_PERSIST_DIR env var
        assert "CHROMA_PERSIST_DIR" in src, "config.py CHROMA_DIR does not reference CHROMA_PERSIST_DIR env var"
