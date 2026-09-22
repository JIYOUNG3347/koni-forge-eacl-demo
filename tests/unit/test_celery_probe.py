"""The difference between `None` and `set()` is the contract.

Callers skip the zombie check on `None` (fail open, so a live job is not
killed) and trust `set()` as "genuinely nothing is running".
"""

from __future__ import annotations

import ast
from pathlib import Path

from server.core.celery_probe import DEFAULT_TIMEOUT, task_ids

REPO = Path(__file__).resolve().parents[2]
ROUTERS = REPO / "server" / "routers"
TOUCHED = ["eval.py", "train.py", "data.py", "admin.py"]


# ── Pure part ───────────────────────────────────────────────────────────────


def test_task_ids_flattens_the_worker_map():
    assert task_ids({"w1": [{"id": "a"}, {"id": "b"}], "w2": [{"id": "c"}]}) == {"a", "b", "c"}


def test_task_ids_tolerates_none_worker_entries():
    """A worker can return `None`; raising here would disable the zombie check entirely."""
    assert task_ids({"w1": None, "w2": [{"id": "a"}]}) == {"a"}


def test_task_ids_of_nothing_is_an_empty_set():
    assert task_ids(None) == set()
    assert task_ids({}) == set()


def test_timeout_matches_what_the_call_sites_used():
    """Keeps the `timeout=1` all four callers used."""
    assert DEFAULT_TIMEOUT == 1.0


def test_failure_returns_none_not_an_empty_set(monkeypatch):
    """This distinction is the contract.

    Turning a failure into `set()` would read as "nothing is running" while the
    worker is fine, and declare every job a zombie.
    """
    import server.core.celery_probe as probe

    def _boom(timeout):
        raise RuntimeError("broker down")

    monkeypatch.setattr(probe, "_inspect", _boom)
    assert probe.active_task_ids() is None


def test_success_with_no_tasks_is_an_empty_set(monkeypatch):
    import server.core.celery_probe as probe

    class _Insp:
        def active(self):
            return {}

    monkeypatch.setattr(probe, "_inspect", lambda timeout: _Insp())
    assert probe.active_task_ids() == set()


def test_module_imports_without_celery():
    """A CI venv has no celery, so celery is imported lazily inside the function."""
    src = (REPO / "server" / "core" / "celery_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = " ".join(ast.unparse(n) for n in top_level)
    assert "celery" not in names, names
