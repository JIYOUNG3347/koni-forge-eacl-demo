"""When a worker restarts, the message it was handling is gone but
`train:state:*` remains, and with nothing to rewind it, it stays queued forever.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("redis") is None, reason="redis client not installed")


from pathlib import Path

from server.core.job_reconcile import (
    INTERRUPTED,
    NON_TERMINAL,
    needs_reconcile,
    reconciled,
    summarize,
)
from server.core.queue_view import RUNNING_STATUSES, WAITING_STATUSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── A. What needs clearing ───────────────────────────────────────────
def test_non_terminal_states_are_reconciled():
    for status in ("queued", "QUEUED", "started", "PROGRESS", "retry", "pending"):
        assert needs_reconcile({"status": status}) is True, status


def test_terminal_states_are_left_alone():
    """Touching a finished job would corrupt its completion history."""
    for status in ("completed", "SUCCESS", "failed", "FAILURE", "cancelled", INTERRUPTED):
        assert needs_reconcile({"status": status}) is False, status


def test_malformed_state_is_ignored():
    assert needs_reconcile(None) is False
    assert needs_reconcile({}) is False
    assert needs_reconcile("queued") is False


# ── B. The cleared result ────────────────────────────────────────────
def test_reconciled_keeps_the_original_fields():
    """What the job was and how it was configured must survive, so it can be re-run."""
    state = {"status": "queued", "job_id": "j1", "model_name": "Qwen3-4B", "epochs": 3}
    out = reconciled(state)
    assert out["status"] == INTERRUPTED
    assert out["model_name"] == "Qwen3-4B" and out["epochs"] == 3 and out["job_id"] == "j1"
    assert out["message"]


def test_reconciled_does_not_mutate_input():
    state = {"status": "queued", "job_id": "j1"}
    reconciled(state)
    assert state["status"] == "queued"


def test_interrupted_is_distinct_from_failed():
    """Not "ran and failed" but "never ran and was lost" — the distinction matters."""
    assert INTERRUPTED not in ("failed", "FAILURE", "error")


def test_summarize_lists_what_was_cleaned():
    states = [{"status": "queued", "job_id": "a"}, {"status": "completed", "job_id": "b"}]
    assert summarize(states) == ["a"]


# ── C. Agreement with the queue view ─────────────────────────────────
def test_interrupted_does_not_appear_in_the_queue():
    """Clearing it but leaving it in the queue solves nothing."""
    up = INTERRUPTED.upper()
    assert up not in WAITING_STATUSES and up not in RUNNING_STATUSES


def test_every_queue_status_is_covered_by_reconcile():
    """Every status the queue reads as waiting or running must be cleared, or ghosts remain."""
    for status in WAITING_STATUSES | RUNNING_STATUSES:
        assert status.lower() in NON_TERMINAL, status


# ── D. Wiring guard ──────────────────────────────────────────────────
def test_hooked_on_worker_start():
    """Nothing is running when a worker starts, which is the only safe moment to clear."""
    src = (PROJECT_ROOT / "celery_app/gpu_hooks.py").read_text(encoding="utf-8")
    assert "worker_ready" in src
    assert "reconcile_orphans" in src




class _FakeRedis:
    """`job:` is a hash and the rest are strings; this mimics the real storage."""

    def __init__(self, hashes=None, strings=None):
        self.hashes = dict(hashes or {})
        self.strings = dict(strings or {})

    def scan_iter(self, match=""):
        prefix = match.rstrip("*")
        for k in list(self.hashes) + list(self.strings):
            if k.startswith(prefix):
                yield k

    def type(self, key):
        if key in self.hashes:
            return "hash"
        return "string" if key in self.strings else "none"

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping=None, **_kw):
        self.hashes.setdefault(key, {}).update(mapping or {})


def _run(fake):
    import server.core.state as state_mod
    from server.core import job_reconcile as jr

    original = state_mod._get_redis
    state_mod._get_redis = lambda: fake  # type: ignore[assignment]
    try:
        return jr.reconcile_orphan_job_hashes()
    finally:
        state_mod._get_redis = original  # type: ignore[assignment]


def test_uppercase_running_statuses_are_reconciled():
    """`job:` uses upper case, so the decision must ignore case."""
    from server.core.job_reconcile import INTERRUPTED

    fake = _FakeRedis(
        hashes={
            "job:pipe-old": {"job_id": "pipe-old", "status": "STARTED", "progress": "66"},
            "job:train-x": {"job_id": "train-x", "status": "PROGRESS"},
        }
    )
    cleaned = _run(fake)
    assert sorted(cleaned) == ["pipe-old", "train-x"]
    assert fake.hashes["job:pipe-old"]["status"] == INTERRUPTED


def test_terminal_jobs_are_untouched():
    fake = _FakeRedis(
        hashes={
            "job:a": {"job_id": "a", "status": "SUCCESS"},
            "job:b": {"job_id": "b", "status": "FAILURE"},
            "job:c": {"job_id": "c", "status": "interrupted"},
        }
    )
    assert _run(fake) == []
    assert fake.hashes["job:a"]["status"] == "SUCCESS"


def test_other_fields_survive():
    """What the job was must survive so the user can trace it — this corrects, never deletes."""
    fake = _FakeRedis(hashes={"job:x": {"job_id": "x", "status": "STARTED", "progress": "26"}})
    _run(fake)
    assert fake.hashes["job:x"]["progress"] == "26"
    assert fake.hashes["job:x"]["message"]


def test_non_hash_keys_are_skipped():
    """Another key type starting with `job:` would kill the sweep with WRONGTYPE."""
    fake = _FakeRedis(
        hashes={"job:ok": {"job_id": "ok", "status": "STARTED"}},
        strings={"job:weird": "not-a-hash"},
    )
    assert _run(fake) == ["ok"]


def test_the_two_stores_are_cleaned_together():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "celery_app" / "gpu_hooks.py").read_text(encoding="utf-8")
    assert "reconcile_orphans()" in src
    assert "reconcile_orphan_job_hashes()" in src


def test_authoritative_reader_and_cleaner_agree():
    """Every status `_live_status` reads as running must be cleared.

    A value added on only one side leaves a job stuck in that state forever.
    """
    from server.core.job_reconcile import NON_TERMINAL
    from server.core.queue_view import RUNNING_STATUSES

    assert {s.lower() for s in RUNNING_STATUSES} <= NON_TERMINAL
