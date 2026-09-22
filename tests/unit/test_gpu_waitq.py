"""Admission decides whether a job may run; the wait queue decides who is next.

Without an order, a refused task simply retries on a countdown, so whichever
job polls at the right moment when training ends wins — and a user who waited
half an hour can lose to one that just arrived.
"""

from pathlib import Path

from server.core.gpu_admission import decide_admission
from server.core.gpu_waitq import (
    WAIT_TTL_S,
    Waiter,
    fifo_reason,
    waiter_from_dict,
    waiter_to_dict,
    waiters_ahead,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _w(job_id, at, kind="training", user_id=""):
    return Waiter(job_id=job_id, kind=kind, user_id=user_id, enqueued_at=at)


# ── ordering (pure) ──────────────────────────────────────────────────
def test_earlier_waiter_goes_first():
    waiters = [_w("a", 100.0), _w("b", 200.0)]
    assert [w.job_id for w in waiters_ahead("b", waiters, enqueued_at=200.0)] == ["a"]
    assert waiters_ahead("a", waiters, enqueued_at=100.0) == []


def test_new_arrival_cannot_jump_the_queue():
    """An unregistered job (just arrived) sorts behind every valid waiter."""
    waiters = [_w("a", 100.0), _w("b", 200.0)]
    ahead = waiters_ahead("fresh", waiters, enqueued_at=None)
    assert [w.job_id for w in ahead] == ["a", "b"]


def test_self_is_not_ahead_of_self():
    waiters = [_w("a", 100.0)]
    assert waiters_ahead("a", waiters, enqueued_at=100.0) == []


def test_blocked_user_does_not_block_the_line():
    """An entry whose user is already running cannot be admitted, so it is skipped.

    Otherwise everyone stalls behind an entry that can never be admitted.
    """
    waiters = [_w("a", 100.0, user_id="user01"), _w("b", 200.0, user_id="user02")]
    ahead = waiters_ahead("b", waiters, enqueued_at=200.0, busy_user_ids=["user01"])
    assert ahead == []


def test_order_is_stable_across_nodes():
    """Ties break on job_id so every process sees the same order."""
    waiters = [_w("z", 100.0), _w("a", 100.0)]
    assert [w.job_id for w in waiters_ahead("z", waiters, enqueued_at=100.0)] == ["a"]
    assert waiters_ahead("a", waiters, enqueued_at=100.0) == []


def test_reason_names_the_head():
    reason = fifo_reason([_w("a", 100.0, kind="training", user_id="user01")])
    assert "behind 1 job(s)" in reason and "job=a" in reason and "user01" in reason


# ── serialisation ────────────────────────────────────────────────────
def test_roundtrip():
    w = _w("a", 123.5, kind="kbd_probe", user_id="admin")
    assert waiter_from_dict(waiter_to_dict(w)) == w


def test_broken_payload_is_ignored():
    assert waiter_from_dict(None) is None
    assert waiter_from_dict({}) is None  # no job_id
    assert waiter_from_dict({"job_id": "a", "enqueued_at": "??"}) is None


# ── integration with the admission decision ──────────────────────────
def _decide(kind, job_id, leases=(), waiters=(), enqueued_at=None, user=""):
    return decide_admission(
        kind,
        job_id,
        leases,
        requester_user_id=user,
        waiters=waiters,
        enqueued_at=enqueued_at,
    )


def test_idle_gpu_still_waits_for_earlier_job():
    """An earlier waiter wins even when the device is free — the point of FIFO."""
    d = _decide("training", "late", waiters=[_w("early", 100.0)], enqueued_at=200.0)
    assert d.admit is False
    assert "queued behind" in d.reason


def test_head_of_queue_is_admitted():
    d = _decide("training", "early", waiters=[_w("early", 100.0), _w("late", 200.0)], enqueued_at=100.0)
    assert d.admit is True


def test_no_waiters_admits_immediately():
    assert _decide("training", "solo").admit is True


def test_fifo_does_not_override_resource_denial():
    """Being first still waits when the device is busy, and says so."""
    from server.core.gpu_allocator import GpuLease

    running = [GpuLease(indices=(0,), token="t", consumer_kind="training", mode="managed", meta={"job_id": "j0"})]
    d = _decide("training", "early", leases=running, waiters=[_w("early", 100.0)], enqueued_at=100.0)
    assert d.admit is False
    assert "training is exclusive" in d.reason


def test_non_target_kind_bypasses_queue():
    """Chat must never be blocked behind the training queue."""
    assert _decide("chat", "c1", waiters=[_w("early", 100.0)]).admit is True


# ── wiring ───────────────────────────────────────────────────────────
def test_ttl_outlives_polling_intervals():
    """Missing several heartbeats (celery 30s, in-process 15s) must not drop a waiter."""
    assert WAIT_TTL_S >= 30 * 4


def test_admission_paths_register_and_clear():
    src = (PROJECT_ROOT / "server/core/gpu_admission.py").read_text(encoding="utf-8")
    gate = src.split("def gate_or_retry(")[1].split("def inproc_wait_budget(")[0]
    inproc = src.split("def await_admission(")[1]
    # Both paths must take a place in the queue and give it back on admission.
    for name, body in (("gate_or_retry", gate), ("await_admission", inproc)):
        assert "register_waiter(" in body, name
        assert "clear_waiter(" in body, name
    # The timeout path must release it too, or later jobs wait on a ghost.
    assert inproc.count("clear_waiter(") >= 3


def test_enqueue_time_is_preserved_on_retry():
    """Refreshing the time on every retry pushes a long-waiting job to the back."""
    src = (PROJECT_ROOT / "server/core/gpu_waitq.py").read_text(encoding="utf-8")
    assert "nx=True" in src  # only the first registration writes the time
    assert "r.expire(key, WAIT_TTL_S)" in src  # later ones refresh the TTL only
