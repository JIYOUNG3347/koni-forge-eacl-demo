"""FIFO by default; only an operator may change the order.

Rather than a priority value (a second axis), one ordering axis is adjusted:
`order_at`, which defaults to `enqueued_at`. With nobody touching it the
behaviour is identical, and no starvation rule is needed.

`enqueued_at` is never touched — the "how long did it actually wait" metric
must stay clean of operator intervention.
"""

from pathlib import Path

from server.core.gpu_waitq import (
    Waiter,
    waiter_from_dict,
    waiter_to_dict,
    waiters_ahead,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _w(job_id: str, at: float, order_at=None, user="admin") -> Waiter:
    return Waiter(job_id=job_id, kind="training", user_id=user, enqueued_at=at, order_at=order_at)


# ── A. Untouched means plain FIFO ────────────────────────────────────
def test_untouched_queue_is_fifo():
    a, b, c = _w("a", 100.0), _w("b", 200.0), _w("c", 300.0)
    assert sorted([c, a, b], key=lambda w: w.order_key) == [a, b, c]


def test_order_key_defaults_to_arrival():
    assert _w("a", 100.0).order_key == (100.0, "a")


def test_serialization_omits_untouched_order():
    """An unmoved entry keeps the original format, compatible with old data."""
    d = waiter_to_dict(_w("a", 100.0))
    assert "order_at" not in d and "moved_by" not in d


def test_round_trip_preserves_adjustment():
    w = Waiter("a", "training", "admin", 200.0, order_at=50.0, moved_by="admin")
    assert waiter_from_dict(waiter_to_dict(w)) == w


def test_broken_order_falls_back_to_arrival():
    """A broken override must not lose the order — it falls back to arrival."""
    w = waiter_from_dict({"job_id": "a", "enqueued_at": 100.0, "order_at": "nonsense"})
    assert w is not None and w.order_at is None and w.order_key == (100.0, "a")










# ── C. The metric is preserved ───────────────────────────────────────
def test_arrival_time_is_never_changed():
    """The wait-time metric must stay clean of operator intervention."""
    w = _w("a", 100.0, order_at=1.0)
    assert w.enqueued_at == 100.0  # moved in order only; the wait time is unchanged


# ── D. Agreement with the admission decision ─────────────────────────
def test_admission_sees_the_adjusted_order():
    """If the decision read the old order, the screen and reality would disagree."""
    q = [_w("a", 100.0), _w("b", 200.0)]
    assert [w.job_id for w in waiters_ahead("b", q, enqueued_at=200.0)] == ["a"]

    # Deferring a puts nothing ahead of b.
    q2 = [_w("a", 100.0, order_at=999.0), _w("b", 200.0)]
    assert waiters_ahead("b", q2, enqueued_at=200.0) == []


def test_caller_passes_sort_at_not_arrival():
    """Passing enqueued_at from the caller ignored the move — that was a real bug."""
    src = (PROJECT_ROOT / "server/core/gpu_admission.py").read_text(encoding="utf-8")
    body = src.split("def _waitq_state(")[1].split("def ")[0]
    assert "w.sort_at" in body and "w.enqueued_at" not in body


# ── E. Permission and wiring guards ──────────────────────────────────




def test_intervention_is_visible():
    """An invisible intervention leaves other users wondering why the order changed."""
    src = (PROJECT_ROOT / "server/core/queue_view.py").read_text(encoding="utf-8")
    assert '"moved_by"' in src
