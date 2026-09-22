"""Wait queue — the arrival order behind the admission gate.

Admission decides *whether* a job may run; this decides *who goes first*.
Without it a refused task just retries on a countdown, so whichever job
happens to poll at the moment training ends wins, and a user who waited half
an hour can lose to one that just arrived.

Design:
  * ``enqueued_at`` is the position. Refreshing it on every retry would push a
    long-waiting job to the back, so only the first registration writes the
    time; later ones refresh the TTL only (heartbeat).
  * A job waits while an earlier one is queued, even when the device is free.
    New arrivals are included in the comparison, so they cannot jump ahead.
  * Head-of-line blocking is avoided: an entry that cannot be admitted for
    another reason (its user already has a job running) is skipped, because
    waiting for it would stall everyone.
  * Dead waiters disappear when their heartbeat expires. Nothing cleans up
    after a process that died.
  * A lookup failure returns an empty list: losing the ordering is better than
    blocking admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

WAIT_PREFIX = "gpu:wait:"

# Heartbeat expiry. The celery path refreshes every 30s and the in-process one
# every 15s, so this leaves room to miss several and still stay queued.
WAIT_TTL_S = 180


@dataclass(frozen=True)
class Waiter:
    """One entry in the wait queue."""

    job_id: str
    kind: str
    user_id: str
    enqueued_at: float
    order_at: Optional[float] = None
    # Who moved this entry. An invisible reorder reads as a bug.
    moved_by: str = ""

    @property
    def sort_at(self) -> float:
        """Time used for ordering: the override if set, otherwise arrival."""
        return self.enqueued_at if self.order_at is None else self.order_at

    @property
    def order_key(self) -> "tuple[float, str]":
        """Sort key. Ties break on job_id so every process sees one order."""
        return (self.sort_at, self.job_id)


def waiter_from_dict(data: Any) -> Optional[Waiter]:
    """Redis value to Waiter. A malformed entry is None and gets ignored."""
    if not isinstance(data, dict):
        return None
    job_id = str(data.get("job_id") or "")
    if not job_id:
        return None
    try:
        enqueued_at = float(data.get("enqueued_at") or 0.0)
    except (TypeError, ValueError):
        return None
    order_at: Optional[float]
    try:
        raw_order = data.get("order_at")
        order_at = float(raw_order) if raw_order is not None else None
    except (TypeError, ValueError):
        order_at = None  # a broken override falls back to arrival order
    return Waiter(
        job_id=job_id,
        kind=str(data.get("kind") or ""),
        user_id=str(data.get("user_id") or ""),
        enqueued_at=enqueued_at,
        order_at=order_at,
        moved_by=str(data.get("moved_by") or ""),
    )


def waiter_to_dict(w: Waiter) -> dict:
    out = {
        "job_id": w.job_id,
        "kind": w.kind,
        "user_id": w.user_id,
        "enqueued_at": w.enqueued_at,
    }
    # Unmoved entries carry no key, which keeps "not moved" distinct from
    # "moved to exactly its arrival time".
    if w.order_at is not None:
        out["order_at"] = w.order_at
    if w.moved_by:
        out["moved_by"] = w.moved_by
    return out


def waiters_ahead(
    job_id: str,
    waiters: Iterable[Waiter],
    *,
    enqueued_at: Optional[float] = None,
    busy_user_ids: Sequence[str] = (),
) -> "list[Waiter]":
    """Valid waiters ahead of this job, in order.

    Args:
        job_id: the job being judged.
        waiters: wait queue snapshot.
        enqueued_at: our own enqueue time. None means not registered yet (a
            fresh arrival), and then every valid waiter counts as ahead.
        busy_user_ids: users currently holding the device. Their entries
            cannot be admitted anyway, so they are skipped to avoid a stall.
    """
    busy = {u for u in busy_user_ids if u}
    mine_key = (enqueued_at, str(job_id)) if enqueued_at is not None else None

    ahead: list[Waiter] = []
    for w in waiters:
        if w.job_id == job_id:
            continue
        if w.user_id and w.user_id in busy:
            continue  # cannot be admitted — waiting for it stalls both
        if mine_key is None or w.order_key < mine_key:
            ahead.append(w)
    ahead.sort(key=lambda w: w.order_key)
    return ahead


def fifo_reason(ahead: Sequence[Waiter]) -> str:
    """Wait reason, reused verbatim by the UI so both say the same thing."""
    head = ahead[0]
    who = f"{head.kind or 'GPU job'}(job={head.job_id or '?'}"
    if head.user_id:
        who += f", {head.user_id}"
    who += ")"
    return f"queued behind {len(ahead)} job(s): {who}"


# ── Redis I/O (lazy) ─────────────────────────────────────────────────
def _redis() -> Any:
    from server.core.state import _get_redis

    return _get_redis()


def register_waiter(kind: str, job_id: str, user_id: str = "", now_ts: Optional[float] = None) -> None:
    """Register (idempotent) and refresh the heartbeat.

    The first enqueue time is preserved: rewriting it on every retry would
    push the longest-waiting job to the back, inverting the order.
    """
    if not job_id:
        return
    try:
        import json
        import time

        payload = json.dumps(
            waiter_to_dict(
                Waiter(
                    job_id=str(job_id),
                    kind=str(kind or ""),
                    user_id=str(user_id or ""),
                    enqueued_at=float(now_ts if now_ts is not None else time.time()),
                )
            ),
            ensure_ascii=False,
        )
        r = _redis()
        key = f"{WAIT_PREFIX}{job_id}"
        r.set(key, payload, ex=WAIT_TTL_S, nx=True)  # first write only, to keep the time
        r.expire(key, WAIT_TTL_S)  # heartbeat

    except Exception:
        pass


def clear_waiter(job_id: str) -> None:
    """Leave the queue (idempotent), on admission or on giving up."""
    if not job_id:
        return
    try:
        _redis().delete(f"{WAIT_PREFIX}{job_id}")
    except Exception:
        pass


def snapshot() -> "list[Waiter]":
    """Wait queue snapshot in order. A lookup failure returns an empty list."""
    out: list[Waiter] = []
    try:
        import json

        r = _redis()
        for key in r.scan_iter(match=f"{WAIT_PREFIX}*"):
            raw = r.get(key)
            if not raw:
                continue
            try:
                w = waiter_from_dict(json.loads(raw))
            except (ValueError, TypeError):
                continue
            if w is not None:
                out.append(w)
    except Exception:
        return []
    out.sort(key=lambda w: w.order_key)
    return out
