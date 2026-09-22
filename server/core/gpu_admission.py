"""Admission gate — decides whether a GPU job may start now.

Active only when ``GPU_ALLOC_MODE=managed``; in legacy (the default) every
entry point is a no-op.

Policy:
  * Training is exclusive — any other live batch lease makes it wait.
  * One job per user at a time.
  * Waiting jobs are served in arrival order, so a job that has waited does
    not lose its slot to one that just arrived.
  * An operator can hold all new jobs.
  * Anything the gate cannot decide is admitted — a guard must never block
    legitimate work.

Waiting is implemented as a celery retry with a countdown, so no separate
physical queue is needed. The pure decision (:func:`decide_admission`) is
separated from the I/O (:func:`gate_or_retry`) so it can be unit tested.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

# Consumers that hold the device for a long time — the ones worth gating.
BATCH_EXCLUSIVE_KINDS = frozenset({"training", "kbd_probe"})

DEFAULT_RETRY_S = 30
# In-process waits cannot use task.retry, so they poll. Training runs for
# hours, so the ceiling is generous.
INPROC_WAIT_TIMEOUT_S = 7200
INPROC_WAIT_INTERVAL_S = 15

HOLD_KEY = "scheduler:hold"


@dataclass(frozen=True)
class AdmissionDecision:
    admit: bool
    reason: str = ""
    retry_after_s: int = DEFAULT_RETRY_S


def _lease_view(lease: Any) -> "tuple[str, str, tuple[int, ...], str]":
    """Extract ``(kind, job_id, indices, user_id)`` from a lease or a plain dict."""
    kind = str(
        getattr(lease, "consumer_kind", "") or (lease.get("consumer_kind", "") if isinstance(lease, dict) else "")
    )
    meta = getattr(lease, "meta", None)
    if meta is None and isinstance(lease, dict):
        meta = lease.get("meta")
    job_id = str((meta or {}).get("job_id") or "")
    indices = getattr(lease, "indices", None)
    if indices is None and isinstance(lease, dict):
        indices = lease.get("indices")
    try:
        idx = tuple(int(i) for i in (indices or ()))
    except (TypeError, ValueError):
        idx = ()
    user_id = str((meta or {}).get("user_id") or "")
    return kind, job_id, idx, user_id


def _fifo_gate(
    job_id: str,
    waiters: Sequence[Any],
    enqueued_at: Optional[float],
    others: Sequence[Any],
) -> Optional[AdmissionDecision]:
    """Wait while an older job is still queued, even if the device is free.

    Without this, whichever job happens to poll at the moment training ends
    wins, and a user who waited half an hour loses to one that just arrived.
    A failure here returns None (admit) — only ordering is lost.
    """
    if not waiters:
        return None
    try:
        from server.core.gpu_waitq import fifo_reason, waiters_ahead

        busy_users = [o[3] for o in others if len(o) > 3 and o[3]]
        ahead = waiters_ahead(job_id, waiters, enqueued_at=enqueued_at, busy_user_ids=busy_users)
        if not ahead:
            return None
        return AdmissionDecision(False, reason=fifo_reason(ahead))
    except Exception:  # noqa: BLE001 — an ordering failure must not block execution
        return None


def hold_message(hold: Optional[Mapping[str, Any]]) -> str:
    """Who put the hold on and why, so a stalled queue does not read as an outage."""
    if not isinstance(hold, Mapping) or not hold.get("on"):
        return ""
    who = str(hold.get("by") or "").strip()
    note = str(hold.get("note") or "").strip()
    base = f"An administrator ({who}) is holding new jobs" if who else "An administrator is holding new jobs"
    return f"{base} — {note}" if note else base


def decide_admission(
    kind: str,
    job_id: str,
    active_leases: Iterable[Any],
    *,
    requester_user_id: str = "",
    waiters: Sequence[Any] = (),
    enqueued_at: Optional[float] = None,
    hold: Optional[Mapping[str, Any]] = None,
) -> AdmissionDecision:
    """Pure admission decision over a snapshot of the shared ledger.

    The caller's own lease (already claimed by the prerun hook) is excluded by
    ``job_id``. Without a job_id self-exclusion is impossible, so the decision
    is skipped and the job is admitted.
    """
    if kind not in BATCH_EXCLUSIVE_KINDS:
        return AdmissionDecision(True, reason="not a gated kind")
    if not job_id:
        return AdmissionDecision(True, reason="no job_id — cannot self-exclude, decision skipped")

    hold_reason = hold_message(hold)
    if hold_reason:
        return AdmissionDecision(False, reason=hold_reason)

    others = []
    for lease in active_leases:
        l_kind, l_job, l_idx, l_user = _lease_view(lease)
        if l_kind not in BATCH_EXCLUSIVE_KINDS:
            continue
        if l_job and l_job == job_id:
            continue  # our own lease
        others.append((l_kind, l_job, l_idx, l_user))

    if not others:
        return _fifo_gate(job_id, waiters, enqueued_at, others) or AdmissionDecision(True)

    if requester_user_id:
        mine = [o for o in others if o[3] and o[3] == requester_user_id]
        if mine:
            held = ", ".join(f"{k}(job={j or '?'})" for k, j, _, _ in mine)
            return AdmissionDecision(
                False,
                reason=f"this user already has a GPU job running — starts when it finishes: {held}",
            )

    held = ", ".join(f"{k}(job={j or '?'})" for k, j, _, _ in others)
    if kind == "training":
        return AdmissionDecision(False, reason=f"training is exclusive — GPU jobs active: {held}")
    return AdmissionDecision(False, reason=f"another GPU job is running: {held}")


# ---------------------------------------------------------------------------
# I/O layer
# ---------------------------------------------------------------------------


def get_hold() -> "dict[str, Any]":
    """Global hold state. A lookup failure reads as "not held"."""
    try:
        import json

        from server.core.state import _get_redis

        raw = _get_redis().get(HOLD_KEY)
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _waitq_state(job_id: str) -> "tuple[list[Any], Optional[float]]":
    """``(queue snapshot, my enqueue time)``. Failure is ``([], None)`` — fail open.

    A job not yet registered gets ``None``, and :func:`waiters_ahead` then
    treats every valid waiter as ahead of it. That is what stops a job that
    just arrived from jumping the queue.
    """
    try:
        from server.core.gpu_waitq import snapshot

        waiters = snapshot()
    except Exception:  # noqa: BLE001
        return [], None
    mine = next((w.sort_at for w in waiters if w.job_id == job_id), None)
    return list(waiters), mine


def _own_gpu_count(leases: Any, job_id: str) -> Optional[int]:
    """Devices held by this job. None on failure — observability never blocks."""
    try:
        from server.core.gpu_allocator import own_lease_gpu_count

        return own_lease_gpu_count(leases, job_id)
    except Exception:  # noqa: BLE001
        return None


def gate_or_retry(task: Any, kind: str, job_id: str, user_id: str = "") -> None:
    """Admission gate at the entry of a GPU celery task.

    No-op in legacy mode. When managed and refused, raises ``task.retry`` so
    the message goes back on the queue and never returns to the caller. Any
    error in the decision admits the job.
    """
    from server.core.gpu_allocator import MODE_MANAGED, get_mode, shared_leases
    from server.core.gpu_waitq import clear_waiter, register_waiter

    if get_mode() != MODE_MANAGED:
        return

    try:
        from server.core.logging import sys_log

        waiters, mine_at = _waitq_state(str(job_id or ""))
        # Read the ledger once so the decision and the recorded device count
        # come from the same snapshot.
        leases = list(shared_leases().values())
        decision = decide_admission(
            kind,
            str(job_id or ""),
            leases,
            requester_user_id=str(user_id or ""),
            waiters=waiters,
            enqueued_at=mine_at,
            hold=get_hold(),
        )
    except Exception:
        return  # a failed decision must not block legitimate work

    if decision.admit:
        clear_waiter(str(job_id or ""))  # give up our slot so the next job moves up
        return

    # Hold our place and heartbeat. The original enqueue time is preserved, so
    # retrying does not push us to the back.
    register_waiter(kind, str(job_id or ""), str(user_id or ""))
    sys_log(f"[admission] waiting ({kind}, job={job_id}): {decision.reason} — retry in {decision.retry_after_s}s")
    max_retries = int(os.getenv("GPU_ADMISSION_MAX_RETRIES", "240"))  # 30s x 240 = 2h
    raise task.retry(countdown=decision.retry_after_s, max_retries=max_retries)


def inproc_wait_budget(env: Optional[Mapping[str, str]] = None) -> "tuple[int, int]":
    """``(timeout seconds, poll interval seconds)`` for an in-process wait."""
    if env is None:
        env = os.environ  # type: ignore[assignment]

    def _int(key: str, default: int, lo: int, hi: int) -> int:
        raw = str((env or {}).get(key) or "").strip()
        try:
            value = int(raw) if raw else default
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    return (
        _int("GPU_INPROC_WAIT_TIMEOUT_S", INPROC_WAIT_TIMEOUT_S, 30, 86400),
        _int("GPU_INPROC_WAIT_INTERVAL_S", INPROC_WAIT_INTERVAL_S, 1, 300),
    )


def await_admission(kind: str, job_id: str, user_id: str = "") -> None:
    """Wait for admission around an in-process GPU section.

    Pipeline stages call GPU work as plain functions, so they never pass
    through ``gate_or_retry``. Without this they would appear in the ledger —
    making others avoid them — while avoiding no one themselves.

    Outside celery there is no ``task.retry``, so this polls. Legacy mode or a
    failed decision passes straight through.
    """
    from server.core.gpu_allocator import MODE_MANAGED, get_mode, shared_leases

    if get_mode() != MODE_MANAGED:
        return

    from server.core.gpu_waitq import clear_waiter, register_waiter
    from server.core.logging import sys_log

    timeout_s, interval_s = inproc_wait_budget()
    started = time.time()
    announced = False

    while True:
        try:
            waiters, mine_at = _waitq_state(str(job_id or ""))
            leases = list(shared_leases().values())
            decision = decide_admission(
                kind,
                str(job_id or ""),
                leases,
                requester_user_id=str(user_id or ""),
                waiters=waiters,
                enqueued_at=mine_at,
                hold=get_hold(),
            )
        except Exception:  # noqa: BLE001 — a failed decision must not block the pipeline
            clear_waiter(str(job_id or ""))
            return

        if decision.admit:
            clear_waiter(str(job_id or ""))
            if announced:
                sys_log(f"[admission] admitted ({kind}, job={job_id}) after {int(time.time() - started)}s")
            return

        # Hold our place; the heartbeat refreshes every poll and expires if we die.
        register_waiter(kind, str(job_id or ""), str(user_id or ""))

        elapsed = time.time() - started
        if elapsed >= timeout_s:
            # Do not disguise a timeout as an admission: say so and continue.
            # Warning and proceeding is safer for a pipeline than failing it.
            sys_log(
                f"[admission] wait limit exceeded ({kind}, job={job_id}) after {int(elapsed)}s — "
                f"proceeding without admission. Last reason: {decision.reason}",
                level="WARNING",
            )
            # Give up the slot too, or later jobs wait on one already running.
            clear_waiter(str(job_id or ""))
            return

        if not announced:
            announced = True
            sys_log(f"[admission] waiting ({kind}, job={job_id}): {decision.reason}")
        time.sleep(interval_s)
