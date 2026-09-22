"""Assemble the queue view the badge and popover render.

Everything shown already exists in ``*:state:*`` (run and wait state), the GPU
leases (who holds the device) and ``decide_admission`` (the wait reason). This
module only joins them into one screenful; it collects nothing new.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

# Statuses counted as running (compared upper case).
RUNNING_STATUSES = frozenset({"STARTED", "PROGRESS", "RUNNING"})
WAITING_STATUSES = frozenset({"QUEUED", "PENDING", "RETRY"})

# Rows shown at most. The popover is for skimming; a long list is worse.
MAX_ROWS = 12

# Internal kind to display label.
KIND_LABELS = {
    "training": "Training",
    "train": "Training",
    "domain_eval": "Evaluation",
    "kbd_probe": "KBD",
    "pipeline": "Pipeline",
    "data_gen": "Generation",
}


def kind_label(kind: Optional[str]) -> str:
    key = (kind or "").strip().lower()
    return KIND_LABELS.get(key, key or "Job")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def elapsed_seconds(started_at: Any, now_ts: float) -> Optional[int]:
    """Start time (epoch seconds or ISO string) to elapsed seconds. None on failure."""
    ts = _num(started_at)
    if ts is None and isinstance(started_at, str) and started_at.strip():
        try:
            from datetime import datetime

            text = started_at.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            ts = dt.timestamp()
        except (ValueError, TypeError):
            return None
    if ts is None:
        return None
    return max(0, int(now_ts - ts))


def format_elapsed(seconds: Optional[int]) -> str:
    """Human-readable elapsed time. None gives an empty string.

    45s / 12m / 2h 14m / 3d 5h
    """
    if seconds is None:
        return ""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        # Drop a zero minutes component.
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def waiting_reason(state: Mapping[str, Any], has_running: bool) -> str:
    """Why a job is waiting.

    The scheduler's own reason (`admission_reason`) is used verbatim, so the
    screen never says something different from the decision. Without one, a
    short phrase derived from the state.
    """
    reason = state.get("admission_reason") or state.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "starts when the current job finishes" if has_running else "waiting"


def build_rows(
    states: Iterable[Mapping[str, Any]],
    now_ts: float,
    lease_by_job: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> tuple[List[dict], List[dict]]:
    """State records to (running rows, waiting rows), longest-running first.

    Args:
        states: records shaped ``{kind, job_id, user_id, status, started_at}``.
        now_ts: current epoch seconds, injected so this stays pure.
        lease_by_job: job_id to lease, used to attach device info to running rows.
    """
    leases = lease_by_job or {}
    running: List[dict] = []
    waiting: List[dict] = []

    for state in states:
        if not isinstance(state, Mapping):
            continue
        status = str(state.get("status") or "").upper()
        job_id = str(state.get("job_id") or "")
        kind = str(state.get("kind") or "")
        elapsed = elapsed_seconds(state.get("started_at") or state.get("created_at"), now_ts)

        row = {
            "kind": kind,
            "label": kind_label(kind),
            "job_id": job_id,
            "user_id": str(state.get("user_id") or ""),
            "elapsed_s": elapsed,
            "elapsed": format_elapsed(elapsed),
        }

        if status in RUNNING_STATUSES:
            try:
                from server.core.job_eta import format_eta, running_eta

                eta = running_eta(state.get("live_message"), float(elapsed or 0), state)
                if eta:
                    row.update(eta)
                    row["eta"] = format_eta(eta.get("eta_s"))
            except Exception:  # noqa: BLE001 — a failed ETA must not hide the queue
                pass
            lease = leases.get(job_id) or {}
            indices = lease.get("indices") or lease.get("gpu_indices")
            if indices:
                try:
                    row["gpu_indices"] = [int(i) for i in indices]
                except (TypeError, ValueError):
                    pass
            row["status"] = "running"
            running.append(row)
        elif status in WAITING_STATUSES:
            row["status"] = "waiting"
            # Carry the raw reason along so the scheduler wording is not lost.
            row["_state"] = state
            waiting.append(row)

    has_running = bool(running)
    for row in waiting:
        row["reason"] = waiting_reason(row.pop("_state", {}), has_running)

    # Default order is longest wait first; the real admission order overrides it below.
    waiting.sort(key=lambda r: r.get("elapsed_s") or 0, reverse=True)

    try:
        from server.core.gpu_waitq import snapshot as _wait_snapshot

        snap = list(_wait_snapshot())
        rank = {w.job_id: i for i, w in enumerate(sorted(snap, key=lambda x: x.order_key))}
        moved = {w.job_id: w.moved_by for w in snap if w.moved_by}
        for row in waiting:
            who = moved.get(row.get("job_id", ""))
            if who:
                row["moved_by"] = who
        if rank:
            # Entries not in the queue (just left, or unregistered) go last.
            waiting.sort(key=lambda r: rank.get(r.get("job_id", ""), len(rank)))
    except Exception:  # noqa: BLE001 — on failure keep the elapsed-time order
        pass

    try:
        from server.core.job_eta import format_eta, start_eta

        eta_s = start_eta([r.get("eta_s") for r in running])
        if waiting and eta_s is not None:
            waiting[0]["start_eta_s"] = round(eta_s, 1)
            waiting[0]["start_eta"] = format_eta(eta_s)
    except Exception:  # noqa: BLE001
        pass

    running.sort(key=lambda r: r.get("elapsed_s") or 0, reverse=True)
    # `waiting` is already in real admission order; re-sorting by elapsed time
    # would push a job an operator moved up back down the list.
    return running[:MAX_ROWS], waiting[:MAX_ROWS]


def build_gpu_rows(gpus: Optional[Sequence[Mapping[str, Any]]]) -> List[dict]:
    """One-line device summary. The screen uses only utilisation and idle state."""
    out: List[dict] = []
    for gpu in gpus or []:
        if not isinstance(gpu, Mapping):
            continue
        phys = gpu.get("physical") if isinstance(gpu.get("physical"), Mapping) else gpu
        used = _num(phys.get("memory_used_mb")) or 0.0
        total = _num(phys.get("memory_total_mb")) or 0.0
        util = _num(phys.get("utilization_pct"))
        out.append(
            {
                "index": int(gpu.get("index") or phys.get("index") or 0),
                "memory_used_mb": round(used),
                "memory_total_mb": round(total),
                "utilization_pct": round(util) if util is not None else None,
                "busy": used > 1024,  # over 1GB counts as in use, ignoring driver overhead
            }
        )
    return out


def build_queue_view(
    *,
    counts: Mapping[str, Any],
    running: Sequence[Mapping[str, Any]],
    waiting: Sequence[Mapping[str, Any]],
    gpus: Sequence[Mapping[str, Any]],
    mode: str,
    hold_message: str = "",
) -> dict:
    """Assemble the final response, preserving the existing top-level keys."""
    payload = dict(counts)
    payload.update(
        {
            "mode": mode,
            "running": list(running),
            "waiting": list(waiting),
            "gpus": list(gpus),
            "hold": str(hold_message or ""),
        }
    )
    return payload
