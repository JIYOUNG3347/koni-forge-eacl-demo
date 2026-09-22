"""Clear job state left behind by a worker restart.

When a worker restarts, the message it was handling is gone but `train:state:*`
remains. Nothing rewinds it, so it stays `queued` forever and the queue view
reads it as waiting.

Filtering by age would be wrong: a job legitimately waiting three hours behind
a long training run would vanish from the screen with no way to track it. So
the decision uses a fact instead of a clock: if the worker has just started,
nothing is running.

With `task_acks_late=True` any message still in the broker is redelivered and
rewrites its own state, which is what makes clearing at startup safe.

`job:` is a hash rather than a JSON string, and its statuses are upper case
(`STARTED`, `SUCCESS`), so the storage shape differs. The decision
(`needs_reconcile`) ignores case, so both share it.
"""

from typing import Any, Iterable, Mapping

# Statuses to clear (case-insensitive). Terminal statuses are left alone.
NON_TERMINAL = frozenset({"queued", "pending", "retry", "started", "progress", "running"})

# Status after clearing, distinct from failure: not "ran and failed" but
# "never ran and was lost".
INTERRUPTED = "interrupted"
INTERRUPTED_MESSAGE = "Lost to a worker restart. Please run it again."

STATE_PREFIXES = ("train:state:", "eval:state:")

JOB_PREFIX = "job:"


def needs_reconcile(state: Any) -> bool:
    """Whether this status needs clearing. Terminal or unrecognised is False."""
    if not isinstance(state, Mapping):
        return False
    return str(state.get("status") or "").strip().lower() in NON_TERMINAL


def reconciled(state: Mapping[str, Any]) -> "dict[str, Any]":
    """The cleared state. Original fields are kept and only status and message
    change, so the user can still see what the job was and run it again."""
    out = dict(state)
    out["status"] = INTERRUPTED
    out["message"] = INTERRUPTED_MESSAGE
    return out


def summarize(states: Iterable[Mapping[str, Any]]) -> "list[str]":
    """job_ids that need clearing, so the log can say what was cleared."""
    return [str(s.get("job_id") or "?") for s in states if needs_reconcile(s)]


# ── Redis IO(lazy) ───────────────────────────────────────────────────
def reconcile_orphans(prefixes: Iterable[str] = STATE_PREFIXES) -> "list[str]":
    """Clear non-terminal states to interrupted and return the job_ids cleared.

    Failure is harmless: without the cleanup the ghosts simply remain visible.
    The state is never deleted — the user must be able to see what the job was.
    """
    cleaned: list[str] = []
    try:
        import json as _json

        from server.core.state import _get_redis

        r = _get_redis()
        for prefix in prefixes:
            for key in r.scan_iter(match=f"{prefix}*"):
                raw = r.get(key)
                if not raw:
                    continue
                try:
                    state = _json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not needs_reconcile(state):
                    continue
                r.set(key, _json.dumps(reconciled(state), ensure_ascii=False))
                cleaned.append(str(state.get("job_id") or key))
    except Exception:  # noqa: BLE001 — a cleanup failure must not block worker start
        return cleaned
    return cleaned


def reconcile_orphan_job_hashes(prefix: str = JOB_PREFIX) -> "list[str]":
    """The `job:` hash is a separate store from the submission metadata, and the
    screen treats it as authoritative for "running" (`_live_status`). Clearing
    only one of the two leaves the ghost in place.

    Nothing is deleted, so the user can still trace the job. Keys that are not
    hashes are skipped.
    """
    cleaned: list[str] = []
    try:
        from server.core.state import _get_redis

        r = _get_redis()
        for key in r.scan_iter(match=f"{prefix}*"):
            try:
                if r.type(key) != "hash":
                    continue
                fields = r.hgetall(key) or {}
            except Exception:  # noqa: BLE001 — one bad key must not stop the sweep
                continue
            if not needs_reconcile(fields):
                continue
            # Keep the original fields and change only these two, as above.
            r.hset(key, mapping={"status": INTERRUPTED, "message": INTERRUPTED_MESSAGE})
            cleaned.append(str(fields.get("job_id") or key[len(prefix) :] or key))
    except Exception:  # noqa: BLE001 — a cleanup failure must not block worker start
        return cleaned
    return cleaned
