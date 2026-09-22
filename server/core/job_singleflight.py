"""Whether this instance may run a job (:func:`should_run`):
  * No owner -> run (first).
  * The owner is **us** -> run (re-entry, the normal flow of one instance).
  * Someone else owns it but **the heartbeat is stale** -> run (take over).
  * Someone else owns it and is **alive** -> skip (duplicate).

Being heartbeat-based means a genuinely dead worker does not lock a job forever.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_STALE_SECONDS = 1800

RUN = "run"
SKIP_DUPLICATE = "skip_duplicate"


def should_run(
    owner_token: Optional[str],
    my_token: str,
    owner_heartbeat_ts: Optional[float],
    now_ts: float,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
) -> str:
    """Whether this instance may run the job: :data:`RUN` or :data:`SKIP_DUPLICATE`.

    Args:
        owner_token: the recorded owner token, or ``None``.
        my_token: this instance's token.
        owner_heartbeat_ts: when the owner last reported alive (epoch seconds).
        now_ts: now (epoch seconds).
        stale_seconds: no heartbeat for longer than this counts as dead.
    """
    if not owner_token:
        return RUN  # first run
    if owner_token == my_token:
        return RUN  # re-entry, we are the owner
    if is_stale(owner_heartbeat_ts, now_ts, stale_seconds):
        return RUN  # the owner is dead, take over
    return SKIP_DUPLICATE  # a live owner holds it


def is_stale(heartbeat_ts: Optional[float], now_ts: float, stale_seconds: float = DEFAULT_STALE_SECONDS) -> bool:
    """Whether the heartbeat is stale. A missing or invalid value counts as stale,

    so an unreadable heartbeat cannot lock a job forever — a permanent lock is a
    worse failure than a duplicate run.
    """
    if heartbeat_ts is None:
        return True
    try:
        return (float(now_ts) - float(heartbeat_ts)) > float(stale_seconds)
    except (TypeError, ValueError):
        return True
