"""Non-blocking wrapper around celery inspect.

`celery ... control.inspect(timeout=1).active()` broadcasts to the broker and
holds the thread while it waits. Called directly inside an `async def` handler,
that second was a second in which the whole app stopped: it runs
`uvicorn --workers 1`, so there is exactly one event loop.

    redis KEYS gen_job_src:*             0.001s
    celery inspect(timeout=1).active()   1.027s   <- the bottleneck
    redis scan job:* (296 keys)          0.003s

Multiplied by the front-end polling rate, 38% of the loop was blocked even
while idle (68 of 180 seconds).

**A failure is `None`, not an empty set.** Callers depend on the difference:
``None`` skips the zombie check (fail open, so a live job is not killed),
while ``set()`` is trusted as "genuinely nothing running". Turning a failure
into an empty set would declare every job a zombie while the worker is fine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("core.celery_probe")

#: Broker response timeout in seconds, the value all four callers used.
DEFAULT_TIMEOUT = 1.0


def _inspect(timeout: float):
    """A celery Inspect handle. Missing celery or bad config is the caller's problem."""
    from celery_app import app as celery_app

    return celery_app.control.inspect(timeout=timeout)


def task_ids(active_map: Optional[Dict[str, Any]]) -> Set[str]:
    """``inspect().active()`` response to a set of task ids.

    The response is ``{worker: [{"id": ...}, ...]}``, and a worker may return
    ``None``, which is accepted too.
    """
    ids: Set[str] = set()
    for task_list in (active_map or {}).values():
        for task in task_list or []:
            if isinstance(task, dict):
                ids.add(str(task.get("id", "")))
    return ids


def active_task_ids(timeout: float = DEFAULT_TIMEOUT) -> Optional[Set[str]]:
    """Task ids currently running. **This blocks — call it from a thread.**

    Returns:
        The id set on success (possibly empty), or **``None`` on failure**.
        Callers skip the zombie check when it is ``None`` (fail open).
    """
    try:
        return task_ids(_inspect(timeout).active())
    except Exception as exc:  # broker down, celery missing, or timeout
        logger.warning("celery inspect failed — zombie check skipped: %s", exc)
        return None


