"""Keep the broker from redelivering a task that is still running.

The Redis broker defaults ``visibility_timeout`` to 3600s. With
``task_acks_late=True`` a task is acked only after it finishes, and if no ack
arrives in that window the broker assumes the worker died and requeues the
message. GPU training and evaluation routinely exceed an hour, so without this
every long job runs twice.

**Invariant: ``visibility_timeout`` > ``task_time_limit``.**
The broker must not redeliver before the task is hard-killed, which makes
"alive but presumed dead" impossible by construction.

Deliberate trade-off: when a worker really does die, redelivery is that much
slower. A GPU job silently running twice (contention, overwritten state, OOM)
is far worse.
"""

from __future__ import annotations

# Grace after the hard kill, in seconds, for shutdown and ack propagation.
DEFAULT_MARGIN_SECONDS = 600

# Floor for a misconfigured setting; must exceed the 3600 default.
MIN_VISIBILITY_TIMEOUT = 7200


def broker_visibility_timeout(task_time_limit: object, margin: int = DEFAULT_MARGIN_SECONDS) -> int:
    """Broker visibility_timeout in seconds, always beyond ``task_time_limit``.

    Args:
        task_time_limit: the celery hard-kill limit, in seconds.
        margin: grace before redelivery, in seconds.
    """
    try:
        limit = int(task_time_limit)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        limit = 0
    try:
        pad = max(0, int(margin))
    except (TypeError, ValueError):
        pad = DEFAULT_MARGIN_SECONDS
    return max(limit + pad, MIN_VISIBILITY_TIMEOUT)


