"""Celery signal wiring for GPU leases.

The lease has to cover the whole task run, so it is claimed in ``task_prerun``
and released in ``task_postrun``. The decisions live in the celery-free
:mod:`server.core.gpu_allocator`; this module only connects the signals.

Which tasks count as GPU tasks comes from ``gpu_resources.GPU_TASK_KINDS`` —
anything else is a no-op. No hook failure may block a task.
"""

from celery.signals import task_postrun, task_prerun, worker_ready

from server.core import gpu_allocator


@task_prerun.connect
def _gpu_lease_start(task_id=None, task=None, kwargs=None, **_extra) -> None:
    try:
        gpu_allocator.on_gpu_task_start(getattr(task, "name", ""), task_id, kwargs or {})
    except Exception:
        pass


@task_postrun.connect
def _gpu_lease_end(task_id=None, kwargs=None, task=None, **_extra) -> None:
    try:
        gpu_allocator.on_gpu_task_end(task_id)
    except Exception:
        pass


@worker_ready.connect
def _reconcile_orphan_jobs(**_extra) -> None:
    """At worker start nothing is running, so any non-terminal state is stale.

    ``task_acks_late=True`` means messages still in the broker are redelivered
    and rewrite their own state, so clearing here is safe.
    """
    try:
        from server.core.job_reconcile import reconcile_orphan_job_hashes, reconcile_orphans
        from server.core.logging import sys_log

        def _report(label: str, cleaned: list) -> None:
            if not cleaned:
                return
            shown = ", ".join(cleaned[:5]) + (f" and {len(cleaned) - 5} more" if len(cleaned) > 5 else "")
            sys_log(f"[reconcile] cleared {len(cleaned)} {label}: {shown}")

        _report("jobs lost to a worker restart", reconcile_orphans())
        _report("jobs lost mid-progress", reconcile_orphan_job_hashes())
    except Exception:  # noqa: BLE001 — cleanup failure must not block worker start
        pass
