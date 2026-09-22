"""Single entry point for GPU jobs.

Every GPU task (training, indexing) is dispatched through ``submit_gpu_task`` so
submission time is recorded in one place and the worker-side GPU hooks see a
uniform task signature.
"""

from typing import Any, Callable, Optional


def submit_gpu_task(
    task_func: Callable,
    *,
    consumer_kind: Optional[str] = None,
    task_id: Optional[str] = None,
    **params,
) -> Any:
    """Dispatch a Celery GPU task on the ``gpu`` queue.

    Args:
        task_func: Celery task (training / indexing)
        consumer_kind: kind recorded in the job timeline
        task_id: explicit Celery task id (job_id == task_id keeps stop / zombie logic keyed)
        **params: parameters passed to the task

    Returns:
        AsyncResult — handle for revoke etc.
    """
    sig = task_func.si(**params)
    if task_id:
        return sig.apply_async(queue="gpu", task_id=task_id)
    return sig.apply_async(queue="gpu")


