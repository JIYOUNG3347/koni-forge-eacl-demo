"""Every worker task must be a registered celery task.

A decorator separated from its function — an edit landing between the two —
leaves a plain function behind. It still imports and still passes a unit test
that calls it directly, but dispatch fails at runtime with
``'function' object has no attribute 'si'``.
"""

from __future__ import annotations

import pytest

import celery_app

TASKS = [
    "celery_app.tasks.data_tasks.index_documents_task",
    "celery_app.tasks.kbd_tasks.kbd_probe_task",
    "celery_app.tasks.model_tasks.download_hf_task",
    "celery_app.tasks.train_tasks.train_sft_task",
]


@pytest.mark.parametrize("name", TASKS)
def test_the_task_is_registered(name):
    assert name in celery_app.app.tasks, f"{name} is not registered — check its @shared_task"


@pytest.mark.parametrize("name", TASKS)
def test_the_task_can_be_dispatched(name):
    """submit_gpu_task and .apply_async both go through the signature API."""
    task = celery_app.app.tasks[name]
    assert hasattr(task, "si") and hasattr(task, "apply_async")


def test_every_declared_module_contributes_a_task():
    registered = {n for n in celery_app.app.tasks if n.startswith("celery_app.")}
    for module in celery_app.TASK_MODULES:
        assert any(n.startswith(module + ".") for n in registered), f"{module} registered no task"


def test_the_routes_name_real_tasks():
    """A route for a name that is not registered silently sends to the wrong queue."""
    from celery_app.config import CeleryConfig

    for name in CeleryConfig.task_routes:
        assert name in celery_app.app.tasks, f"task_routes names {name}, which is not registered"
