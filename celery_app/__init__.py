"""Celery application — queues, configuration and task discovery."""

from celery import Celery, _state

from celery_app.config import CeleryConfig

app = Celery("koni_forge")
app.config_from_object(CeleryConfig)
_state.set_default_app(app)

TASK_MODULES = [
    "celery_app.tasks.data_tasks",
    "celery_app.tasks.kbd_tasks",
    "celery_app.tasks.model_tasks",
    "celery_app.tasks.train_tasks",
]
app.autodiscover_tasks(TASK_MODULES, force=True)

# Importing this connects the GPU lease hooks to the celery signals.
from celery_app import gpu_hooks  # noqa: E402,F401
