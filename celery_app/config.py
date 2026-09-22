"""Celery configuration — Redis broker and result backend, two queues."""

import os

from kombu import Queue

from server.core.celery_timeouts import broker_visibility_timeout
from server.core.host_config import HOST_CONFIG


class CeleryConfig:
    """Worker configuration: host.toml for concurrency, env vars for URLs."""

    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

    task_serializer = "json"
    result_serializer = "json"
    accept_content = ["json"]
    timezone = "UTC"
    enable_utc = True

    task_queues = (
        Queue("default", routing_key="default"),
        Queue("gpu", routing_key="gpu"),
    )
    task_default_queue = "default"
    task_default_routing_key = "default"

    task_routes = {
        "celery_app.tasks.train_tasks.train_sft_task": {"queue": "gpu"},
        "celery_app.tasks.kbd_tasks.kbd_probe_task": {"queue": "gpu"},
        "celery_app.tasks.data_tasks.index_documents_task": {"queue": "default"},
        "celery_app.tasks.model_tasks.download_hf_task": {"queue": "default"},
    }

    worker_concurrency = max(2, HOST_CONFIG.concurrency.max_concurrent_train_jobs)
    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 50  # restart after N tasks to cap memory growth

    task_soft_time_limit = int(os.getenv("TRAIN_TIMEOUT_SECONDS", "7200"))
    task_time_limit = task_soft_time_limit + 300  # hard kill 5 min after the soft limit
    result_expires = 86400

    broker_transport_options = {
        "visibility_timeout": broker_visibility_timeout(task_time_limit),
    }

    task_track_started = True
    task_acks_late = True  # ack after completion so a lost worker redelivers
    worker_send_task_events = True
    task_send_sent_event = True

    result_extended = True

    task_reject_on_worker_lost = True
    task_default_retry_delay = 60
    task_max_retries = 3
