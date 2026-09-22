"""Read model of who is using the accelerator right now.

Pure transforms only: the I/O (Redis scan, device probe) belongs to the
callers, which keeps this unit-testable without fastapi or redis.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Workload classes.
#   interactive — user-facing chat, latency sensitive
#   batch       — long running (training)
#   transient   — loaded and released quickly (KBD probe)
#   background  — side work (embedding / indexing)
CLASS_INTERACTIVE = "interactive"
CLASS_BATCH = "batch"
CLASS_TRANSIENT = "transient"
CLASS_BACKGROUND = "background"

# Consumer kind -> workload class. Register new GPU consumers here.
CONSUMER_TAXONOMY: dict[str, str] = {
    "training": CLASS_BATCH,
    "kbd_probe": CLASS_TRANSIENT,
    "chat": CLASS_INTERACTIVE,
    "embedding": CLASS_BACKGROUND,
}

# Celery task name -> consumer kind.
GPU_TASK_KINDS: dict[str, str] = {
    "celery_app.tasks.train_tasks.train_sft_task": "training",
    "celery_app.tasks.kbd_tasks.kbd_probe_task": "kbd_probe",
}

# Statuses that count as occupying the device. Training writes upper case,
# pipelines write lower case; accept both.
RUNNING_UPPER = {"STARTED", "PROGRESS", "RUNNING"}


def classify(kind: str) -> Optional[str]:
    """Workload class of a consumer kind, or None if it is not registered."""
    return CONSUMER_TAXONOMY.get(kind)


def infer_consumer_kind(task_name: str, default: Optional[str] = None) -> Optional[str]:
    """Consumer kind for a celery task name."""
    return GPU_TASK_KINDS.get(str(task_name or ""), default)


def resolve_consumer_gpus(consumer: dict[str, Any], topology: dict[str, Any], count: int = 1) -> tuple[int, ...]:
    """Device indices a consumer occupies. Empty when it uses no accelerator."""
    devices = list(topology.get("device_indices") or [])
    if not devices or classify(consumer.get("kind", "")) is None:
        return ()
    try:
        n = max(1, int(count))
    except (TypeError, ValueError):
        n = 1
    return tuple(devices[:n])


def resolve_consumer_gpu(consumer: dict[str, Any], topology: dict[str, Any]) -> Optional[int]:
    """First device index a consumer occupies, or None."""
    indices = resolve_consumer_gpus(consumer, topology)
    return indices[0] if indices else None


def build_gpu_resources(
    topology: dict[str, Any],
    gpu_metrics_by_index: dict[int, Any],
    active_consumers: list[dict],
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the read model.

    Args:
        topology: ``{device_count, backend, device_indices}``
        gpu_metrics_by_index: device index -> probe entry (None when unknown)
        active_consumers: live records, each with at least ``{"kind": ...}``
        generated_at: ISO timestamp, injected so tests stay deterministic
    """
    device_indices = topology.get("device_indices") or []

    by_index: dict[int, list[dict[str, Any]]] = {i: [] for i in device_indices}
    unbound: list[dict[str, Any]] = []
    for c in active_consumers:
        enriched = dict(c)
        workload_class = classify(c.get("kind", ""))
        if workload_class is not None:
            enriched["workload_class"] = workload_class
        idx = resolve_consumer_gpu(c, topology)
        enriched["gpu_index"] = idx
        if idx is None or idx not in by_index:
            unbound.append(enriched)
        else:
            by_index[idx].append(enriched)

    gpus = [
        {
            "index": i,
            "physical": gpu_metrics_by_index.get(i),  # None means not observed
            "active_consumers": by_index.get(i, []),
        }
        for i in device_indices
    ]

    return {
        "topology": topology,
        "gpus": gpus,
        "unbound_consumers": unbound,
        # Say what is measured and what is only inferred from the taxonomy.
        "coverage": {
            "live_observed": ["training"],
            "class_mapped_only": ["chat", "embedding", "kbd_probe"],
            "note": "chat and embedding run in-process, so they have no individual live state.",
        },
        "generated_at": generated_at,
    }


def is_running(status: Optional[str]) -> bool:
    s = status or ""
    return s.upper() in RUNNING_UPPER


def scan_active_consumers(r) -> list[dict]:
    """Scan live training state into consumer records. ``r`` is a redis client."""
    consumers: list[dict] = []
    for key in r.scan_iter("train:state:*"):
        try:
            meta = json.loads(r.get(key) or "{}")
        except Exception:
            continue
        if not is_running(meta.get("status")):
            continue
        consumers.append(
            {
                "kind": "training",
                "job_id": meta.get("job_id"),
                "user_id": meta.get("user_id"),
                "status": meta.get("status"),
                "model_name": meta.get("model_name"),
                "method": meta.get("method"),
            }
        )
    return consumers
