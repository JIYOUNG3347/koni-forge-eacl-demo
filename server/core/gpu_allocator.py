"""GPU leases — the ledger of which job holds the accelerator.

A lease is claimed when a GPU task starts and released when it ends, wired up
by the celery signals in ``celery_app/gpu_hooks.py``. The ledger is kept both
in-process (for this worker) and in Redis (so the API can render the queue).

``GPU_ALLOC_MODE`` selects what the lease means:
  - ``legacy`` (default): observation only, nothing waits.
  - ``managed``: :mod:`server.core.gpu_admission` gates task entry on it.

Redis writes are best effort — observability must never raise into a task.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from server.core import gpu_resources

MODE_LEGACY = "legacy"
MODE_MANAGED = "managed"

LEASE_PREFIX = "gpu:lease:"


@dataclass(frozen=True)
class GpuLease:
    """One claim on the accelerator. ``indices`` is empty when none is used."""

    indices: tuple[int, ...]
    token: str
    consumer_kind: str
    mode: str
    meta: dict[str, Any] = field(default_factory=dict)


_active: dict[str, GpuLease] = {}
_lock = threading.Lock()
_mode_announced = False
# task_id -> lease token; prerun and postrun run in the same worker process.
_task_tokens: dict[str, str] = {}


def _lease_ttl_seconds() -> int:
    """TTL for a shared lease. The end hook normally deletes it; this covers crashes."""
    try:
        return int(os.getenv("TRAIN_TIMEOUT_SECONDS", "28800")) + 3600
    except ValueError:
        return 32400


def lease_to_dict(lease: GpuLease) -> dict[str, Any]:
    return {
        "token": lease.token,
        "consumer_kind": lease.consumer_kind,
        "indices": list(lease.indices),
        "mode": lease.mode,
        "meta": lease.meta,
    }


def lease_from_dict(data: Any) -> Optional[GpuLease]:
    """Deserialise leniently — a malformed entry is None, not an error."""
    if not isinstance(data, dict) or not data.get("token"):
        return None
    try:
        indices = tuple(int(i) for i in (data.get("indices") or []))
    except (TypeError, ValueError):
        indices = ()
    return GpuLease(
        indices=indices,
        token=str(data["token"]),
        consumer_kind=str(data.get("consumer_kind", "")),
        mode=str(data.get("mode", MODE_LEGACY)),
        meta=data.get("meta") if isinstance(data.get("meta"), dict) else {},
    )


def _shared_store(lease: GpuLease) -> None:
    try:
        from server.core.state import _get_redis

        _get_redis().set(
            f"{LEASE_PREFIX}{lease.token}",
            json.dumps(lease_to_dict(lease), ensure_ascii=False),
            ex=_lease_ttl_seconds(),
        )
    except Exception:
        pass


def _shared_drop(token: str) -> None:
    try:
        from server.core.state import _get_redis

        _get_redis().delete(f"{LEASE_PREFIX}{token}")
    except Exception:
        pass


def shared_leases() -> dict[str, GpuLease]:
    """Snapshot of the Redis ledger across processes. Empty dict on failure."""
    out: dict[str, GpuLease] = {}
    try:
        from server.core.state import _get_redis

        r = _get_redis()
        for key in r.scan_iter(match=f"{LEASE_PREFIX}*"):
            raw = r.get(key)
            if not raw:
                continue
            try:
                lease = lease_from_dict(json.loads(raw))
            except (ValueError, TypeError):
                continue
            if lease is not None:
                out[lease.token] = lease
    except Exception:
        return {}
    return out


def get_mode() -> str:
    """Allocation mode from ``GPU_ALLOC_MODE``; unknown values fall back to legacy."""
    m = (os.getenv("GPU_ALLOC_MODE") or MODE_LEGACY).strip().lower()
    return m if m in (MODE_LEGACY, MODE_MANAGED) else MODE_LEGACY


def _topology() -> dict[str, Any]:
    """Device topology from the detected accelerator."""
    from server.core import accelerator

    indices = accelerator.device_indices()
    return {"device_count": len(indices), "device_indices": indices}


def _announce_mode_once(mode: str) -> None:
    """Log the mode once so an unexpected no-op is visible in the log."""
    global _mode_announced
    if _mode_announced:
        return
    _mode_announced = True
    try:
        from server.core.logging import sys_log

        if mode == MODE_MANAGED:
            sys_log("[gpu_allocator] GPU_ALLOC_MODE=managed — admission enforced, jobs may wait")
        else:
            sys_log("[gpu_allocator] GPU_ALLOC_MODE=legacy — ledger only, no admission")
    except Exception:
        pass


def claim(
    consumer_kind: str,
    count: int = 1,
    *,
    topology: Optional[dict[str, Any]] = None,
    job_id: str = "",
    user_id: str = "",
    vram_mb: Optional[float] = None,
    share: bool = False,
) -> GpuLease:
    """Issue a lease for a consumer.

    Both modes record the same ledger entry; only admission differs, and that
    happens at task entry in :func:`gpu_admission.gate_or_retry`.

    Args:
        job_id: job identifier, recorded in the lease meta.
        user_id: owner, used by the per-user concurrency limit.
        vram_mb: estimated memory demand, recorded for the admission policy.
        share: also write to the Redis ledger (best effort).
    """
    mode = get_mode()
    _announce_mode_once(mode)

    topo = topology if topology is not None else _topology()
    indices = gpu_resources.resolve_consumer_gpus({"kind": consumer_kind}, topo, count)

    meta: dict[str, Any] = {"requested_count": count}
    if job_id:
        meta["job_id"] = job_id
    if user_id:
        meta["user_id"] = str(user_id)
    if vram_mb is not None:
        meta["vram_mb"] = round(float(vram_mb))

    lease = GpuLease(
        indices=indices,
        token=uuid.uuid4().hex,
        consumer_kind=consumer_kind,
        mode=mode,
        meta=meta,
    )
    with _lock:
        _active[lease.token] = lease
    if share:
        _shared_store(lease)
    return lease


def own_lease_gpu_count(leases: "Iterable[GpuLease]", job_id: str) -> Optional[int]:
    """Devices held by one job — the input to the GPU-time total in the timeline."""
    if not job_id:
        return None
    for lease in leases:
        if not isinstance(lease, GpuLease):
            continue
        if str((lease.meta or {}).get("job_id") or "") != str(job_id):
            continue
        if lease.indices:
            return len(lease.indices)
        try:
            n = int((lease.meta or {}).get("requested_count") or 0)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None
    return None


def release(token: str) -> None:
    """Release a lease, local and shared. Idempotent; unknown tokens are ignored."""
    with _lock:
        _active.pop(token, None)
    _shared_drop(token)


@contextmanager
def gpu_lease(consumer_kind: str, job_id: str = "", user_id: str = "", admit: bool = False, count: int = 1):
    """Shared lease around an in-process GPU section.

    Args:
        admit: wait for admission before entering. Without it a pipeline run
            could start while another training job holds the device. No-op in
            legacy mode.
    """
    if admit:
        try:
            from server.core.gpu_admission import await_admission

            await_admission(consumer_kind, job_id, user_id)
        except Exception:  # noqa: BLE001 — a failed wait must not block execution
            pass

    token: Optional[str] = None
    try:
        token = claim(consumer_kind, count=max(1, int(count or 1)), job_id=job_id, user_id=user_id, share=True).token
    except Exception:
        token = None
    try:
        yield
    finally:
        if token:
            try:
                release(token)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Celery task lifecycle — celery-free logic; the signals live in gpu_hooks.py
# ---------------------------------------------------------------------------


def task_param(kwargs: Optional[Mapping[str, Any]], key: str) -> Any:
    """Read one parameter from celery task kwargs, flat or nested under ``params``."""
    if not isinstance(kwargs, Mapping):
        return None
    nested = kwargs.get("params")
    if isinstance(nested, Mapping) and nested.get(key) is not None:
        return nested.get(key)
    return kwargs.get(key)


def lease_gpu_count(kwargs: Optional[Mapping[str, Any]], device_count: int = 0) -> int:
    """Devices to record on the lease. Unknown means 1, capped by the device count.

    Over-recording only makes other jobs wait; under-recording invites overlap
    and OOM, so the declared value is trusted.
    """
    try:
        count = int(task_param(kwargs, "num_gpus") or 1)
    except (TypeError, ValueError):
        return 1
    count = max(1, count)
    if device_count > 0:
        count = min(count, device_count)
    return count


def on_gpu_task_start(task_name: str, task_id: Any, kwargs: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Claim a shared lease for a GPU task. None for any other task."""
    kind = gpu_resources.infer_consumer_kind(task_name)
    if not kind or not task_id:
        return None
    try:
        kw = kwargs or {}
        try:
            device_count = int(_topology().get("device_count") or 0)
        except Exception:  # noqa: BLE001 — without the topology we only lose the cap
            device_count = 0
        lease = claim(
            kind,
            count=lease_gpu_count(kw, device_count),
            job_id=str(task_param(kw, "job_id") or ""),
            user_id=str(task_param(kw, "user_id") or ""),
            share=True,
        )
        with _lock:
            _task_tokens[str(task_id)] = lease.token
        return lease.token
    except Exception:
        return None


def on_gpu_task_end(task_id: Any) -> None:
    """Release the lease claimed at start, whether the task succeeded or not."""
    if not task_id:
        return
    with _lock:
        token = _task_tokens.pop(str(task_id), None)
    if token:
        try:
            release(token)
        except Exception:
            pass


def active_leases() -> dict[str, GpuLease]:
    """Snapshot of the in-process ledger, for observation and tests."""
    with _lock:
        return dict(_active)
