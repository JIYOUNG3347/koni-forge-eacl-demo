"""System router — /api/system

GET  /gpu            — SSE stream of accelerator status (UI dashboard)
GET  /gpu-queue      — global job queue view
GET  /gpu/resources  — per-device read model with live consumers
GET  /settings       — system + per-user configuration
PUT  /settings       — update settings
GET  /storage        — storage usage summary
GET  /health         — liveness check
GET  /ready          — readiness check
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.core import accelerator, gpu_resources
from server.core.config import (
    LOGS_DIR,
    REDIS_URL,
    STORAGE_ROOT,
    SYSTEM_CONFIG_PATH,
    VERSION,
)
from server.core.dataset_paths import corpus_root
from server.core.logging import sys_log

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(STORAGE_ROOT / "chroma")))

router = APIRouter(prefix="/api/system", tags=["System"])

LOG_DIR = LOGS_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_redis():
    import redis

    return redis.from_url(REDIS_URL, decode_responses=True)


def _load_system_config() -> dict:
    if SYSTEM_CONFIG_PATH.exists():
        try:
            return json.loads(SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "setup_complete": False,
        "agent_autonomy": "guided",
    }


def _save_system_config(config: dict):
    SYSTEM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


USER_SCOPED_FIELDS = {
    "agent_autonomy",
    "pipeline_goal",
}


def _user_config_path(user_id: str) -> Path:
    return STORAGE_ROOT / "outputs" / user_id / "user_config.json"


def _load_user_config(user_id: str) -> dict:
    path = _user_config_path(user_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_config(user_id: str, data: dict) -> None:
    path = _user_config_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Accelerator status
# ---------------------------------------------------------------------------


@router.get("/gpu")
async def gpu_status_stream(request: Request):
    """SSE stream of accelerator status."""

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            data = json.dumps(_query_gpu(), ensure_ascii=False)
            yield f"data: {data}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _query_gpu() -> dict:
    """Accelerator status in the shape the UI dashboard expects."""
    result = accelerator.probe()
    gpus = result.get("gpus") or []
    return {
        **result,
        "gpu_count": len(gpus),
        "timestamp": datetime.now().isoformat(),
    }


# Device selection and live-state scanning live in the fastapi-free core module
# so unit tests can exercise them without importing the router package.
_select_gpu = accelerator.select_gpu
_scan_active_consumers = gpu_resources.scan_active_consumers


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/settings")
async def get_settings(request: Request):
    """System settings merged with the caller's per-user settings."""
    config = _load_system_config()
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        user_config = _load_user_config(user_id)
        config.update(user_config)
        # An existing user_config.json without the flag means onboarding predates it.
        if "onboarding_complete" not in user_config:
            config["onboarding_complete"] = _user_config_path(user_id).exists()
    else:
        config["onboarding_complete"] = False
    return config


class SettingsUpdate(BaseModel):
    agent_autonomy: Optional[str] = None
    openai_api_key: Optional[str] = None
    pipeline_goal: Optional[str] = None
    system_prompt: Optional[str] = None


ADMIN_ONLY_FIELDS = {"openai_api_key"}


@router.put("/settings")
async def update_settings(req: SettingsUpdate, request: Request):
    """Update system settings."""
    user_id = getattr(request.state, "user_id", "default")
    user_role = getattr(request.state, "user_role", "user")

    config = _load_system_config()
    updates = req.model_dump(exclude_none=True)

    forbidden = ADMIN_ONLY_FIELDS & set(updates.keys())
    if forbidden and user_role != "admin":
        raise HTTPException(
            status_code=403,
            detail=f"Admin role required to change: {', '.join(sorted(forbidden))}",
        )

    # Secrets go to the environment, never into plain JSON.
    if "openai_api_key" in updates:
        os.environ["OPENAI_API_KEY"] = updates.pop("openai_api_key")

    user_updates = {k: v for k, v in updates.items() if k in USER_SCOPED_FIELDS}
    global_updates = {k: v for k, v in updates.items() if k not in USER_SCOPED_FIELDS}

    if user_updates and user_id:
        current_user_config = _load_user_config(user_id)
        current_user_config.update(user_updates)
        _save_user_config(user_id, current_user_config)

    config.update(global_updates)
    _save_system_config(config)

    sys_log(f"[System] Settings updated by user={user_id}: {list(updates.keys())}")
    return {"message": "Settings updated", "updated_keys": list(updates.keys())}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health_check():
    """Liveness check (no auth required)."""
    return {
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check (no auth required)."""
    checks = {}

    try:
        _get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    checks["storage"] = "ok" if STORAGE_ROOT.exists() else "missing"

    gpu = _query_gpu()
    checks["accelerator"] = gpu.get("backend", "cpu") if gpu.get("available") else gpu.get("error", "unavailable")

    # A CPU-only host is usable, just slow — it must not fail readiness.
    all_ok = checks["redis"] == "ok" and checks["storage"] == "ok"
    return {
        "ready": all_ok,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

RUNNING_STATUSES = {"STARTED", "PROGRESS", "RUNNING"}

_EMPTY_QUEUE = {
    "total": 0,
    "train": 0,
    "busy": False,
    "mode": "legacy",
    "running": [],
    "waiting": [],
    "gpus": [],
}


@router.get("/gpu-queue")
async def get_gpu_queue_status(request: Request):
    """Global queue state (read-only, not filtered by user)."""
    records: list = []
    try:
        r = _get_redis()
        train_count = 0

        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="train:state:*", count=200)
            for key in keys:
                try:
                    raw = r.get(key)
                    if not raw:
                        continue
                    meta = json.loads(raw)
                    job_id = meta.get("job_id") or str(key).rsplit(":", 1)[-1]
                    status = _live_status(job_id, (meta.get("status") or "").upper())
                    if status in RUNNING_STATUSES:
                        train_count += 1
                    records.append(
                        {
                            "kind": "training",
                            "status": status,
                            "job_id": job_id,
                            "user_id": meta.get("user_id"),
                            "started_at": meta.get("started_at") or meta.get("created_at"),
                            "admission_reason": meta.get("admission_reason"),
                            "live_message": _live_message(job_id),
                        }
                    )
                except Exception:
                    pass
            if cursor == 0:
                break

        counts = {
            "total": train_count,
            "train": train_count,
            "busy": train_count > 0,
        }

        try:
            payload = _build_queue_detail(counts, records)
        except Exception as e:  # noqa: BLE001 — a detail failure must not kill the badge
            sys_log(f"[gpu-queue] detail build failed: {e}", level="warning")
            payload = {**counts, "mode": "legacy", "running": [], "waiting": [], "gpus": []}
        return payload
    except Exception as e:
        sys_log(f"[gpu-queue] error: {e}", level="warning")
        return dict(_EMPTY_QUEUE)


def _hold_message() -> str:
    """Reason the scheduler is holding all jobs. Empty means no banner."""
    try:
        from server.core.gpu_admission import get_hold, hold_message

        return hold_message(get_hold())
    except Exception:  # noqa: BLE001
        return ""


def _live_message(job_id: str) -> str:
    """Progress line such as ``training · step 127/747``. Empty on lookup failure."""
    try:
        from server.core.state import get_job

        return str((get_job(job_id) or {}).get("message") or "")
    except Exception:  # noqa: BLE001
        return ""


def _live_status(job_id: str, meta_status: str) -> str:
    """Live status from ``job:{id}``, which the worker updates.

    ``train:state:*`` is submission metadata written once by ``POST /start``
    and never refreshed. On lookup failure keep the submitted status so the
    view degrades instead of emptying.
    """
    try:
        from server.core.state import get_job

        live = (get_job(job_id) or {}).get("status")
        if live:
            return str(live).upper()
    except Exception:  # noqa: BLE001
        pass
    return meta_status


def _build_queue_detail(counts: dict, records: list) -> dict:
    import time

    from server.core.gpu_allocator import get_mode, shared_leases
    from server.core.queue_view import build_gpu_rows, build_queue_view, build_rows

    lease_by_job: dict = {}
    try:
        for lease in shared_leases().values():
            meta = getattr(lease, "meta", None) or {}
            job_id = str(meta.get("job_id") or "")
            if job_id:
                lease_by_job[job_id] = {"indices": getattr(lease, "indices", ()) or ()}
    except Exception:  # noqa: BLE001 — without the ledger only the device column is missing
        lease_by_job = {}

    running, waiting = build_rows(records, time.time(), lease_by_job)

    return build_queue_view(
        counts=counts,
        running=running,
        waiting=waiting,
        gpus=build_gpu_rows(accelerator.probe().get("gpus") or []),
        mode=get_mode(),
        hold_message=_hold_message(),
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@router.get("/storage")
async def storage_summary(request: Request):
    """Per-category storage usage plus disk usage."""
    dirs_to_scan = {
        "models": STORAGE_ROOT / "models",
        "raw_corpus": STORAGE_ROOT / "raw_corpus",
        "corpus": corpus_root(),
        "outputs": STORAGE_ROOT / "outputs",
        "checkpoints": STORAGE_ROOT / "checkpoints",
        "chroma": CHROMA_DIR,
        "logs": LOG_DIR,
    }

    summary = {}
    total_size = 0
    for name, path in dirs_to_scan.items():
        size = 0
        item_count = 0
        if path.exists():
            files = [f for f in path.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            item_count = len(files)
        summary[name] = {
            "path": str(path),
            "size_mb": round(size / (1024 * 1024), 2),
            "item_count": item_count,
        }
        total_size += size

    try:
        usage = shutil.disk_usage(str(STORAGE_ROOT))
        disk_info = {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "usage_pct": round(usage.used / usage.total * 100, 1),
        }
    except Exception:
        disk_info = {}

    return {
        "categories": summary,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "disk": disk_info,
        "storage_root": str(STORAGE_ROOT),
    }


# ---------------------------------------------------------------------------
# Accelerator read model
# ---------------------------------------------------------------------------


@router.get("/gpu/resources")
async def get_gpu_resources(request: Request):
    """Single read model of accelerator usage: device metrics + live consumers."""
    indices = accelerator.device_indices()
    topology = {
        "device_count": len(indices),
        "backend": accelerator.backend(),
        "device_indices": indices,
    }

    probe = _query_gpu()
    metrics_by_index: dict[int, Optional[dict]] = {idx: _select_gpu(probe, idx) for idx in indices}

    consumers: list[dict] = []
    try:
        consumers.extend(_scan_active_consumers(_get_redis()))
    except Exception as e:
        sys_log(f"[gpu/resources] redis scan error: {e}", level="warning")

    payload = gpu_resources.build_gpu_resources(
        topology,
        metrics_by_index,
        consumers,
        generated_at=datetime.now().isoformat(),
    )
    payload["train_capacity"] = _train_capacity(metrics_by_index)
    return payload


def _train_capacity(metrics_by_index: "dict[int, Optional[dict]]") -> "dict[str, Any]":
    """Largest trainable model size per method, from free device memory."""
    try:
        from server.core.train_capacity import build_train_capacity

        free_gb = [
            max(0.0, float(m.get("memory_free_mb") or 0.0) / 1024.0) for m in metrics_by_index.values() if m
        ]
        return build_train_capacity(free_gb)
    except Exception as e:  # noqa: BLE001 — observability must not break the view
        sys_log(f"[gpu/resources] train_capacity build failed: {e}", level="warning")
        return {"gpu_free_gb": [], "options": []}
