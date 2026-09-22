"""Training router — /api/train

POST /start               — dispatch SFT / LoRA training via submit_gpu_task
GET  /status/{job_id}      — query Celery AsyncResult + Redis state
POST /stop/{job_id}        — revoke Celery task
GET  /checkpoints          — list user's checkpoints
GET  /results/{job_id}     — get training results
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server.core.celery_probe import active_task_ids
from server.core.config import REDIS_URL, STORAGE_ROOT
from server.core.dataset_paths import dataset_roots
from server.core.error_messages import error as _err
from server.core.error_messages import lang_from_request as _errlang
from server.core.lang import resolve_lang
from server.core.logging import sys_log
from server.core.pipeline_config import load_pipeline_config
from server.core.progress_messages import progress
from server.core.ttl_cache import TTLCache

router = APIRouter(prefix="/api/train", tags=["Train"])

CHECKPOINTS_DIR = STORAGE_ROOT / "checkpoints"
OUTPUTS_DIR = STORAGE_ROOT / "outputs"

_checkpoints_cache = TTLCache(10.0)


def _checkpoints_signature(user_id: str) -> tuple:
    """Cheap invalidation signature: the mtimes of the relevant roots, which change with a new checkpoint."""
    sig = []
    for p in (OUTPUTS_DIR / user_id / "completed", CHECKPOINTS_DIR, CHECKPOINTS_DIR / user_id):
        try:
            sig.append((str(p), p.stat().st_mtime))
        except OSError:
            sig.append((str(p), None))
    return tuple(sig)


# ---------------------------------------------------------------------------
# Redis helper
# ---------------------------------------------------------------------------


def _get_redis():
    import redis

    return redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TrainStartRequest(BaseModel):
    model_name: str
    model_path: Optional[str] = None
    dataset_name: str
    method: str = "sft"  # sft | lora
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_grad_norm: float = 1.0
    system_prompt: Optional[str] = None
    # User-supplied display/identifier. When set, used verbatim as the Celery
    # task id (and therefore as the disk folder name under /storage/outputs/…).
    job_name: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /start
# ---------------------------------------------------------------------------


@router.post("/start")
async def start_training(req: TrainStartRequest, request: Request):
    """Dispatch SFT / LoRA training via submit_gpu_task."""
    user_id = getattr(request.state, "user_id", "default")

    from server.core.train_model_source import resolve_train_model_source

    sys_log(f"[Train] start: model_name={req.model_name!r} dataset_name={req.dataset_name!r}")
    try:
        resolved_model_path = resolve_train_model_source(req.model_name, req.model_path, STORAGE_ROOT, user_id)
    except ValueError as e:
        sys_log(f"[Train] 404 model resolve: {e}", level="WARNING")
        raise HTTPException(status_code=404, detail=str(e))

    # Validate dataset exists
    dataset_found = False
    for base in dataset_roots():
        if (base / req.dataset_name).exists():
            dataset_found = True
            break
    if not dataset_found:
        sys_log(f"[Train] 404 dataset not found: {req.dataset_name!r}", level="WARNING")
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_name}' not found")

    from server.core.gpu_queue import submit_gpu_task

    # Honour user-supplied job_name when present (sanitise first). Falls back
    # to the auto-generated id for unnamed runs.
    def _sanitize_job_name(name: str) -> str:
        # Allow letters / digits / dash / underscore / dot; collapse rest to "-"
        import re

        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
        return cleaned.strip("-") or ""

    if req.job_name and _sanitize_job_name(req.job_name):
        job_id = _sanitize_job_name(req.job_name)
        # Prevent overwriting an existing named job (auto-generated ids are
        # always unique so this check only matters for user-supplied names).
        if _get_redis().get(f"train:state:{job_id}"):
            raise HTTPException(
                status_code=409,
                detail=_err("job_name_exists", job_id, lang=_errlang(request, user_id)),
            )
    else:
        job_id = f"train-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex()}"

    from celery_app.tasks.train_tasks import train_sft_task

    task_func = train_sft_task
    # Per-job output directory under outputs/{user}/completed/{job_id} — the same
    # place list_checkpoints surfaces as completed models.
    output_dir = str(STORAGE_ROOT / "outputs" / user_id / "completed" / job_id)

    params = {
        "job_id": job_id,
        "user_id": user_id,
        "model_name": req.model_name,
        "model_name_or_path": resolved_model_path,
        "dataset_name": req.dataset_name,
        "method": req.method,
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "batch_size": req.batch_size,
        "max_seq_length": req.max_seq_length,
        "lora_r": req.lora_r,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": req.lora_dropout,
        "max_grad_norm": req.max_grad_norm,
        "system_prompt": req.system_prompt,
        "output_dir": output_dir,
        "exec_lang": resolve_lang(config=load_pipeline_config(user_id)),
    }

    # submit_gpu_task returns the AsyncResult for the chain so we can revoke
    # it later. The chain has nested task ids (unload → train → reload) that
    # we walk to force-kill the GPU-bound train step on user stop.
    chain_result = submit_gpu_task(task_func, **params)
    chain_task_ids: list[str] = []
    try:
        cur = chain_result
        while cur is not None:
            if getattr(cur, "id", None):
                chain_task_ids.append(cur.id)
            cur = getattr(cur, "parent", None)
    except Exception:
        pass

    # Store initial state in Redis — includes the FULL hyperparameter snapshot
    # so the params modal can render epochs / batch / LR / max_seq / LoRA
    # even while the job is still running (training_params.json only gets
    # written after `trainer.train()` returns).
    r = _get_redis()
    state = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "queued",
        "method": req.method,
        "model_name": req.model_name,
        "dataset_name": req.dataset_name,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "max_seq_length": req.max_seq_length,
        "lora_r": req.lora_r,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": req.lora_dropout,
        "max_grad_norm": req.max_grad_norm,
        "output_dir": output_dir,
        "created_at": datetime.now().isoformat(),
        # Needed for /stop — our user-facing job_id is NOT the Celery task id,
        # so without these we can't revoke the running task.
        "chain_task_ids": chain_task_ids,
    }
    r.set(f"train:state:{job_id}", json.dumps(state), ex=86400 * 7)

    sys_log(f"[Train] Dispatched {req.method} training: {job_id} (user={user_id})")
    return {"job_id": job_id, "status": "queued", "method": req.method}


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------


@router.get("/status/{job_id}")
async def get_status(job_id: str, request: Request):
    """Query training progress.

    State is split across two Redis keys — merge both:
      * `train:state:{job_id}`  — initial meta (method, model, dataset, rounds)
         written once by `POST /start`.
      * `job:{job_id}` — live per-step updates (`status`, `progress`,
         `message`, `result`, `error`) written by `_run_sft_training` and
         the worker via `update_job()`.

    `submit_gpu_task` dispatches a Celery *chain* whose tasks have opaque
    UUIDs, so `AsyncResult(job_id)` with our user-facing job id always
    returns PENDING — that's why the original implementation was stuck at
    "queued" even after training finished. We use the authoritative live
    state from `job:{job_id}` instead.
    """
    from server.core.state import get_job

    r = _get_redis()
    state_raw = r.get(f"train:state:{job_id}")
    redis_meta = json.loads(state_raw) if state_raw else {}

    live = get_job(job_id) or {}

    # Decide final status — prefer live status (uppercased by Celery/update_job),
    # fall back to the initial meta.
    raw_status = live.get("status") or redis_meta.get("status") or "queued"
    s_upper = str(raw_status).upper()
    if s_upper == "SUCCESS":
        status = "completed"
    elif s_upper == "FAILURE":
        status = "failed"
    elif s_upper in ("REVOKED", "STOPPED"):
        status = "stopped"
    elif s_upper in ("STARTED", "PROGRESS", "RUNNING"):
        status = "running"
    elif s_upper in ("PENDING", "QUEUED"):
        status = "queued"
    else:
        status = str(raw_status).lower()

    # Surface the full hyperparameter snapshot (stored at /start time) so the
    # frontend's params modal can render them for both in-flight AND completed
    # jobs without depending on training_params.json being on disk yet.
    return {
        "job_id": job_id,
        "status": status,
        "celery_status": s_upper,
        "progress": live.get("progress", redis_meta.get("progress", 0)),
        "message": live.get("message") or redis_meta.get("message") or "",
        "current_epoch": live.get("current_epoch"),
        "current_step": live.get("current_step"),
        "loss": live.get("loss"),
        "result": live.get("result"),
        "error": live.get("error") or "",
        "method": redis_meta.get("method"),
        "model_name": redis_meta.get("model_name"),
        "dataset_name": redis_meta.get("dataset_name"),
        "epochs": redis_meta.get("epochs"),
        "batch_size": redis_meta.get("batch_size"),
        "learning_rate": redis_meta.get("learning_rate"),
        "max_seq_length": redis_meta.get("max_seq_length"),
        "lora_r": redis_meta.get("lora_r"),
        "lora_alpha": redis_meta.get("lora_alpha"),
        "lora_dropout": redis_meta.get("lora_dropout"),
        "output_dir": redis_meta.get("output_dir"),
        "created_at": redis_meta.get("created_at") or live.get("created_at"),
    }


# ---------------------------------------------------------------------------
# POST /stop/{job_id}
# ---------------------------------------------------------------------------


@router.post("/stop/{job_id}")
async def stop_training(job_id: str, request: Request):
    """Revoke a running Celery training task.

    The user-facing `job_id` is NOT the Celery task id — `submit_gpu_task`
    builds a chain (unload → train → reload), each with its own UUID. We
    stored those UUIDs in `train:state:{job_id}` when starting, so look them
    up and revoke each with `terminate=True` to SIGTERM the GPU worker.
    """
    user_id = getattr(request.state, "user_id", "default")
    r = _get_redis()

    state_raw = r.get(f"train:state:{job_id}")
    if not state_raw:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    state = json.loads(state_raw)
    chain_task_ids: list[str] = state.get("chain_task_ids", []) or []

    from celery.result import AsyncResult

    revoked: list[str] = []
    for tid in chain_task_ids:
        try:
            AsyncResult(tid).revoke(terminate=True, signal="SIGTERM")
            revoked.append(tid)
        except Exception as e:
            sys_log(f"[Train] revoke({tid}) failed: {e}", level="WARNING")

    # Belt-and-braces: also mark the live job state as REVOKED so the status
    # endpoint stops reporting "running".
    try:
        from server.core.state import update_job

        update_job(
            job_id,
            status="REVOKED",
            message=progress("user_stopped", lang=resolve_lang(config=load_pipeline_config(user_id))),
        )
    except Exception as e:
        sys_log(f"[Train] update_job(REVOKED) failed: {e}", level="WARNING")

    state["status"] = "stopped"
    r.set(f"train:state:{job_id}", json.dumps(state), ex=86400 * 7)

    sys_log(f"[Train] Stop requested for {job_id} — revoked {len(revoked)} chain task(s) (user={user_id})")
    return {
        "job_id": job_id,
        "status": "stopped",
        "revoked_task_ids": revoked,
    }


# ---------------------------------------------------------------------------
# GET /checkpoints
# ---------------------------------------------------------------------------


def _is_model_dir(p: Path) -> bool:
    """A directory is a 'trained model' if it has adapter_config.json (LoRA)
    or config.json (full FT / HF model)."""
    return (p / "adapter_config.json").exists() or (p / "config.json").exists()


def _checkpoint_entry(path: Path, job_id: str, name: str, location: str) -> dict:
    try:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        size = 0
    try:
        created_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        created_at = ""

    # Enrich from Redis state so the UI can show method / dataset etc. on each row without a follow-up /status round-trip per job.
    method = None
    dataset_name = None
    job_status = None
    try:
        r = _get_redis()
        state_raw = r.get(f"train:state:{job_id}")
        if state_raw:
            s = json.loads(state_raw)
            method = s.get("method")
            dataset_name = s.get("dataset_name")
            job_status = s.get("status")  # "stopped" | "failed" | None
    except Exception:
        pass

    return {
        "job_id": job_id,
        "name": name,
        "path": str(path),
        "size_mb": round(size / (1024 * 1024), 2),
        "is_lora": (path / "adapter_config.json").exists(),
        "created_at": created_at,
        "location": location,  # "completed" | "checkpoint"
        "method": method,
        "dataset_name": dataset_name,
        "job_status": job_status,  # "stopped" | "failed" | None (None = normal/unknown)
    }


@router.get("/active")
async def list_active_training(request: Request):
    """List training jobs that are currently running / queued.

    Scans Redis for `train:state:*` / `job:train-*` entries and cross-checks
    against Celery active tasks to avoid showing stale zombie jobs.
    Used by the header QueueStatus badge.
    """
    user_id = getattr(request.state, "user_id", "default")
    r = _get_redis()

    active: list[dict] = []
    try:
        # Collect all candidate job_ids from train:state:* and job:train-*
        cursor = 0
        candidates: set[str] = set()
        while True:
            cursor, keys = r.scan(cursor, match="train:state:*", count=200)
            for k in keys:
                candidates.add(k[len("train:state:") :])
            if cursor == 0:
                break
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="job:train-*", count=200)
            for k in keys:
                candidates.add(k[len("job:") :])
            if cursor == 0:
                break

        celery_active_ids: Optional[set[str]] = await asyncio.to_thread(active_task_ids)

        from server.core.state import get_job

        for job_id in candidates:
            state_raw = r.get(f"train:state:{job_id}")
            meta = json.loads(state_raw) if state_raw else {}
            if meta.get("user_id") != user_id:
                continue
            live = get_job(job_id) or {}
            raw_status = (live.get("status") or meta.get("status") or "").upper()
            if raw_status not in ("STARTED", "PROGRESS", "RUNNING", "PENDING", "QUEUED"):
                continue

            if celery_active_ids is not None:
                chain_ids = meta.get("chain_task_ids") or []
                if chain_ids and not any(tid in celery_active_ids for tid in chain_ids):
                    sys_log(f"[Train] Zombie job (not in Celery active set): {job_id}")
                    continue  # Skip: don't include in active list

            active.append(
                {
                    "job_id": job_id,
                    "status": raw_status,
                    "progress": live.get("progress", 0),
                    "message": live.get("message") or "",
                    "method": meta.get("method") or live.get("method") or "",
                    "model_name": meta.get("model_name") or "",
                    "dataset_name": meta.get("dataset_name") or "",
                    "created_at": meta.get("created_at"),
                }
            )
    except Exception as e:
        sys_log(f"[Train] list_active_training failed: {e}", level="WARNING")

    active.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return {"active": active, "total": len(active)}


@router.get("/checkpoints")
async def list_checkpoints(request: Request):
    """List every trained-model artifact visible to the user.

    Sources (merged + deduped by path):
      1. `/storage/outputs/{user_id}/completed/*` — finalised training outputs
         (what `train_sft_task` writes). These are the shipped models.
      2. `/storage/checkpoints/{job_name}/{checkpoint-N | round_N}` — legacy
         flat layout (manual `train_sft_task`).
      3. `/storage/checkpoints/{user_id}/pipeline_pipe-*/round_N/` — pipeline
         runs nested 3 levels deep.

    Older implementation only scanned #2, so #1 (the real finished models) and
    #3 never appeared in "completed training".
    """
    user_id = getattr(request.state, "user_id", "default")
    return _checkpoints_cache.get_or_compute(
        user_id,
        lambda: _build_checkpoints(user_id),
        signature=lambda: _checkpoints_signature(user_id),
    )


def _build_checkpoints(user_id: str) -> dict:
    """Build the checkpoint list by scanning disk. Called only on a cache miss."""
    checkpoints: list[dict] = []
    seen_paths: set[str] = set()

    def _add(path: Path, job_id: str, name: str, location: str) -> None:
        key = str(path.resolve())
        if key in seen_paths:
            return
        seen_paths.add(key)
        checkpoints.append(_checkpoint_entry(path, job_id, name, location))

    # 1) Completed outputs — /storage/outputs/{user}/completed/{job_name}/
    completed_root = OUTPUTS_DIR / user_id / "completed"
    if completed_root.exists():
        for job_dir in sorted(completed_root.iterdir(), reverse=True):
            if not job_dir.is_dir():
                continue
            if _is_model_dir(job_dir):
                _add(job_dir, job_id=job_dir.name, name="final", location="completed")

    # 2) Checkpoints under /storage/checkpoints/
    if CHECKPOINTS_DIR.exists():
        for top in sorted(CHECKPOINTS_DIR.iterdir(), reverse=True):
            if not top.is_dir():
                continue

            # 2) /storage/checkpoints/{job_name}/{checkpoint-N | round_N}
            for item in sorted(top.iterdir()):
                if not item.is_dir():
                    continue
                if _is_model_dir(item) or item.name.startswith(("checkpoint-", "round_")):
                    _add(item, job_id=top.name, name=item.name, location="checkpoint")

    # Sort newest-first by mtime
    checkpoints.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return {"checkpoints": checkpoints, "total": len(checkpoints)}


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id}
# ---------------------------------------------------------------------------


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str, request: Request, path: Optional[str] = None):
    """Remove on-disk artifacts for a completed training job.

    Path resolution mirrors `list_checkpoints`:
      * If `path` query param is given, delete exactly that path (must live
        under `/storage/outputs/{user}/completed` or `/storage/checkpoints`
        to prevent traversal).
      * Otherwise delete every top-level folder with name == job_id found in:
          - `/storage/outputs/{user}/completed/{job_id}/`
          - `/storage/checkpoints/{job_id}/`
          - `/storage/checkpoints/{user}/{job_id}/` (pipeline run id)
    """
    import shutil

    user_id = getattr(request.state, "user_id", "default")
    completed_root = OUTPUTS_DIR / user_id / "completed"
    user_ckpt_root = CHECKPOINTS_DIR / user_id

    # Allowed roots — anything outside these is refused
    safe_roots = [completed_root.resolve(), CHECKPOINTS_DIR.resolve()]

    def _is_safe(target: Path) -> bool:
        try:
            t = target.resolve()
        except OSError:
            return False
        return any(str(t).startswith(str(root) + os.sep) or t == root for root in safe_roots)

    targets: list[Path] = []

    if path:
        # Explicit path mode — one specific artifact
        target = Path(path)
        if not _is_safe(target):
            raise HTTPException(status_code=400, detail="Path outside allowed roots")
        if target.exists() and target.is_dir():
            targets.append(target)
    else:
        # Job-id mode — collect all top-level folders that match
        for candidate in (
            completed_root / job_id,
            CHECKPOINTS_DIR / job_id,
            user_ckpt_root / job_id,
        ):
            if candidate.exists() and candidate.is_dir() and _is_safe(candidate):
                targets.append(candidate)

    from server.core.job_delete import not_found_detail, should_delete

    _has_state = False
    try:
        _r0 = _get_redis()
        _has_state = bool(_r0.exists(f"train:state:{job_id}") or _r0.exists(f"job:{job_id}"))
    except Exception:  # noqa: BLE001 — on failure fall back to the disk scan
        _has_state = False

    if not should_delete(has_disk_targets=bool(targets), has_state=_has_state, path_mode=bool(path)):
        raise HTTPException(status_code=404, detail=not_found_detail(job_id, bool(path)))

    deleted: list[str] = []
    errors: list[str] = []
    for t in targets:
        try:
            shutil.rmtree(t)
            deleted.append(str(t))
        except Exception as e:
            errors.append(f"{t}: {e}")

    # Clean up Redis state so the job name can be reused
    try:
        _r = _get_redis()
        _r.delete(f"train:state:{job_id}")
        _r.delete(f"job:{job_id}")
    except Exception as e:
        sys_log(f"[Train] Failed to delete Redis keys for '{job_id}': {e}", level="WARNING")

    sys_log(f"[Train] Deleted job '{job_id}' — {len(deleted)} path(s) (user={user_id}, errors={len(errors)})")
    return {
        "job_id": job_id,
        "deleted": deleted,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# GET /results/{job_id}
# ---------------------------------------------------------------------------


@router.get("/results/{job_id}")
async def get_results(job_id: str, request: Request):
    """Get training results (loss curves, metrics).

    Searches multiple layouts produced by our different task paths:
      * `/storage/outputs/{user}/completed/{job_id}/` — single-run SFT/LoRA
      * Legacy: `{OUTPUTS_DIR}/{job_id}/` and `{CHECKPOINTS_DIR}/{job_id}/`
    Uses recursive glob for `training_params.json`, `training_results.json`,
    and `trainer_state.json` so nested checkpoint-K folders are picked up.
    """
    import glob

    user_id = getattr(request.state, "user_id", "default")
    r = _get_redis()

    state_raw = r.get(f"train:state:{job_id}")
    state = json.loads(state_raw) if state_raw else {}

    # Candidate roots that could contain this job's artifacts.
    roots = [
        OUTPUTS_DIR / user_id / "completed" / job_id,  # #1
        CHECKPOINTS_DIR / user_id / job_id,  # #2
        OUTPUTS_DIR / job_id,  # legacy
        CHECKPOINTS_DIR / job_id,  # legacy
    ]

    job_dir: Optional[Path] = None
    for cand in roots:
        if cand.exists():
            job_dir = cand
            break

    disk_results: dict = {}
    training_params: dict = {}
    log_history: list = []

    if job_dir is not None:
        # Recursive file collection — pick newest of each kind.
        def _latest(pattern: str) -> Optional[Path]:
            matches = glob.glob(str(job_dir / pattern), recursive=True)
            if not matches:
                return None
            matches.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
            return Path(matches[0])

        results_file = _latest("**/training_results.json") or _latest("training_results.json")
        params_file = _latest("**/training_params.json") or _latest("training_params.json")
        trainer_state_file = _latest("**/trainer_state.json") or _latest("trainer_state.json")

        if results_file:
            try:
                disk_results = json.loads(results_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if params_file:
            try:
                training_params = json.loads(params_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if trainer_state_file:
            try:
                ts = json.loads(trainer_state_file.read_text(encoding="utf-8"))
                log_history = ts.get("log_history", [])
            except Exception:
                pass

    # Fallback: seed training_params from the Redis snapshot so the UI has
    # SOMETHING to display during the in-flight window (before the worker
    # writes training_params.json to disk).
    if not training_params and state:
        training_params = {
            k: state.get(k)
            for k in (
                "method",
                "model_name",
                "dataset_name",
                "epochs",
                "batch_size",
                "learning_rate",
                "max_seq_length",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
            )
            if state.get(k) is not None
        }

    # Also check Redis job state for result data
    from server.core.state import get_job

    redis_job = get_job(job_id) or {}
    redis_result = redis_job.get("result", {})
    if isinstance(redis_result, str):
        try:
            redis_result = json.loads(redis_result)
        except Exception:
            redis_result = {}

    return {
        "job_id": job_id,
        "status": state.get("status") or redis_job.get("status", "unknown"),
        "training_results": disk_results or training_params or redis_result,
        "log_history": log_history[-100:],
        "training_params": training_params,
    }
