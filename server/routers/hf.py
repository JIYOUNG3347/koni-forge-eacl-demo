"""HuggingFace local-model router — /api/hf

GET  /models          — list base and fine-tuned models
POST /load            — load a model into the app process
POST /unload          — release it
GET  /status          — what is loaded
POST /probe           — run KBD probes on the worker
GET  /probe/{job_id}  — poll that run

KBD, tuning and chat all run against the HuggingFace models under the models
directory.
"""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server.core.error_messages import error as _err
from server.core.error_messages import lang_from_request as _errlang
from server.core.logging import sys_log

router = APIRouter(prefix="/api/hf", tags=["HF"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HfLoadRequest(BaseModel):
    model_name: str


# ---------------------------------------------------------------------------
# GET /models — list available HF models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_hf_models(request: Request):
    """Scan /storage/models and /storage/outputs/{user_id}/completed for HF models.

    Returns separate arrays so callers (KBD probe picker, tuning recommender)
    can filter. Base models = config.json present, no adapter/training marker.
    Fine-tuned models are scanned from user-specific completed output directories
    (/storage/outputs/{user_id}/completed/{job_id}) as written by train_sft_task.
    """
    from pathlib import Path

    from pipelines.dialog import BASE_MODELS_ROOT, OUTPUTS_ROOT, TRAINED_OUTPUTS_ROOT

    base_models: List[dict] = []
    trained_outputs: List[dict] = []
    seen_names: set = set()

    base_root = Path(BASE_MODELS_ROOT)
    if base_root.is_dir():
        for entry in sorted(base_root.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "config.json").exists():
                continue
            # Exclude LoRA adapter folders and training outputs
            if (entry / "adapter_config.json").exists():
                continue
            if (entry / "training_params.json").exists():
                continue
            base_models.append({"name": entry.name})

    def _add_trained(job: Path) -> None:
        if job.name in seen_names:
            return
        has_cfg = (job / "config.json").exists() or (job / "adapter_config.json").exists()
        if has_cfg:
            seen_names.add(job.name)
            trained_outputs.append({"name": job.name, "kind": "finetuned"})

    # Legacy: /storage/outputs/completed/{job_id}
    legacy_root = Path(TRAINED_OUTPUTS_ROOT)
    if legacy_root.is_dir():
        for job in sorted(legacy_root.iterdir()):
            if job.is_dir():
                _add_trained(job)

    # Primary: /storage/outputs/{user_id}/completed/{job_id}
    outputs_root = Path(OUTPUTS_ROOT)
    if outputs_root.is_dir():
        for user_dir in sorted(outputs_root.iterdir()):
            if not user_dir.is_dir():
                continue
            completed_root = user_dir / "completed"
            if completed_root.is_dir():
                for job in sorted(completed_root.iterdir(), reverse=True):
                    if job.is_dir():
                        _add_trained(job)

    return {
        "base_models": base_models,
        "trained_outputs": trained_outputs,
        "count": len(base_models) + len(trained_outputs),
    }


@router.post("/load")
async def hf_load(req: HfLoadRequest, request: Request):
    """Load an HF base (or fine-tuned) model into GPU memory.
    Blocks until the weights are resident — the first request for a large
    model can take tens of seconds. Subsequent calls for the same model are
    cache hits.

    """
    user_id = getattr(request.state, "user_id", "default")
    try:
        from pipelines.dialog import activate_model, inference_state

        activate_model(req.model_name)
        sys_log(f"[HF] Loaded: {req.model_name} (user={user_id})")
        return {
            "status": "loaded",
            "model": req.model_name,
            "loaded_path": inference_state.get("model_name"),
        }
    except ValueError as e:
        # Model not found in any of the search roots
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        sys_log(f"[HF] Load failed: {req.model_name} — {e}", level="ERROR")
        raise HTTPException(
            status_code=500,
            detail=_err("model_load_failed", e, lang=_errlang(request, user_id)),
        )


# ---------------------------------------------------------------------------
# POST /unload — release GPU memory
# ---------------------------------------------------------------------------


@router.post("/unload")
async def hf_unload(request: Request):
    """Release the model loaded in the app process."""
    user_id = getattr(request.state, "user_id", "default")
    try:
        from pipelines.dialog import release_loaded_model

        release_loaded_model()
        sys_log(f"[HF] Unloaded (user={user_id})")
        return {"status": "unloaded"}
    except Exception as e:
        sys_log(f"[HF] Unload failed: {e}", level="WARNING")
        # Unload failures are non-fatal — return ok so callers don't retry
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# GET /status — which model is currently loaded
# ---------------------------------------------------------------------------


@router.get("/status")
async def hf_status(request: Request):
    """Which model is loaded in the app process."""
    try:
        from pipelines.dialog import inference_state

        return {
            "loaded": inference_state.get("model") is not None,
            "model_path": inference_state.get("model_name"),
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


# ---------------------------------------------------------------------------
# POST /probe, GET /probe/{job_id}
# ---------------------------------------------------------------------------


class ProbeRequest(BaseModel):
    model_name: str
    probes: List[dict]


@router.post("/probe")
async def start_probe(req: ProbeRequest, request: Request):
    """Queue a KBD probe run on the worker.

    Probing loads a model, so it runs on the GPU queue rather than in the app
    process, where it would block the event loop and compete with training.
    """
    user_id = getattr(request.state, "user_id", "default")
    if not req.model_name.strip():
        raise HTTPException(status_code=400, detail="model_name is required")
    if not req.probes:
        raise HTTPException(status_code=400, detail="probes is required")

    from celery_app.tasks.kbd_tasks import kbd_probe_task
    from server.core.gpu_queue import submit_gpu_task
    from server.core.state import create_job, update_job

    job_id = f"kbd-{uuid.uuid4().hex[:12]}"
    create_job("kbd_probe", user_id, params={"model_name": req.model_name, "probes": len(req.probes)}, job_id=job_id)
    try:
        submit_gpu_task(
            kbd_probe_task,
            consumer_kind="kbd_probe",
            task_id=job_id,
            job_id=job_id,
            user_id=user_id,
            model_name=req.model_name,
            probes=req.probes,
        )
    except Exception as e:
        # Otherwise the caller polls a job that no worker will ever pick up.
        update_job(job_id, status="FAILURE", error=f"dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not queue the probe: {e}") from e

    sys_log(f"[HF] Queued KBD probe: {req.model_name}, {len(req.probes)} probes (job={job_id}, user={user_id})")
    return {"job_id": job_id, "status": "queued", "total": len(req.probes)}


@router.get("/probe/{job_id}")
async def probe_status(job_id: str, request: Request):
    """Progress of a queued probe run."""
    from server.core.state import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No probe job '{job_id}'")
    return {
        "job_id": job_id,
        "status": job.get("status", "PENDING"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "error": job.get("error", ""),
        "result": job.get("result"),
    }
