"""Models router — /api/models

GET    /base                      — list locally installed base models
GET    /trained                   — list fine-tuned models
POST   /download/hf               — start a HuggingFace download (celery)
GET    /download/{job_id}/status  — poll that download
DELETE /base/{name}               — delete a base model
DELETE /trained/{name}            — delete a fine-tuned model
"""

import json
import re
import shutil
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server.core.config import STORAGE_ROOT
from server.core.logging import sys_log
from server.core.ttl_cache import TTLCache

router = APIRouter(prefix="/api/models", tags=["Models"])

MODELS_DIR = STORAGE_ROOT / "models"
OUTPUTS_DIR = STORAGE_ROOT / "outputs" / "completed"

_base_models_cache = TTLCache(10.0)


def _models_dir_signature() -> object:
    try:
        return MODELS_DIR.stat().st_mtime
    except OSError:
        return None


CHECKPOINTS_DIR = STORAGE_ROOT / "checkpoints"

#: A HuggingFace repo id: ``org/model``. Anything else is rejected before it
#: reaches the filesystem, so a crafted id cannot escape the models directory.
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


# ---------------------------------------------------------------------------
# GET /base
# ---------------------------------------------------------------------------


@router.get("/base")
async def list_base_models(request: Request):
    """List base HuggingFace models stored locally."""
    return _base_models_cache.get_or_compute(
        "base",
        _build_base_models,
        signature=_models_dir_signature,
    )


def _build_base_models() -> dict:
    """Build the base model list by scanning disk. Called only on a cache miss."""
    models = []
    if MODELS_DIR.exists():
        for entry in sorted(MODELS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            config_path = entry / "config.json"
            if not config_path.exists():
                continue

            info = {"name": entry.name, "path": str(entry)}
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                info["model_type"] = config.get("model_type", "unknown")
                info["architectures"] = config.get("architectures", [])
            except Exception:
                pass

            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            info["size_gb"] = round(size / (1024**3), 2)
            info["created_at"] = datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
            models.append(info)

    return {"models": models, "total": len(models)}


# ---------------------------------------------------------------------------
# GET /trained
# ---------------------------------------------------------------------------


@router.delete("/base/{name}")
async def delete_base_model(name: str, request: Request):
    """Delete a base model."""
    user_id = getattr(request.state, "user_id", "default")
    target = MODELS_DIR / name

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Base model '{name}' not found")

    shutil.rmtree(target)
    sys_log(f"[Models] Base model deleted: {name} (user={user_id})")
    return {"message": f"Base model '{name}' deleted"}


# ---------------------------------------------------------------------------
# DELETE /trained/{name}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# POST /download/hf, GET /download/{job_id}/status
# ---------------------------------------------------------------------------


class HfDownloadRequest(BaseModel):
    model_id: str


@router.post("/download/hf")
async def download_hf_model(req: HfDownloadRequest, request: Request):
    """Queue a HuggingFace model download.

    Gated repos need ``HF_TOKEN`` in the environment; the worker reads it.
    Returns the job id to poll, or 409 when the same model is already being
    downloaded.
    """
    user_id = getattr(request.state, "user_id", "default")
    model_id = (req.model_id or "").strip()

    if not _REPO_ID.match(model_id):
        raise HTTPException(
            status_code=400,
            detail=f"'{model_id}' is not a HuggingFace repo id. Use the org/model form, e.g. Qwen/Qwen2.5-0.5B-Instruct.",
        )

    from celery_app.tasks.model_tasks import claim_download, download_hf_task, flattened_name
    from server.core.state import create_job

    if not claim_download(model_id):
        raise HTTPException(status_code=409, detail=f"{model_id} is already being downloaded.")

    job_id = f"dl-{flattened_name(model_id)}"
    try:
        create_job("model_download", user_id, params={"model_id": model_id}, job_id=job_id)
        download_hf_task.si(job_id=job_id, model_id=model_id, user_id=user_id).apply_async(
            queue="default", task_id=job_id
        )
    except Exception as e:
        # Dispatch failed, so nothing will ever release the lock.
        from celery_app.tasks.model_tasks import release_download

        release_download(model_id)
        raise HTTPException(status_code=500, detail=f"Could not queue the download: {e}") from e

    sys_log(f"[Models] Queued HF download: {model_id} (job={job_id}, user={user_id})")
    return {"job_id": job_id, "model_id": model_id, "status": "queued"}


@router.get("/download/{job_id}/status")
async def download_status(job_id: str, request: Request):
    """Progress of a queued download."""
    from server.core.state import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No download job '{job_id}'")
    return {
        "job_id": job_id,
        "status": job.get("status", "PENDING"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "error": job.get("error", ""),
        "result": job.get("result"),
    }
