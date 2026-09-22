"""Model tasks — HuggingFace downloads on the celery ``default`` queue.

A repo id is flattened for the filesystem: ``Qwen/Qwen2.5-3B-Instruct`` is
stored as ``Qwen--Qwen2.5-3B-Instruct``. ``pipelines.dialog.locate_model_directory``
resolves either spelling back to that folder.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from celery import shared_task

from server.core.config import MODELS_DIR
from server.core.logging import sys_log
from server.core.state import update_job

#: Redis key marking a download as in flight, so two requests for the same
#: model cannot write the same folder at once. The TTL covers a worker that
#: dies without clearing it.
LOCK_PREFIX = "model:download:"
LOCK_TTL_SECONDS = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "3600")) + 300

#: Weights and the files needed to load them. Everything else in a repo
#: (demo notebooks, ONNX exports, GGUF quantisations) is skipped, which keeps
#: a download to the weights the trainer actually reads.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.jinja",
]


def flattened_name(model_id: str) -> str:
    """Repo id to its on-disk folder name."""
    return model_id.replace("/", "--")


@shared_task(
    name="celery_app.tasks.model_tasks.download_hf_task",
    bind=True,
    soft_time_limit=int(os.getenv("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "3600")),
    time_limit=int(os.getenv("MODEL_DOWNLOAD_TIMEOUT_SECONDS", "3600")) + 300,
    max_retries=0,
)
def download_hf_task(self, **params) -> Dict[str, Any]:
    """Download a HuggingFace model into the models directory.

    ``HF_TOKEN`` is read from the environment and is required for gated repos.
    A partial download is removed, so a failure never leaves a folder that
    looks like a usable model.
    """
    job_id = params.get("job_id") or self.request.id or "unknown"
    model_id = str(params.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required")

    dest = MODELS_DIR / flattened_name(model_id)
    token = os.getenv("HF_TOKEN") or None

    sys_log(f"[download_hf] {model_id} -> {dest}" + ("" if token else " (no HF_TOKEN; public repos only)"))
    update_job(job_id, status="STARTED", progress=0, message=f"Downloading {model_id}")

    try:
        return _download(job_id, model_id, dest, token)
    finally:
        release_download(model_id)


def _download(job_id: str, model_id: str, dest: Path, token) -> Dict[str, Any]:
    """The download itself. Existing files are reused, so a retry is cheap."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(dest),
            allow_patterns=ALLOW_PATTERNS,
            token=token,
        )
    except Exception as e:
        # A half-written folder would be listed as an installed model and then
        # fail at load time, which is harder to diagnose than a missing one.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        error = f"{type(e).__name__}: {e}"
        if not token and "gated" in str(e).lower():
            error += " — this repo is gated; set HF_TOKEN in .env"
        update_job(job_id, status="FAILURE", error=error)
        sys_log(f"[download_hf] {model_id} failed: {error}", level="ERROR")
        raise

    if not (dest / "config.json").exists():
        error = f"{model_id} downloaded but has no config.json — not a loadable model"
        shutil.rmtree(dest, ignore_errors=True)
        update_job(job_id, status="FAILURE", error=error)
        raise RuntimeError(error)

    size_mb = round(sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 * 1024), 1)
    result = {"model_id": model_id, "path": str(dest), "name": dest.name, "size_mb": size_mb}
    update_job(job_id, status="SUCCESS", progress=100, message=f"Downloaded {model_id}", result=result)
    sys_log(f"[download_hf] {model_id} done ({size_mb} MB)")
    return result


def claim_download(model_id: str) -> bool:
    """Take the download lock for this model. False when one is already running."""
    try:
        from server.core.state import _get_redis

        return bool(_get_redis().set(f"{LOCK_PREFIX}{flattened_name(model_id)}", "1",
                                     ex=LOCK_TTL_SECONDS, nx=True))
    except Exception:  # noqa: BLE001 — a Redis failure must not block a download
        return True


def release_download(model_id: str) -> None:
    """Release the lock. Idempotent."""
    try:
        from server.core.state import _get_redis

        _get_redis().delete(f"{LOCK_PREFIX}{flattened_name(model_id)}")
    except Exception:  # noqa: BLE001
        pass


