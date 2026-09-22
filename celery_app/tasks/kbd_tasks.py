"""KBD probe task — answers probe questions with a base model, no retrieval."""

import os
from typing import Any, Dict, List

from celery import shared_task

from server.core import accelerator
from server.core.logging import sys_log
from server.core.state import update_job

PROBE_TEMPERATURE = 0.1
PROBE_MAX_TOKENS = 512


@shared_task(
    name="celery_app.tasks.kbd_tasks.kbd_probe_task",
    bind=True,
    soft_time_limit=int(os.getenv("KBD_PROBE_TIMEOUT_SECONDS", "3600")),
    time_limit=int(os.getenv("KBD_PROBE_TIMEOUT_SECONDS", "3600")) + 300,
    max_retries=0,
)
def kbd_probe_task(self, **params) -> Dict[str, Any]:
    """Ask the model every probe question closed-book and record its answers.

    Returns ``{"results": [{question, ground_truth, category, model_answer}],
    "total": int, "errors": int}``, which is what the judge then scores.
    """
    job_id = params.get("job_id") or self.request.id or "unknown"
    model_name = str(params.get("model_name") or "").strip()
    probes: List[Dict[str, Any]] = params.get("probes") or []

    if not model_name:
        raise ValueError("model_name is required")
    if not probes:
        raise ValueError("probes is required")

    from server.core.gpu_admission import gate_or_retry

    gate_or_retry(self, "kbd_probe", job_id, user_id=str(params.get("user_id") or ""))

    from pipelines.dialog import activate_model, generate_response, release_loaded_model

    sys_log(f"[kbd_probe] {job_id}: {len(probes)} probes on {model_name}")
    update_job(job_id, status="STARTED", progress=0, message=f"Loading {model_name}")

    try:
        activate_model(model_name)

        results: List[Dict[str, Any]] = []
        errors = 0
        for i, probe in enumerate(probes):
            question = str(probe.get("question") or "").strip()
            entry = {
                "question": question,
                "ground_truth": probe.get("ground_truth", ""),
                "category": probe.get("category", ""),
            }
            if not question:
                errors += 1
                results.append({**entry, "model_answer": "", "error": "empty question"})
                continue
            try:
                entry["model_answer"] = generate_response(
                    question, temperature=PROBE_TEMPERATURE, max_tokens=PROBE_MAX_TOKENS
                )
            except Exception as e:
                # One bad probe must not lose the answers already collected.
                errors += 1
                entry["model_answer"] = ""
                entry["error"] = f"{type(e).__name__}: {e}"
            results.append(entry)

            progress = int(((i + 1) / len(probes)) * 100)
            update_job(job_id, status="STARTED", progress=progress, message=f"Probing {i + 1}/{len(probes)}")

        result = {"model": model_name, "results": results, "total": len(results), "errors": errors}
        update_job(job_id, status="SUCCESS", progress=100, message="Probing complete", result=result)
        sys_log(f"[kbd_probe] {job_id} done: {len(results)} answers, {errors} errors")
        return result

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        update_job(job_id, status="FAILURE", error=error)
        sys_log(f"[kbd_probe] {job_id} failed: {error}", level="ERROR")
        raise

    finally:
        release_loaded_model()
        accelerator.empty_cache()
