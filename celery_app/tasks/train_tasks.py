"""Training tasks — SFT and LoRA runs on the celery ``gpu`` queue."""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from server.core import accelerator
from server.core.config import (
    CHECKPOINTS_DIR,
    TRAIN_SEED,
    TRAIN_TIMEOUT_SECONDS,
)
from server.core.dataset_paths import storage_root
from server.core.lang import resolve_lang
from server.core.logging import sys_log
from server.core.pipeline_config import load_pipeline_config
from server.core.pipeline_stage_texts import stage_text as _stage_text
from server.core.state import update_job
from server.core.train_batch_guard import apply_batch_guard
from server.core.train_dataset_record import dataset_record, resolve_dataset_path


def _free_device_mb() -> Optional[float]:
    """Free accelerator memory in MB, or None when there is no device."""
    free = accelerator.free_mb_by_index()
    return max(free.values()) if free else None


def _dataset_file_names(dataset_path: str) -> list:
    """File names inside a dataset folder. Empty on failure — only the record is thinner."""
    try:
        p = Path(dataset_path)
        if p.is_dir():
            return [f.name for f in p.iterdir() if f.is_file()]
        if p.is_file():
            return [p.name]
    except Exception:  # noqa: BLE001 — a record failure must not block training
        pass
    return []


def _cleanup_gpu_memory():
    """Order matters: collect first, then empty the cache.

    ``empty_cache()`` only returns blocks that are already fully free, so
    calling it before ``gc.collect()`` leaves the freshly freed blocks behind.
    """
    gc.collect()
    accelerator.empty_cache()


def _claim_job_slot(job_id: str, my_token: str) -> bool:
    """Take ownership of a job so a redelivered message cannot run it twice.

    The decision lives in ``server.core.job_singleflight``; this only does the
    Redis read and write. A Redis error means "allow" — the guard must never
    block a legitimate run.
    """
    from server.core.job_singleflight import DEFAULT_STALE_SECONDS, RUN, should_run

    try:
        from server.core.state import _get_redis

        r = _get_redis()
        key = f"train:owner:{job_id}"
        raw = r.get(key)
        owner_token, owner_hb = None, None
        if raw:
            try:
                info = json.loads(raw)
                owner_token, owner_hb = info.get("token"), float(info.get("heartbeat", 0))
            except (ValueError, TypeError):
                owner_token, owner_hb = None, None
        verdict = should_run(owner_token, my_token, owner_hb, time.time())
        if verdict != RUN:
            return False
        r.set(
            key,
            json.dumps({"token": my_token, "heartbeat": time.time()}),
            ex=int(DEFAULT_STALE_SECONDS * 4),
        )
        return True
    except Exception as e:
        sys_log(f"[train] duplicate-run guard skipped (Redis error): {e}", level="WARNING")
        return True


def _run_sft_training(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run one SFT job and return its result metrics.

    Training dependencies are imported lazily so the celery main process never
    loads torch or transformers.
    """
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Add project root to path for pipeline imports
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from pipelines.records import load_corpus_records, render_chat_sample

    # Resolve paths — support both direct paths and name-based lookup from router
    storage = storage_root()
    _ds_name = params.get("dataset_name", "")
    dataset_path = resolve_dataset_path(_ds_name, str(storage), params.get("dataset_path"))
    model_path = params.get("model_name_or_path") or str(storage / "models" / params.get("model_name", ""))
    output_dir = params.get("output_dir", str(CHECKPOINTS_DIR / "sft_output"))
    mode = params.get("mode") or params.get("method", "lora")
    seed = params.get("seed") or TRAIN_SEED

    # Load tokenizer and model
    torch_dtype = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Clamp batch / sequence length to what the free device memory can hold,
    # before any weights are loaded.
    apply_batch_guard(params, _free_device_mb(), where="train_sft")

    device = accelerator.torch_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Load and prepare dataset
    train_dataset = load_corpus_records(dataset_path)
    train_dataset = train_dataset.map(
        lambda ex: render_chat_sample(ex, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    train_dataset = train_dataset.shuffle(seed=seed)

    # SFT Config
    os.makedirs(output_dir, exist_ok=True)

    # TRL derives bf16 from fp16 when both are left unset, which raises on
    # hosts without bf16 (CPU, MPS, pre-Ampere GPUs). Set both explicitly.
    use_bf16 = accelerator.supports_bf16()
    use_fp16 = device == accelerator.CUDA and not use_bf16

    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=params.get("num_train_epochs") or params.get("epochs", 3),
        per_device_train_batch_size=params.get("per_device_train_batch_size") or params.get("batch_size", 1),
        learning_rate=params.get("learning_rate", 2e-4),
        max_grad_norm=params.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
        save_total_limit=params.get("save_total_limit", 2),
        logging_steps=params.get("logging_steps", 5),  # smaller default so short runs log multiple times
        save_strategy=params.get("save_strategy", "no"),
        save_steps=params.get("save_steps", 100),
        seed=seed,
        report_to=["tensorboard"],
        bf16=use_bf16,
        fp16=use_fp16,
    )
    # Only set when the batch guard turned it on to shrink activations.
    if params.get("_grad_checkpointing"):
        sft_kwargs["gradient_checkpointing"] = True
    try:
        sft_config = SFTConfig(max_seq_length=params.get("max_seq_length", 2048), **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(**sft_kwargs)

    # LoRA config
    peft_config = None
    if mode.lower() == "lora":
        target_modules = [
            m.strip() for m in params.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj").split(",") if m.strip()
        ]
        peft_config = LoraConfig(
            r=params.get("lora_r", 16),
            lora_alpha=params.get("lora_alpha", 32),
            lora_dropout=params.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    callbacks = []

    # Progress reporter — updates Redis `job:{id}` every few steps so the UI
    # progress bar animates mid-training instead of jumping 0% → 100%.
    # Resolves `job_id` lazily to avoid cross-scope capture surprises.
    job_id_for_progress = params.get("job_id")
    _progress_lang = resolve_lang(config=load_pipeline_config(str(params.get("user_id") or "")))
    if job_id_for_progress:
        try:
            from transformers import TrainerCallback

            class _ProgressReporter(TrainerCallback):
                """Report progress, and stop the run when the user asks.

                A revoke cannot terminate a running task under the thread pool
                (macOS), so the stop request is honoured here instead: the flag
                is read from the job state and turned into the trainer's own
                ``should_training_stop``.
                """

                #: Seconds between stop-flag reads, so the check costs one Redis
                #: round trip every few steps rather than one per step.
                STOP_POLL_SECONDS = 2.0

                def __init__(self, jid: str, lang: str):
                    self._jid = jid
                    self._lang = lang
                    self._last_pct = -1
                    self._last_loss: Optional[float] = None
                    self._last_stop_check = 0.0

                def _stop_requested(self) -> bool:
                    now = time.time()
                    if now - self._last_stop_check < self.STOP_POLL_SECONDS:
                        return False
                    self._last_stop_check = now
                    try:
                        from server.core.state import get_job

                        return str((get_job(self._jid) or {}).get("status", "")).upper() == "REVOKED"
                    except Exception:  # noqa: BLE001 — a failed check must not stop a healthy run
                        return False

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    if "loss" in logs:
                        try:
                            self._last_loss = float(logs["loss"])
                        except (TypeError, ValueError):
                            pass

                def on_step_end(self, args, state, control, **kwargs):
                    if not state.is_world_process_zero:
                        return
                    if self._stop_requested():
                        control.should_training_stop = True
                        sys_log(f"[train_sft] {self._jid}: stop requested — ending after this step")
                        return
                    total = state.max_steps or 0
                    step = state.global_step or 0
                    pct = int((step / total) * 100) if total > 0 else 0
                    # Only push when % moved (cut Redis traffic)
                    if pct != self._last_pct:
                        self._last_pct = pct
                        try:
                            update_job(
                                self._jid,
                                status="STARTED",
                                progress=pct,
                                message=(
                                    _stage_text("train_progress", self._lang, step=step, total=total)
                                    + (
                                        _stage_text(
                                            "train_loss_suffix",
                                            self._lang,
                                            loss=f"{self._last_loss:.4f}",
                                        )
                                        if self._last_loss is not None
                                        else ""
                                    )
                                ),
                            )
                        except Exception:
                            pass  # never fail training because of progress reporting

            callbacks.append(_ProgressReporter(job_id_for_progress, _progress_lang))
        except Exception as _cb_err:
            sys_log(f"[train_sft] progress callback skipped: {_cb_err}", level="DEBUG")

    # Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks if callbacks else None,
    )

    train_result = trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Persist audit artifacts for the results modal ───────────────────
    # 1) training_params.json — caller's input (method, model, dataset, ...)
    # 2) training_results.json — output metrics (train_runtime, loss, tokens/sec, …)
    # 3) trainer_state.json    — log_history (per-step/per-epoch loss/acc)
    # The UI reads #2 for the summary card and #3 for the Loss/Accuracy charts.
    metrics: Dict[str, Any] = {}
    try:
        metrics = dict(getattr(train_result, "metrics", None) or {})
    except Exception:
        metrics = {}
    if not metrics and hasattr(train_result, "training_loss"):
        metrics["train_loss"] = train_result.training_loss
    if "global_step" not in metrics and hasattr(train_result, "global_step"):
        metrics["global_step"] = train_result.global_step

    training_params = {
        "output_dir": output_dir,
        "model_path": model_path,
        "mode": mode,
        "seed": seed,
        "epochs_completed": params.get("num_train_epochs") or params.get("epochs", 3),
        "batch_size": params.get("per_device_train_batch_size") or params.get("batch_size"),
        "learning_rate": params.get("learning_rate"),
        "max_grad_norm": params.get("max_grad_norm", 1.0),
        "max_seq_length": params.get("max_seq_length"),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 4),
        "batch_guard": params.get("_batch_guard_info"),  # what the batch guard clamped
        **dataset_record(dataset_path, _dataset_file_names(dataset_path)),
    }
    training_results = {
        **training_params,
        # Merge metrics on top so train_runtime/train_loss/... are at the top
        # level where the UI's `trainResult` selector reads them from.
        **metrics,
    }

    try:
        with open(os.path.join(output_dir, "training_params.json"), "w", encoding="utf-8") as f:
            json.dump(training_params, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        sys_log(f"[train_sft] training_params.json write failed: {e}", level="WARNING")

    try:
        with open(os.path.join(output_dir, "training_results.json"), "w", encoding="utf-8") as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        sys_log(f"[train_sft] training_results.json write failed: {e}", level="WARNING")

    # trainer_state.json — log_history is captured in trainer.state during
    # training. When save_strategy="no" HF doesn't persist it; dump it here so
    # the Loss / Accuracy charts have data to render.
    try:
        state = getattr(trainer, "state", None)
        if state is not None and hasattr(state, "log_history"):
            trainer_state = {
                "log_history": list(state.log_history or []),
                "global_step": getattr(state, "global_step", None),
                "epoch": getattr(state, "epoch", None),
                "max_steps": getattr(state, "max_steps", None),
            }
            with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        sys_log(f"[train_sft] trainer_state.json write failed: {e}", level="WARNING")

    return training_results


def _stop_was_requested(job_id: str) -> bool:
    """Whether /api/train/stop marked this job while it was running."""
    try:
        from server.core.state import get_job

        return str((get_job(job_id) or {}).get("status", "")).upper() == "REVOKED"
    except Exception:  # noqa: BLE001
        return False


@shared_task(
    name="celery_app.tasks.train_tasks.train_sft_task",
    bind=True,
    soft_time_limit=int(os.getenv("TRAIN_TIMEOUT_SECONDS", "7200")),
    time_limit=int(os.getenv("TRAIN_TIMEOUT_SECONDS", "7200")) + 300,
    max_retries=0,
)
def train_sft_task(self, **params) -> Dict[str, Any]:
    """Run one SFT job, updating the Redis job state as it goes."""
    job_id = params.get("job_id") or self.request.id or "unknown"

    # Admission gate — re-queues the task when another job holds the device.
    from server.core.gpu_admission import gate_or_retry

    gate_or_retry(self, "training", job_id, user_id=str(params.get("user_id") or ""))
    user_id = params.get("user_id", "system")

    sys_log(f"[train_sft] Starting job {job_id} for user {user_id}")

    own_token = str(self.request.id or job_id)
    if not _claim_job_slot(job_id, own_token):
        sys_log(f"[train_sft] job {job_id} is already running — skipping this delivery", level="WARNING")
        return {"skipped": "duplicate", "job_id": job_id}

    try:
        update_job(job_id, status="STARTED", progress=0, message="Training started")
        result = _run_sft_training(params)
        # A user stop ends the trainer loop normally, so without this check the
        # run would be reported as a completed 100% job.
        if _stop_was_requested(job_id):
            result["stopped"] = True
            update_job(job_id, status="REVOKED", result=result)
            sys_log(f"[train_sft] Job {job_id} stopped by the user")
            return result
        update_job(
            job_id,
            status="SUCCESS",
            progress=100,
            message="Training completed",
            result=result,
        )
        sys_log(f"[train_sft] Job {job_id} completed successfully")
        return result

    except SoftTimeLimitExceeded:
        update_job(
            job_id,
            status="FAILURE",
            error=f"Training timed out after {TRAIN_TIMEOUT_SECONDS}s",
        )
        sys_log(f"[train_sft] Job {job_id} timed out", level="ERROR")
        raise

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        update_job(job_id, status="FAILURE", error=error_msg)
        sys_log(f"[train_sft] Job {job_id} failed: {error_msg}", level="ERROR")
        raise

    finally:
        _cleanup_gpu_memory()
