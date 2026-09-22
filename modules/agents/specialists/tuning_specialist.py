"""
KONI-Forge TuningSpecialist — Training hyperparameter recommendation and monitoring agent

Roles:
- Recommend hyperparameters for dataset/model/GPU
- Monitor training progress and analysis
- Interpret training logs and diagnose issues
- Checkpoint management advice
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx

from server.core.agent_tool_errors import error_text as _err
from server.core.corpus_files import count_samples
from server.core.dataset_paths import find_dataset_file

from ..foundation import AgentBase

logger = logging.getLogger("agent.tuning")
INTERNAL_API_URL = os.getenv("INTERNAL_API_URL", "http://localhost:8000")
_INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")


def _internal_headers(on_behalf_of: str | None = None) -> dict:
    h = {"X-Internal-Token": _INTERNAL_TOKEN}
    # "default" is AgentBase's initial placeholder; "system" is not an impersonation target.
    if on_behalf_of and on_behalf_of not in ("default", "system"):
        h["X-On-Behalf-Of"] = on_behalf_of
    return h


class TuningSpecialist(AgentBase):
    default_autonomy = "guided"
    name = "tuning"
    description = "Recommends and monitors training hyperparameters"
    _llm_max_retries = 8  # 9 attempts; leaves ~1% residual failure on a flaky link.
    _llm_timeout = httpx.Timeout(600.0, connect=10.0)
    system_prompt = """You are the LLM fine-tuning specialist of KONI-Forge.

Role:
- Recommend hyperparameters for SFT (supervised fine-tuning).
- Weigh full fine-tuning (FFT) against LoRA given the data size and GPU.
- Interpret training logs (loss curve, gradient norm) and diagnose problems (OOM, loss spikes, divergence).
- Advise on checkpoint selection.

== Pipeline auto-run mode ==
If the message contains "[pipeline auto run]", the user has already confirmed: do not ask again. Immediately run list_models -> get_gpu_info -> list_datasets -> start_training_job.

== Pipeline guided-run mode ==
If the message contains "[pipeline guided run]": first call list_models -> get_gpu_info -> list_datasets, summarise the recommended settings (model, method, lr, epochs, batch size), and end with "Shall I start training with these settings?". Call start_training_job only after the user confirms.

== KBD integration (highest priority) ==
If the message contains [KBD analysis], these rules outrank the general guidance:
- hallucination rate ≥ 15 %  -> conservative learning rate (5e-5)
- coverage < 30 %            -> recommend FFT even with little data (small ~1B models fit FFT in 20 GB VRAM)
- coverage 30–50 %           -> LoRA with more epochs (8+)
- coverage ≥ 50 %            -> LoRA with fewer epochs (3–5)
- If weak categories are listed, suggest focusing the training data on them.
If the KBD verdict is "Fine-Tuning" or "Hybrid", default to FFT. Apply the general guidance below only when there is no KBD result.

== General guidance (no KBD result) ==
State explicitly whether you recommend LoRA or FFT, and let the user pick the other:
- **LoRA**: memory-efficient, good for small datasets, fast. Prefer it with ≤ 16 GB VRAM or < 500 samples. Defaults: lora_r=16, lora_alpha=32, lora_dropout=0.1.
- **FFT**: best quality for large datasets, needs more VRAM. Consider it with ≥ 24 GB VRAM and ≥ 1000 samples; small (~1B) models are fine either way.

Recommendation format:
"Recommended training settings:
- Method: **LoRA** (small dataset, memory-efficient) — FFT is also possible
- Base model: ...
- ..."

Give practical, concrete numbers based on VRAM, dataset size and model size, and show the VRAM / batch-size reasoning.

== When the user confirms ==
When the user says "yes", "go ahead", "start" or similar:
- Do not restate the plan or say "starting" without acting.
- Call start_training_job immediately with the parameters you already recommended."""

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "list_models",
                "description": "List the available base models and trained models.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_gpu_info",
                "description": "Read the current device status (free memory, utilisation, temperature).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_job_history",
                "description": "List recent training jobs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "How many jobs to return (default 10)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "read_training_log",
                "description": "Read the tail of a training job log file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Job id",
                        },
                        "lines": {
                            "type": "integer",
                            "description": "How many lines to read (default 50)",
                        },
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "list_datasets",
                "description": "List the datasets available for training.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "start_training_job",
                "description": "Start a training job. Takes the model, dataset and hyperparameters and runs full fine-tuning or LoRA.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "job_name": {
                            "type": "string",
                            "description": "Training job name",
                        },
                        "base_model": {
                            "type": "string",
                            "description": "Base model name (e.g. Qwen--Qwen2-0.5B-Instruct)",
                        },
                        "dataset": {
                            "type": "string",
                            "description": "Training dataset name (e.g. qa_dataset_20260317_twist)",
                        },
                        "method": {
                            "type": "string",
                            "description": "Method: 'sft' (full fine-tuning) or 'lora'. Default: lora",
                            "enum": ["sft", "lora"],
                        },
                        "epochs": {
                            "type": "string",
                            "description": "Number of epochs (default 3)",
                        },
                        "batch_size": {
                            "type": "string",
                            "description": "Batch size (default 4)",
                        },
                        "learning_rate": {
                            "type": "string",
                            "description": "Learning rate (default 2e-5)",
                        },
                        "max_seq_length": {
                            "type": "integer",
                            "description": "Maximum sequence length (default 2048)",
                        },
                        "lora_r": {
                            "type": "integer",
                            "description": "LoRA rank (only with method=lora, default 8)",
                        },
                        "lora_alpha": {
                            "type": "integer",
                            "description": "LoRA alpha (only with method=lora, default 16)",
                        },
                        "lora_dropout": {
                            "type": "number",
                            "description": "LoRA dropout (only with method=lora, default 0.1)",
                        },
                    },
                    "required": ["job_name", "base_model", "dataset"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        if tool_name == "list_models":
            return self._list_models()
        elif tool_name == "get_gpu_info":
            return self._get_gpu_info()
        elif tool_name == "get_job_history":
            return self._get_job_history(tool_input.get("limit", 10))
        elif tool_name == "read_training_log":
            return self._read_training_log(
                tool_input["task_id"],
                tool_input.get("lines", 50),
            )
        elif tool_name == "list_datasets":
            return self._list_datasets()
        elif tool_name == "start_training_job":
            return await self._start_training_job(tool_input)
        return f"Unknown tool: {tool_name}"

    def _list_models(self) -> str:
        result = {"base_models": [], "trained_models": [], "checkpoints": []}

        models_dir = Path(os.getenv("STORAGE_BASE_PATH", "storage")) / "models"
        if models_dir.exists():
            for d in sorted(models_dir.iterdir()):
                if d.is_dir():
                    has_config = (d / "config.json").exists()
                    has_adapter = (d / "adapter_config.json").exists()
                    entry = {"name": d.name, "has_config": has_config}
                    if has_adapter:
                        entry["type"] = "lora_adapter"
                    result["base_models"].append(entry)

        outs_dir = self.resolve_path()
        if outs_dir.exists():
            for d in sorted(outs_dir.iterdir()):
                if d.is_dir() and d.name != "logs":
                    has_config = (d / "config.json").exists()
                    has_adapter = (d / "adapter_config.json").exists()
                    checkpoints = [c.name for c in d.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")]
                    entry = {
                        "name": d.name,
                        "type": "lora" if has_adapter else "fft",
                        "checkpoints": len(checkpoints),
                    }
                    result["trained_models"].append(entry)

        return json.dumps(result, ensure_ascii=False)

    def _get_gpu_info(self) -> str:
        """Device memory, used to size the training run."""
        from server.core import accelerator

        try:
            info = accelerator.probe()
            return json.dumps(
                {
                    "backend": info["backend"],
                    "gpus": info["gpus"],
                    "_next_step": self._rt("next_list_training_datasets"),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": _err("gpu_query_failed", self._lang, error=str(e))})

    def _get_job_history(self, limit: int = 10) -> str:
        tasks_file = self.resolve_path("jobs_history.json")
        if not tasks_file.exists():
            return json.dumps({"tasks": [], "message": self._rt("res_no_job_history")})

        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)

            # Sort by created_at descending, take limit
            if isinstance(tasks, list):
                tasks = sorted(tasks, key=lambda j: j.get("created_at", 0), reverse=True)
                tasks = tasks[:limit]

            # Summarize for agent
            summary = []
            for j in tasks:
                summary.append(
                    {
                        "id": j.get("id", ""),
                        "name": j.get("name", ""),
                        "method": j.get("method", ""),
                        "status": j.get("status", ""),
                        "progress": j.get("progress", 0),
                        "created_at": j.get("created_at", 0),
                    }
                )
            return json.dumps({"tasks": summary}, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": _err("job_history_failed", self._lang, error=str(e))})

    def _read_training_log(self, task_id: str, lines: int = 50) -> str:
        # Look for log file in common locations
        log_paths = [
            Path(f"/tmp/kf-task-{task_id}.log"),
            Path(f"/app/logs/{task_id}.log"),
            self.resolve_path(task_id, "training.log"),
        ]

        for log_path in log_paths:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        all_lines = f.readlines()
                    tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    return json.dumps(
                        {
                            "task_id": task_id,
                            "log_path": str(log_path),
                            "total_lines": len(all_lines),
                            "content": "".join(tail),
                        },
                        ensure_ascii=False,
                    )
                except Exception as e:
                    return json.dumps({"error": _err("log_read_failed", self._lang, error=str(e))})

        return json.dumps({"error": _err("log_file_not_found", self._lang, task_id=task_id)})

    def _list_datasets(self) -> str:
        """Query available training datasets.

        The training endpoint looks in both roots (`train.py`) and the trainer picks
        files with `select_data_files` — this tool follows the same rules.

        """
        from server.core.agent_datasets import DATASET_ROOTS, listing_hint, order_datasets
        from server.core.corpus_files import select_data_files

        datasets: List[Dict[str, Any]] = []
        seen: set = set()
        for root in DATASET_ROOTS:
            base = Path(root)
            if not base.exists():
                continue
            for d in sorted(base.iterdir()):
                if not d.is_dir() or d.name in seen:
                    continue
                seen.add(d.name)
                names = [f.name for f in d.iterdir() if f.is_file()]
                selected = select_data_files(names)
                info: Dict[str, Any] = {
                    "name": d.name,
                    "has_data": bool(selected),
                    "root": root,
                }
                if selected:
                    info["sample_count"] = sum(count_samples(d / fname) for fname in selected)
                    info["is_twist"] = d.name.endswith("_twist")
                    info["data_files"] = selected
                datasets.append(info)

        # A dataset chosen by the pipeline goes first, with an instruction to use it.
        # Letting the LLM pick from the list instead would train the wrong data.
        requested = self._pipeline_identity_hint("dataset")
        datasets = order_datasets(datasets, requested)
        trainable = any(d.get("has_data") for d in datasets)
        result: Dict[str, Any] = {
            "datasets": datasets,
            "count": len(datasets),
            # Pointing at start_training_job with nothing to train on sends the
            # LLM into inventing a dataset name and looping on a 404.
            "_next_step": self._rt("next_start_training" if trainable else "next_upload_training_data"),
        }
        hint = listing_hint(datasets, requested, self._lang)
        if hint:
            result["instruction"] = hint
            if requested and any(str(d.get("name")) == requested for d in datasets):
                result["pipeline_dataset"] = requested
        return json.dumps(result, ensure_ascii=False)

    def _pipeline_identity_hint(self, key: str) -> str:
        """Training target fixed by the pipeline, used when a tool argument is missing.

        Reads what the dispatcher pushed in through ``set_train_identity``. Unknown
        means an empty string, and ``resolve_training_params`` then raises an honest
        error rather than starting a run on a guess.

        """
        identity = getattr(self, "_pipeline_train_identity", None) or {}
        value = identity.get(key, "")
        return value.strip() if isinstance(value, str) else ""

    def _small_data_note(self, info: dict) -> str:
        """Describe the small-data adjustment in a sentence."""
        if not info or not info.get("adjusted"):
            return ""
        before = info.get("before") or {}
        after = info.get("after") or {}
        note = self._rt(
            "small_data_note",
            samples=info.get("sample_count"),
            before_epochs=before.get("epochs"),
            before_batch=before.get("batch_size"),
            estimated=info.get("estimated_steps"),
            after_epochs=after.get("epochs"),
            after_batch=after.get("batch_size"),
        )
        return note

    def _training_identity(self, job_dir: Path) -> Dict[str, Any]:
        """``_start_training_job`` holds these as locals, but the status tool
        (`_get_spectrabench_status`) only knows the folder. Two producers of the
        same signal must write the same keys, or the report differs by path.

        The model name is the bare name, without the ``/storage/models/`` prefix.
        When it cannot be read the key is dropped rather than invented, so the
        reader's own default stands.

        """
        out: Dict[str, Any] = {}
        try:
            params_path = job_dir / "training_params.json"
            if not params_path.exists():
                return out
            params = json.loads(params_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"[TuningSpecialist] could not read training_params.json ({job_dir}): {exc}")
            return out

        model_path = str(params.get("model_path") or "")
        if model_path:
            out["model_name"] = model_path.replace("/storage/models/", "").strip("/")
        mode = str(params.get("mode") or "")
        if mode:
            out["method"] = mode
        return out

    async def _start_training_job(self, params: Dict[str, Any]) -> str:
        """Start training job and poll until completion."""
        import asyncio
        import time as _time

        import httpx

        from server.core.train_job_params import resolve_training_params

        _resolved = resolve_training_params(
            params,
            fallback_dataset=self._pipeline_identity_hint("dataset"),
            fallback_base_model=self._pipeline_identity_hint("base_model"),
            lang=self._lang,
        )
        if not _resolved["ok"]:
            logger.warning(f"[TuningSpecialist] start_training_job is missing required parameters: {_resolved['missing']}")
            return json.dumps({"error": _resolved["message"]}, ensure_ascii=False)
        params = _resolved["params"]

        from server.core.train_overrides import (
            apply_train_overrides,
            propagation_error_message,
            verify_propagation,
        )

        _overrides = dict(getattr(self, "_pipeline_train_overrides", None) or {})
        if _overrides:
            params, _applied = apply_train_overrides(params, _overrides)
            logger.info(
                f"[TuningSpecialist] applied pipeline training parameters: {_overrides} (changed: {_applied or 'none'})"
            )

        job_name = params["job_name"]
        # Duplicate names are handled automatically: filesystem check plus a timestamp suffix.
        _completed = self.resolve_path("completed")
        _temp = self.resolve_path("temp")
        _orig_name = job_name
        _suffix = 2
        while (_completed / job_name).exists() or (_temp / job_name).exists():
            job_name = f"{_orig_name}_{_suffix}"
            _suffix += 1
        base_model = params["base_model"].replace("/storage/models/", "").strip("/")
        dataset = params["dataset"].replace("/dataset/", "").strip("/")
        # Normalise a doubled _twist suffix from the LLM (foo_twist_twist -> foo_twist).
        while dataset.endswith("_twist_twist"):
            dataset = dataset[: -len("_twist")]
        logger.info(
            f"[TuningSpecialist] start_training_job params: base_model={base_model!r} dataset={dataset!r} method={params.get('method')!r}"
        )
        # "fft" is a legacy value, equivalent to "sft" internally.
        _raw_method = str(params.get("method", "lora")).lower()
        method = "sft" if _raw_method in ("fft", "sft") else "lora"
        epochs = int(float(params.get("epochs", "3")))
        batch_size = int(float(params.get("batch_size", "4")))
        lr = params.get("learning_rate", "2e-5")

        _dataset_path = find_dataset_file(dataset)
        _sample_count = count_samples(_dataset_path) if _dataset_path is not None else 0
        if _sample_count <= 0:
            logger.warning(
                "[TuningSpecialist] sample count unknown (dataset=%r, file=%s) — small-data adjustment skipped",
                dataset,
                _dataset_path,
            )

        _pre_adjust = {"epochs": epochs, "batch_size": batch_size}

        from server.core.small_data_adjust import plan as _small_data_plan

        epochs, batch_size, _small_data = _small_data_plan(_sample_count, epochs, batch_size)
        if _small_data["adjusted"]:
            logger.info(
                f"Small-data adjustment: {_sample_count} samples, batch={batch_size}, epochs={epochs}, "
                f"steps={_small_data.get('planned_steps')}"
            )

        # Build payload matching /api/train/start's Pydantic schema exactly —
        # fields are FLAT (model_name / dataset_name / max_seq_length / lora_*
        # Flat fields only; a nested `params: {...}` is rejected with
        # 422 so agent-driven training has been broken; this fixes it.
        try:
            lr_float = float(lr) if not isinstance(lr, float) else lr
        except (TypeError, ValueError):
            lr_float = 2e-5

        payload: Dict[str, Any] = {
            "job_name": job_name,
            "model_name": base_model,
            "dataset_name": dataset,
            "method": method,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": lr_float,
            "max_grad_norm": float(params.get("max_grad_norm", 1.0)),
            "max_seq_length": int(params.get("max_seq_length", 2048)),
        }

        if method == "lora":
            payload["lora_r"] = int(params.get("lora_r", 8))
            payload["lora_alpha"] = int(params.get("lora_alpha", 16))
            payload["lora_dropout"] = float(params.get("lora_dropout", 0.1))

        if _overrides:
            _exempt = set()
            for _k, _before in _pre_adjust.items():
                if _k in payload and str(payload[_k]) != str(_before):
                    _exempt.add(_k)
            if method != "lora":
                _exempt |= {"lora_r", "lora_alpha", "lora_dropout"}
            if _exempt:
                logger.info(
                    f"[TuningSpecialist] exempt from propagation check (safety adjustment): {sorted(_exempt)} "
                    f"(before adjustment {_pre_adjust}, final epochs={payload['epochs']} batch={payload['batch_size']})"
                )
            _mismatch = verify_propagation(_overrides, payload, exempt=_exempt)
            if _mismatch:
                _emsg = propagation_error_message(_mismatch)
                logger.error(f"[TuningSpecialist] {_emsg}")
                return json.dumps({"error": _emsg, "mismatches": _mismatch}, ensure_ascii=False)

        job_id = None
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                res = await client.post(
                    f"{INTERNAL_API_URL}/api/train/start", json=payload, headers=_internal_headers(self._user_id)
                )
                if res.status_code == 409:
                    # The same name still exists in Redis — retry with a timestamp suffix.
                    ts = _time.strftime("%H%M%S")
                    job_name = f"{_orig_name}_{ts}"
                    payload["job_name"] = job_name
                    res = await client.post(
                        f"{INTERNAL_API_URL}/api/train/start", json=payload, headers=_internal_headers(self._user_id)
                    )
                if res.status_code != 200:
                    detail = res.text[:300]
                    return json.dumps(
                        {"error": _err("train_start_http_failed", self._lang, status=res.status_code, detail=detail)}
                    )
                job_id = res.json().get("job_id")
        except Exception as e:
            return json.dumps({"error": _err("train_start_failed", self._lang, error=str(e))})

        if not job_id:
            return json.dumps({"error": _err("train_no_job_id", self._lang)})

        if getattr(self, "_pipeline_run_id", None):
            try:
                import os as _os

                import redis as _r_lib

                from server.core.job_revoke import with_sub_task

                _r = _r_lib.from_url(
                    _os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True,
                )
                _key = f"pipeline:state:{self._pipeline_run_id}"
                _raw = _r.get(_key)
                if _raw:
                    _r.set(_key, json.dumps(with_sub_task(json.loads(_raw), job_id), default=str))
            except Exception as _e:  # noqa: BLE001 — a failed registration must not block training
                logger.warning(f"[Tuning] sub_task registration failed ({job_id}): {_e}")

        # Poll for completion via GET /api/train/status/{job_id}.
        # NOTE: filesystem polling (completed_dir.exists()) was removed because
        # The status API is authoritative: not every job writes to
        # outputs/completed/.

        for _ in range(14400):  # ~12 hours at 3s intervals
            await asyncio.sleep(3)
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    st_res = await client.get(
                        f"{INTERNAL_API_URL}/api/train/status/{job_id}",
                        headers=_internal_headers(self._user_id),
                    )
                if st_res.status_code != 200:
                    continue
                st = st_res.json()
                job_status = st.get("status", "")

                if job_status == "completed":
                    output_dir = st.get("output_dir", "")
                    _adjust_note = self._small_data_note(_small_data)
                    return json.dumps(
                        {
                            "success": True,
                            "message": self._rt("res_training_done", job=job_name)
                            + (f" {_adjust_note}" if _adjust_note else ""),
                            "small_data_adjustment": _small_data,
                            "job_name": job_name,
                            "method": method,
                            "output_dir": output_dir,
                            "__chain_signal__": {
                                "stage_completed": "tuning",
                                "metadata": {
                                    "task_id": job_name,
                                    "model_name": base_model,
                                    "method": method,
                                },
                            },
                        },
                        ensure_ascii=False,
                    )

                elif job_status in ("failed", "stopped"):
                    err_detail = st.get("error") or st.get("message") or job_status
                    return json.dumps(
                        {
                            "error": _err("train_job_ended", self._lang, status=job_status, detail=err_detail),
                            "job_name": job_name,
                        },
                        ensure_ascii=False,
                    )

            except Exception:
                pass

        return json.dumps({"error": _err("train_timeout", self._lang)})
