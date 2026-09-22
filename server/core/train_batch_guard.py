"""Batch guard — applies the VRAM estimates to a training params dict.

Called by the worker before the model is loaded. Rules:
  * Clamp only — never raises the requested batch size.
  * Unknown shape or free memory means skip; a guard must not block a valid run.
  * The one hard failure: even batch 1 with gradient checkpointing does not
    fit. A clear RuntimeError beats dying on OOM. ``TRAIN_BATCH_GUARD=0``
    disables it.
  * What was applied is recorded in ``params["_batch_guard_info"]``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from server.core.dataset_paths import storage_root
from server.core.logging import sys_log
from server.core.train_vram import clamp_to_limits, plan_batch, shape_from_config

GUARD_ENV = "TRAIN_BATCH_GUARD"
PLANNED_KEY = "_batch_planned"
INFO_KEY = "_batch_guard_info"
GRAD_CKPT_KEY = "_grad_checkpointing"


def guard_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """On by default; ``TRAIN_BATCH_GUARD=0`` turns it off."""
    if env is None:
        env = os.environ
    return (env.get(GUARD_ENV) or "1").strip() != "0"


def _resolve_model_path(params: Dict[str, Any]) -> Path:
    storage = storage_root()
    return Path(params.get("model_name_or_path") or (storage / "models" / params.get("model_name", "")))


def _load_shape(model_dir: Path):
    """Model shape from ``config.json``, with the parameter count read from disk."""
    cfg_file = Path(model_dir) / "config.json"
    if not cfg_file.exists():
        return None
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    return shape_from_config(cfg, params_b=_params_b_from_weights(Path(model_dir), cfg))


def _params_b_from_weights(model_dir: Path, config: dict) -> Optional[float]:
    """Parameter count in billions, measured from the weight files.

    File names lie — merged, re-trained and user-uploaded models often carry no
    size or the wrong one — but the bytes on disk do not.
    """
    try:
        from server.core.model_size import estimate_params_b

        entries = [(f.name, f.stat().st_size) for f in model_dir.iterdir() if f.is_file()]
        return estimate_params_b(entries, config)
    except Exception:
        return None


def safe_batch_for_model(model_dir: Path, method: str, seq_len: int, free_mb: float) -> Optional[int]:
    """Safe per-device batch ceiling for a model directory.

    Shared with the tuning agent and the UI capacity display so a value the
    guard would later clamp is never recommended in the first place. None when
    the shape or free memory is unknown.
    """
    try:
        from server.core.train_vram import safe_per_device_batch

        shape = _load_shape(model_dir)
        if shape is None or not free_mb or float(free_mb) <= 0:
            return None
        return safe_per_device_batch(shape, method, int(seq_len), float(free_mb))
    except Exception:
        return None


def apply_batch_guard(
    params: Dict[str, Any],
    free_mb: Optional[float],
    where: str = "",
) -> Optional[Dict[str, Any]]:
    """Clamp batch, accumulation and sequence length in place to what fits.

    Returns:
        A summary dict for the run record, or None when the guard is off, the
        plan is already fixed, or the shape / free memory is unknown.

    Raises:
        RuntimeError: even batch 1 does not fit — refuse clearly instead of
        dying on OOM.
    """
    if not guard_enabled() or params.get(PLANNED_KEY):
        return None
    if not free_mb or float(free_mb) <= 0:
        return None  # free memory unknown — do not block on a guess

    plan = None
    shape = None
    limit_reasons: "tuple[str, ...]" = ()
    try:
        shape = _load_shape(_resolve_model_path(params))
        if shape is None:
            return None

        method = str(params.get("mode") or params.get("method") or "lora")
        req_batch = int(params.get("per_device_train_batch_size") or params.get("batch_size") or 1)
        req_accum = int(params.get("gradient_accumulation_steps") or 4)
        req_seq = int(params.get("max_seq_length") or 2048)

        # Ceilings from host.toml [model_limits].
        max_batch = max_seq = None
        try:
            from server.core.host_config import HOST_CONFIG

            max_batch = HOST_CONFIG.model_limits.max_train_batch_size
            max_seq = HOST_CONFIG.model_limits.max_train_seq_length
        except Exception:
            pass
        capped_batch, capped_seq, limit_reasons = clamp_to_limits(req_batch, req_seq, max_batch, max_seq)

        # The effective-batch target follows the request before the ceiling:
        # capping per-device batch must not change the training semantics, so
        # accumulation compensates.
        target_eff = req_batch * req_accum

        plan = plan_batch(
            shape,
            method,
            capped_seq,
            float(free_mb),
            capped_batch,
            req_accum,
            target_effective=target_eff,
        )
    except Exception as e:  # noqa: BLE001 — a guard error must not block training
        sys_log(f"[batch_guard] skip ({where}): {e}", level="WARNING")
        return None

    if plan.per_device_batch == 0:
        raise RuntimeError(
            f"[batch_guard] this run does not fit — {'; '.join(plan.reasons)} "
            f"(model {shape.params_b:.1f}B/{method}, {float(free_mb) / 1024:.0f}GB free). "
            f"Use a smaller model or sequence length, or set TRAIN_BATCH_GUARD=0 to disable the guard."
        )

    # When only the ceiling applied, plan_batch returns early without adjusting
    # accumulation, so the effective batch is preserved here.
    final_accum = max(plan.grad_accum, math.ceil(target_eff / max(1, plan.per_device_batch)))
    adjusted = plan.adjusted or bool(limit_reasons) or final_accum != req_accum
    params["per_device_train_batch_size"] = plan.per_device_batch
    params["batch_size"] = plan.per_device_batch
    params["gradient_accumulation_steps"] = final_accum
    if capped_seq != req_seq:
        params["max_seq_length"] = capped_seq
    if plan.grad_checkpointing:
        params[GRAD_CKPT_KEY] = True
    params[PLANNED_KEY] = True

    info = {
        "where": where,
        "adjusted": adjusted,
        "per_device_batch": plan.per_device_batch,
        "grad_accum": final_accum,
        "grad_checkpointing": plan.grad_checkpointing,
        "est_vram_mb": round(plan.est_vram_mb),
        "free_mb": round(float(free_mb)),
        "reasons": list(limit_reasons) + list(plan.reasons),
    }
    params[INFO_KEY] = info
    if adjusted:
        sys_log(f"[batch_guard] adjusted ({where}): {'; '.join(info['reasons'])}", level="WARNING")
    return info
