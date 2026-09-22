"""Propagate pipeline training parameters through the agent path.

**The request wins, not the LLM.** A value the user set on the pipeline
outranks the LLM's guess. Keys the pipeline did not set are left alone.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Keys naming a model. The same model travels as a path (`/storage/models/X`) and
# as a name (`X`), because the tool normalises with
# `params["base_model"].replace("/storage/models/", "")`. A false mismatch from
# that difference would stop training before it starts.
_MODEL_KEYS = frozenset({"base_model", "model_name", "model_name_or_path"})

#: Tool argument name to **final payload** key, for the keys whose names differ.
#: Without an alias, `verify_propagation` reports "absent" and refuses training.
PAYLOAD_ALIAS: Dict[str, str] = {"base_model": "model_name"}

# Integer, float and string keys, normalised before comparison.
_INT_KEYS = frozenset({"epochs", "batch_size", "max_seq_length", "lora_r", "lora_alpha"})
_FLOAT_KEYS = frozenset({"learning_rate", "lora_dropout", "max_grad_norm"})

# "fft" is a legacy spelling, equivalent to "sft" (same rule as tuning_specialist).
_METHOD_ALIAS = {"fft": "sft"}


def _norm(key: str, value: Any) -> Any:
    """Normalise for comparison. A failed conversion returns the original."""
    if value is None:
        return None
    try:
        if key in _MODEL_KEYS:
            return str(value).replace("/storage/models/", "").strip("/")
        if key == "method":
            v = str(value).strip().lower()
            return _METHOD_ALIAS.get(v, v)
        if key in _INT_KEYS:
            return int(float(value))
        if key in _FLOAT_KEYS:
            return float(value)
    except (TypeError, ValueError):
        return value
    return value


def _same(key: str, a: Any, b: Any) -> bool:
    na, nb = _norm(key, a), _norm(key, b)
    if isinstance(na, float) and isinstance(nb, float):
        # So a float spelling difference (2e-4 vs 0.0002) is not a false mismatch.
        return abs(na - nb) <= max(abs(na), abs(nb), 1.0) * 1e-9
    return na == nb


def apply_train_overrides(
    tool_params: Optional[Mapping[str, Any]],
    overrides: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Overwrite tool arguments with the pipeline request.

    Returns ``(merged, applied)``, where ``applied[key] = {"from": ..., "to": ...}``
    covers only the keys whose value actually changed. Without overrides the input
    is copied unchanged.
    """
    merged: Dict[str, Any] = dict(tool_params or {})
    applied: Dict[str, Dict[str, Any]] = {}
    for key, val in (overrides or {}).items():
        before = merged.get(key)
        if not _same(key, before, val):
            applied[key] = {"from": before, "to": val}
        merged[key] = val
    return merged, applied


def verify_propagation(
    overrides: Optional[Mapping[str, Any]],
    effective: Optional[Mapping[str, Any]],
    exempt: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Check the requested values survived into the final payload. Returns mismatches."""
    ex = set(exempt or ())
    out: List[Dict[str, Any]] = []
    eff = effective or {}
    for key, want in (overrides or {}).items():
        if key in ex:
            continue
        eff_key = key if key in eff else PAYLOAD_ALIAS.get(key, key)
        if eff_key not in eff:
            out.append({"key": key, "requested": want, "effective": None, "reason": "absent"})
            continue
        if not _same(key, want, eff[eff_key]):
            out.append({"key": key, "requested": want, "effective": eff[eff_key], "reason": "mismatch"})
    return out


def propagation_error_message(mismatches: Iterable[Mapping[str, Any]]) -> str:
    """Turn mismatches into a message that says this is not the LLM's to fix."""
    items = list(mismatches or [])
    if not items:
        return ""
    detail = ", ".join(f"{m.get('key')}: requested={m.get('requested')!r} effective={m.get('effective')!r}" for m in items)
    return (
        "Training parameter propagation failed — the values the pipeline requested did "
        f"not reach the actual training configuration ({detail}). This is an internal "
        "error and cannot be fixed by changing tool arguments. Training was not started."
    )
