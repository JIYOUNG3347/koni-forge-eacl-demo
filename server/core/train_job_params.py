"""Recover or explain missing tool parameters for a training job.

A local model does not honour the tool schema's ``required`` list, and indexing
the dict directly raises a bare ``KeyError('job_name')`` — which tells the LLM
nothing, so it repeats the same call until the loop guard stops it. Recover what
can be recovered, and name what cannot in a message the model can act on.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Values the code cannot derive; without these there is no training job at all.
UNRECOVERABLE = ("base_model", "dataset")

_TRAINING_SUFFIX = "_training"


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def derive_job_name(dataset: str) -> str:
    """Build a training job name from the dataset: ``{dataset}_training``.

    An existing suffix is not doubled (no ``x_training_training``).
    """
    name = _clean(dataset)
    if not name:
        return ""
    return name if name.endswith(_TRAINING_SUFFIX) else f"{name}{_TRAINING_SUFFIX}"


def missing_required(params: Mapping[str, Any]) -> list:
    """Unrecoverable missing parameters. Empty means the job can proceed."""
    return [key for key in UNRECOVERABLE if not _clean(params.get(key))]


def missing_params_message(missing: list, lang: Optional[str] = None) -> str:
    """An error message the LLM can read and fix itself.

    Returning only the KeyError string ('job_name') tells the model nothing.
    """
    from server.core.agent_runtime_texts import text

    return text("train_missing_params", lang, fields=", ".join(missing))


def resolve_training_params(
    params: Mapping[str, Any],
    *,
    fallback_dataset: Optional[str] = None,
    fallback_base_model: Optional[str] = None,
    lang: Optional[str] = None,
) -> dict:
    """Recover and validate the tool parameters.

    Returns:
        ``{"ok": True, "params": {...}}`` or
        ``{"ok": False, "missing": [...], "message": "..."}``.

    """
    resolved = dict(params)

    dataset = _clean(resolved.get("dataset")) or _clean(fallback_dataset)
    if dataset:
        resolved["dataset"] = dataset

    base_model = _clean(resolved.get("base_model")) or _clean(fallback_base_model)
    if base_model:
        resolved["base_model"] = base_model

    # job_name can be derived from the dataset, so its absence is not fatal.
    if not _clean(resolved.get("job_name")):
        derived = derive_job_name(dataset)
        if derived:
            resolved["job_name"] = derived

    missing = missing_required(resolved)
    if missing:
        return {"ok": False, "missing": missing, "message": missing_params_message(missing, lang)}
    return {"ok": True, "params": resolved}
