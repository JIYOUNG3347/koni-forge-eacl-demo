"""Resolve the base model for a training run, including trained outputs.

A completed fine-tune lives under ``outputs/{user}/completed/...``, not
``{storage}/models``, so both are searched. An optional ``model_path`` is
confined to the allowed roots, checked for existence, and returned absolute.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, Path]


def allowed_model_roots(storage_root: Path, user_id: str) -> List[Path]:
    """Roots allowed as a training base: base models plus training outputs.

    * ``{storage}/models``                      — base (HuggingFace) models
    * ``{storage}/outputs/{user}/completed``    — completed training output
    * ``{storage}/checkpoints``                 — legacy and pipeline checkpoints
    """
    return [
        storage_root / "models",
        storage_root / "outputs" / user_id / "completed",
        storage_root / "checkpoints",
    ]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_train_model_source(
    model_name: str,
    model_path: Optional[str],
    storage_root: PathLike,
    user_id: str,
) -> str:
    """Resolve and validate the absolute path of the training base model.

    Args:
        model_name: base model folder under ``{storage}/models``, used without model_path.
        model_path: full path to a trained model; confined to the allowed roots.
        storage_root: the storage root.
        user_id: used to compute the completed-output root.

    Returns:
        Absolute path of the validated model directory.

    Raises:
        ValueError: the path is outside the allowed roots (traversal) or does
            not exist, or both arguments are empty. The caller maps this to 404.
    """
    storage = Path(storage_root).resolve()
    roots = [r.resolve() for r in allowed_model_roots(storage, user_id)]

    if model_path:
        cand = Path(model_path).resolve()
        if not any(_is_within(cand, r) for r in roots):
            raise ValueError(f"Model path is not allowed: {model_path}")
        if not cand.is_dir():
            raise ValueError(f"Model path does not exist: {model_path}")
        return str(cand)

    if not model_name:
        raise ValueError("Either model_name or model_path is required.")
    models_root = (storage / "models").resolve()
    base = (models_root / model_name).resolve()
    # Confine model_name too, so '..' cannot escape the models root.
    if not _is_within(base, models_root):
        raise ValueError(f"Model name is not allowed: {model_name}")
    if not base.is_dir():
        raise ValueError(f"Model '{model_name}' not found")
    return str(base)
