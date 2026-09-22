"""Measure a model's parameter count from its weight files, not its name.

Names lie: merged, re-trained and user-uploaded models often carry no size, or
the wrong one. The bytes on disk are always true::

    parameters ~= total weight bytes / bytes per parameter

The dtype comes from ``torch_dtype``/``dtype`` in ``config.json``, defaulting
to bf16 (2 bytes). When the estimate fails, callers fall back to the safe path.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

# Extensions counted as weights. Optimizer state (``optimizer.pt``) and
# tokenizers must be excluded, so :func:`sum_weight_bytes` checks names too.
WEIGHT_EXTS = (".safetensors", ".bin")

# Files that share an extension with weights but are not (resume state).
_NON_WEIGHT_NAMES = (
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pth",
    "training_args.bin",
)

# dtype name to bytes per parameter.
_DTYPE_BYTES = {
    "float32": 4.0,
    "float": 4.0,
    "fp32": 4.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "half": 2.0,
    "int8": 1.0,
    "float8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
}

DEFAULT_BYTES_PER_PARAM = 2.0  # bf16, the de facto standard for recent models


def is_weight_file(name: str) -> bool:
    """Whether this file is model weights. Optimizer and scheduler state is excluded."""
    lowered = name.lower()
    if lowered in _NON_WEIGHT_NAMES:
        return False
    return lowered.endswith(WEIGHT_EXTS)


def sum_weight_bytes(entries: Iterable[Tuple[str, int]]) -> int:
    """Sum the bytes of the weight files in ``(name, size)`` pairs."""
    total = 0
    for name, size in entries:
        if is_weight_file(name):
            try:
                total += max(0, int(size))
            except (TypeError, ValueError):
                continue
    return total


def bytes_per_param(config: Optional[dict]) -> float:
    """Bytes per parameter from ``config.json``. Unknown means bf16 (2.0)."""
    if not config:
        return DEFAULT_BYTES_PER_PARAM
    raw = config.get("torch_dtype") or config.get("dtype")
    if not isinstance(raw, str):
        return DEFAULT_BYTES_PER_PARAM
    return _DTYPE_BYTES.get(raw.strip().lower(), DEFAULT_BYTES_PER_PARAM)


def params_b_from_bytes(total_bytes: int, per_param: float = DEFAULT_BYTES_PER_PARAM) -> Optional[float]:
    """Total weight bytes to parameters in billions. ``None`` when undecidable.

    ``None`` means "unknown", and callers then take the safe path. Zero bytes
    and an unrecognised dtype both give ``None``.
    """
    try:
        total = int(total_bytes)
        pp = float(per_param)
    except (TypeError, ValueError):
        return None
    if total <= 0 or pp <= 0:
        return None
    return (total / pp) / 1e9


def estimate_params_b(entries: Iterable[Tuple[str, int]], config: Optional[dict] = None) -> Optional[float]:
    """``(name, size)`` pairs plus ``config.json`` to parameters in billions."""
    return params_b_from_bytes(sum_weight_bytes(entries), bytes_per_param(config))
