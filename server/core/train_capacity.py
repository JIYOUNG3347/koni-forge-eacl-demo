"""How large a model fits in the free accelerator memory.

Rough estimates that share one formula with the training UI:
  Full FT  ~ free_gb / 17      (weights + grads + Adam states + master copy)
  LoRA     ~ (free_gb - 5) / 2 (frozen base at 2 bytes/param + activations)

Batch size, sequence length and gradient checkpointing move the real limit, so
these are guidance, not a guarantee.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

_FFT_GB_PER_B = 17.0
_LORA_OVERHEAD_GB = 5.0

CAPACITY_METHODS = ("lora", "sft")


def max_train_params_b(method: str, free_gb: float) -> float:
    """Largest trainable model in billions of parameters. Approximate, floor 0."""
    if (method or "").lower() == "lora":
        return max(0.0, (free_gb - _LORA_OVERHEAD_GB) / 2.0)
    return max(0.0, free_gb / _FFT_GB_PER_B)


def capacity_label(method: str) -> str:
    """Display name. The backend owns it so the front-end has no hardcoded list."""
    return "LoRA" if (method or "").strip().lower() == "lora" else "FFT"


def build_train_capacity(gpu_free_gb: Iterable[float]) -> Dict[str, object]:
    """Capacity per method, sent in one payload so the UI needs no round trips."""
    gpus = sorted((float(g) for g in gpu_free_gb), reverse=True)
    options: List[Dict[str, object]] = []
    if not gpus:
        return {"gpu_free_gb": [], "options": options}
    for method in CAPACITY_METHODS:
        options.append(
            {
                "method": method,
                "label": capacity_label(method),
                "max_params_b": round(max_train_params_b(method, gpus[0]), 1),
                "basis_gb": round(gpus[0], 1),
            }
        )
    return {"gpu_free_gb": [round(g, 1) for g in gpus], "options": options}
