"""Estimate training memory before a run and clamp the batch to what fits.

The logit tensor dominates more often than people expect:
``batch x seq x vocab x 4B`` (fp32 upcast) is ~20GiB for a 4B model with a
152k vocabulary at batch 16 / seq 2048.

When the request does not fit, the plan lowers the per-device batch and raises
``gradient_accumulation`` to preserve the effective batch. Estimates are used
to shrink only, never to grow, so an estimation error can only make a run more
conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Approximation constants, assuming bf16 training.
# Full FT: weights 2 + grads 2 + Adam m/v 8 + fp32 master 4 = 16 bytes/param
_FFT_STATIC_BYTES_PER_PARAM = 16.0
# LoRA: frozen weights 2 + adapter and optimizer ~0.3 (an r=8-64 adapter is ~1%)
_LORA_STATIC_BYTES_PER_PARAM = 2.3
# Activations: bytes per token per hidden unit, times the layer count.
_ACT_BYTES_PER_TOKEN_HIDDEN_LAYER = 12.0
# With gradient checkpointing only boundary activations are kept, roughly 1/12.
_ACT_BYTES_PER_TOKEN_HIDDEN_LAYER_CKPT = 1.0
# Logits / cross-entropy: fp32 upcast 4 bytes, doubled for the shift copy.
_LOGIT_BYTES_PER_TOKEN_VOCAB = 8.0
# Headroom for the device context and allocator fragmentation.
_MISC_MB = 3072.0

# Only use this fraction of the free memory, to survive fragmentation.
DEFAULT_MARGIN = 0.9

_MB = 1024 * 1024


@dataclass(frozen=True)
class ModelShape:
    """The model dimensions the estimate needs."""

    params_b: float
    vocab_size: int
    hidden_size: int
    num_layers: int


def shape_from_config(config: dict, params_b: Optional[float] = None) -> Optional[ModelShape]:
    """Build a shape from a HuggingFace config dict. None if a key is missing."""
    try:
        vocab = int(config["vocab_size"])
        hidden = int(config["hidden_size"])
        layers = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError):
        return None
    if vocab <= 0 or hidden <= 0 or layers <= 0:
        return None
    if params_b is None:
        inter = int(config.get("intermediate_size") or hidden * 4)
        approx = vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * inter)
        params_b = approx / 1e9
    return ModelShape(params_b=float(params_b), vocab_size=vocab, hidden_size=hidden, num_layers=layers)


def _is_full_ft(method: str) -> bool:
    return (method or "").strip().lower() in ("sft", "fft", "full")


def estimate_train_vram_mb(
    shape: ModelShape,
    method: str,
    per_device_batch: int,
    seq_len: int,
    grad_checkpointing: bool = False,
) -> float:
    """Estimated training memory in MB. Approximate — used only to clamp.

    DPO adds a frozen reference model (+2 bytes/param) and doubles the token
    count for the chosen / rejected pair.
    """
    m = (method or "").strip().lower()
    params = shape.params_b * 1e9
    if _is_full_ft(m):
        static_bytes = params * _FFT_STATIC_BYTES_PER_PARAM
    else:
        static_bytes = params * _LORA_STATIC_BYTES_PER_PARAM
    tokens = max(0, per_device_batch) * max(0, seq_len)
    act_coef = _ACT_BYTES_PER_TOKEN_HIDDEN_LAYER_CKPT if grad_checkpointing else _ACT_BYTES_PER_TOKEN_HIDDEN_LAYER
    act_bytes = tokens * shape.hidden_size * shape.num_layers * act_coef
    logit_bytes = tokens * shape.vocab_size * _LOGIT_BYTES_PER_TOKEN_VOCAB
    return (static_bytes + act_bytes + logit_bytes) / _MB + _MISC_MB


def clamp_to_limits(
    batch: int,
    seq_len: int,
    max_batch: Optional[int],
    max_seq: Optional[int],
) -> "tuple[int, int, tuple[str, ...]]":
    """Apply the host.toml ``[model_limits]`` ceilings.

    Returns:
        ``(batch, seq_len, reasons)``. A ceiling of None or <= 0 is unlimited.
    """
    reasons: list[str] = []
    out_batch, out_seq = int(batch), int(seq_len)
    if max_batch and max_batch > 0 and out_batch > max_batch:
        reasons.append(f"host.toml max_train_batch_size={max_batch} applied ({out_batch} -> {max_batch})")
        out_batch = int(max_batch)
    if max_seq and max_seq > 0 and out_seq > max_seq:
        reasons.append(f"host.toml max_train_seq_length={max_seq} applied ({out_seq} -> {max_seq})")
        out_seq = int(max_seq)
    return out_batch, out_seq, tuple(reasons)


def safe_per_device_batch(
    shape: ModelShape,
    method: str,
    seq_len: int,
    free_mb: float,
    grad_checkpointing: bool = False,
    margin: float = DEFAULT_MARGIN,
) -> int:
    """Largest per-device batch that fits in ``free_mb``. 0 if even batch 1 does not."""
    budget = free_mb * margin

    def _est(batch: int, ckpt: bool = grad_checkpointing) -> float:
        return estimate_train_vram_mb(shape, method, batch, seq_len, ckpt)

    if _est(1) > budget:
        return 0
    per_batch_mb = _est(2) - _est(1)
    base_mb = _est(0)
    if per_batch_mb <= 0:
        return 1
    return max(1, int((budget - base_mb) // per_batch_mb))


@dataclass(frozen=True)
class BatchPlan:
    """Clamp result. ``per_device_batch == 0`` means it does not fit at all."""

    per_device_batch: int
    grad_accum: int
    grad_checkpointing: bool
    est_vram_mb: float
    adjusted: bool
    reasons: "tuple[str, ...]" = ()


def plan_batch(
    shape: ModelShape,
    method: str,
    seq_len: int,
    free_mb: float,
    requested_batch: int,
    requested_accum: int,
    target_effective: Optional[int] = None,
    allow_grad_ckpt: bool = True,
    margin: float = DEFAULT_MARGIN,
) -> BatchPlan:
    """Shrink the batch to fit, compensating with accumulation.

    Args:
        target_effective: effective batch to preserve. Defaults to
            ``requested_batch * requested_accum``.
    """
    requested_batch = max(1, int(requested_batch))
    requested_accum = max(1, int(requested_accum))
    target = max(1, int(target_effective) if target_effective else requested_batch * requested_accum)

    est = estimate_train_vram_mb(shape, method, requested_batch, seq_len, False)
    if est <= free_mb * margin:
        return BatchPlan(requested_batch, requested_accum, False, est, adjusted=False)

    reasons = [
        f"requested batch {requested_batch} (seq {seq_len}) needs ~{est / 1024:.1f}GB "
        f"> {free_mb / 1024:.1f}GB free x {margin:.0%}"
    ]

    fit = safe_per_device_batch(shape, method, seq_len, free_mb, False, margin)
    use_ckpt = False
    if fit < 1 and allow_grad_ckpt:
        fit = safe_per_device_batch(shape, method, seq_len, free_mb, True, margin)
        use_ckpt = fit >= 1
        if use_ckpt:
            reasons.append("enabled gradient_checkpointing to shrink activations")
    if fit < 1:
        reasons.append("batch 1 does not fit in the free memory")
        return BatchPlan(0, requested_accum, use_ckpt, est, adjusted=True, reasons=tuple(reasons))

    new_batch = min(requested_batch, fit)
    new_accum = max(requested_accum, math.ceil(target / new_batch))
    reasons.append(
        f"per_device_batch {requested_batch} -> {new_batch}, grad_accum {requested_accum} -> {new_accum} "
        f"(effective batch {target} preserved)"
    )
    final_est = estimate_train_vram_mb(shape, method, new_batch, seq_len, use_ckpt)
    return BatchPlan(new_batch, new_accum, use_ckpt, final_est, adjusted=True, reasons=tuple(reasons))
