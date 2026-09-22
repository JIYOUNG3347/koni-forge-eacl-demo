"""Parse training parameters out of the recommendation card.

The dispatcher tells the LLM a markdown format (``- **learning rate**: number``)
and then reads that same output back with a regex. Prompt and parser are
therefore contracted on the same labels, which is why they share
:data:`LABELS`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Mapping

from server.core.lang import EN, normalize

logger = logging.getLogger("core.rec_params")

#: Card row labels — the shared contract between the prompt and the parser.
#: These must match the strings in the prompt exactly (pinned by a test).
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "model": "Base model",
        "method": "Method",
        "epochs": "Epochs",
        "batch": "Batch size",
        "lr": "Learning rate",
    },
}


def _resolve_model(candidate: str, ctx: Mapping[str, Any], default: str) -> str:
    """Resolve the model name in the card to a real folder. Empty when undecidable."""
    from server.core.model_match import resolve_or_none

    if not candidate:
        return default  # the LLM named no model, keep the rule-based default
    known = [mm.get("name", "") for mm in ctx.get("base_models", [])]
    resolved = resolve_or_none(candidate, known)
    if resolved:
        return resolved
    logger.warning(
        f"[rec_params] could not resolve the recommended model: {candidate!r} "
        f"(available: {known}) — leaving it empty so the user chooses."
    )
    return ""


def _parse_en(rec_text: str, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse the English card.

    A full-width colon is accepted too, because local models sometimes mix
    full-width punctuation into their output.
    """
    lb = LABELS["en"]

    def _int(label: str, default: int) -> int:
        m = re.search(rf"{label}[^:：]*[:：]\s*(\d+)", rec_text, re.IGNORECASE)
        try:
            return int(m.group(1)) if m else default
        except (ValueError, IndexError):
            return default

    def _lr(default: str) -> str:
        m = re.search(rf"{lb['lr']}[^:：]*[:：]\s*([\d.]+(?:e[+-]?\d+)?)", rec_text, re.IGNORECASE)
        if m:
            try:
                float(m.group(1))
                return m.group(1)
            except ValueError:
                pass
        return default

    def _method(default: str) -> str:
        m = re.search(rf"{lb['method']}[^:：]*[:：][^*\n]*(FFT|LoRA)", rec_text, re.IGNORECASE)
        return ("sft" if m and m.group(1).upper() == "FFT" else "lora") if m else default

    def _model(default: str) -> str:
        m = re.search(rf"{lb['model']}[^:：]*[:：]\s*([^\s(（\n]+)", rec_text, re.IGNORECASE)
        return _resolve_model(m.group(1).strip() if m else "", ctx, default)

    return {
        "rec_method": _method(str(ctx.get("rec_method", "lora"))),
        "rec_model": _model(str(ctx.get("rec_model", ""))),
        "rec_epochs": _int(lb["epochs"], int(ctx.get("rec_epochs", 3))),
        "rec_batch": _int(lb["batch"], int(ctx.get("rec_batch", 4))),
        "rec_lr": _lr(str(ctx.get("rec_lr", "2e-5"))),
    }


def parse_rec_params(rec_text: str, ctx: Mapping[str, Any], lang: str | None = None) -> Dict[str, Any]:
    """Extract training parameters from the recommendation text.

    Falls back to the ctx defaults when parsing fails.
    """
    if normalize(lang) == EN:
        return _parse_en(rec_text, ctx)
    return _parse_en(rec_text, ctx)
