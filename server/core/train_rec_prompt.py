"""Assemble the training-recommendation prompt from the parser's own labels.

The format block is built from :data:`rec_params.LABELS`, so the prompt and the
parser read the same constants and cannot drift apart. Machine contracts are
never translated: `FFT` and `LoRA` are what ``_parse_*`` matches, and `KBD`,
`GPU`, `VRAM`, model names, dataset names and numbers are proper nouns.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from server.core.lang import EN
from server.core.rec_params import LABELS

#: Labels that appear on the card but are **not parsed**. Mixing them into
CARD_ONLY_LABELS: Dict[str, Dict[str, str]] = {
    EN: {
        "dataset": "Dataset",
    },
}

TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        "heading": "#### Recommended training settings",
        "unknown": "not available",
        "none": "none",
        "intro": "You are an LLM fine-tuning expert. Read the information below and recommend the best training parameters.",
        "models_line": "Available models: {models}",
        "requested_model_line": "Model explicitly requested by the user: {model} — **you must recommend this model as the base model**.\n",
        "gpu_line": "GPU: {gpu}",
        "safe_batch_line": "Safe batch ceiling (VRAM formula, at seq {seq}): {batch} — recommend a batch size at or below this value.\n",
        "kbd_model_line": "Model used for the KBD analysis: {model}",
        "kbd_verdict_line": "KBD verdict: {verdict} (knowledge coverage {coverage})",
        "dataset_line": "Training dataset: {name} ({count} samples)",
        "no_dataset_line": (
            "Training dataset: none available. This edition has no QA generation, so a dataset "
            "arrives only by upload. Do not invent a dataset name and do not propose starting "
            "training — tell the user to upload a QA dataset (JSON or JSONL) on the data page."
        ),
        "method_guide_line": "Method selection guide: {hint}",
        "format_intro": "Answer in exactly this markdown format:",
        "closing": "Keep each item on one line, with only a short reason in parentheses. ",
        "slot_model": "model name (short reason)",
        "slot_method": "FFT or LoRA (state the KBD verdict/coverage basis)",
        "slot_dataset": "dataset name (sample count)",
        "slot_epochs": "number (short reason)",
        "slot_batch": "number (consider GPU VRAM)",
        "slot_lr": "number (short reason)",
        "hint_ft": "The KBD verdict is '{verdict}' (coverage {coverage}), so recommend {method}.",
        "hint_ft_full": "Full Fine-Tuning (FFT)",
        "hint_ft_either": "FFT or LoRA — based on the dataset size",
        "hint_hybrid": "The KBD verdict is 'Hybrid' (coverage {coverage}), so recommend lightweight LoRA.",
        "hint_other": "Choose an appropriate method based on the KBD verdict '{verdict}' / coverage {coverage}.",
        "card_dataset_value": "{name} ({count})",
    },
}

#: Row order of the card and the format block. The parser finds rows by label,
#: so order is free, but both languages share it so users see one structure.
ROWS = (
    "model",
    "method",
    "dataset",
    "epochs",
    "batch",
    "lr",
)


def texts(lang: Optional[str] = None) -> Dict[str, str]:
    return TEXTS[EN]


def labels(lang: Optional[str] = None) -> Dict[str, str]:
    """Label for one card row. Anything parsed comes straight from `LABELS`."""
    key = EN
    return {**LABELS[key], **CARD_ONLY_LABELS[key]}


def format_block(lang: Optional[str] = None) -> str:
    """Markdown format instructions for the LLM, built from `LABELS`."""
    t, lb = texts(lang), labels(lang)
    lines = [t["heading"]] + [f"- **{lb[row]}**: {t[f'slot_{row}']}" for row in ROWS]
    return "\n".join(lines)


def build_prompt(
    ctx: Mapping[str, Any],
    *,
    lang: Optional[str] = None,
    low_coverage_pct: float,
    default_max_seq_len: int,
) -> str:
    """The full training recommendation prompt."""
    t = texts(lang)
    models_str = ", ".join([f"{m['name']}{m['size']}" for m in ctx["base_models"]]) or t["none"]
    verdict = ctx.get("kbd_recommendation") or t["unknown"]
    coverage_raw = ctx.get("kbd_coverage_pct")
    coverage = f"{coverage_raw}%" if coverage_raw is not None else t["unknown"]

    try:
        cov_num = float(coverage_raw) if coverage_raw is not None else 100.0
    except (TypeError, ValueError):
        cov_num = 100.0

    lowered = verdict.lower()
    if "fine-tuning" in lowered and "hybrid" not in lowered:
        hint = t["hint_ft"].format(
            verdict=verdict,
            coverage=coverage,
            method=t["hint_ft_full"] if cov_num < low_coverage_pct else t["hint_ft_either"],
        )
    elif "hybrid" in lowered:
        hint = t["hint_hybrid"].format(coverage=coverage)
    else:
        hint = t["hint_other"].format(verdict=verdict, coverage=coverage)

    requested = ctx.get("requested_model")
    req_line = t["requested_model_line"].format(model=requested) if requested else ""
    safe_line = (
        t["safe_batch_line"].format(seq=default_max_seq_len, batch=ctx["safe_batch"]) if ctx.get("safe_batch") else ""
    )

    return (
        f"{t['intro']}\n\n"
        f"{t['models_line'].format(models=models_str)}\n"
        f"{req_line}"
        f"{t['gpu_line'].format(gpu=ctx['gpu_info'] or t['unknown'])}\n"
        f"{safe_line}"
        f"{t['kbd_model_line'].format(model=ctx['kbd_model'] or t['none'])}\n"
        f"{t['kbd_verdict_line'].format(verdict=verdict, coverage=coverage)}\n"
        f"{_dataset_line(t, ctx)}\n\n"
        f"{t['method_guide_line'].format(hint=hint)}\n\n"
        f"{t['format_intro']}\n\n"
        f"{format_block(lang)}\n\n"
        f"{t['closing']}Write in English."
    )


def _dataset_line(t: Mapping[str, str], ctx: Mapping[str, Any]) -> str:
    """Dataset row, or the instruction to ask for an upload when there is none."""
    if not str(ctx.get("dataset_name") or "").strip():
        return t["no_dataset_line"]
    return t["dataset_line"].format(name=ctx["dataset_name"], count=ctx["dataset_count"])


def fallback_card(ctx: Mapping[str, Any], lang: Optional[str] = None) -> str:
    """Rule-based recommendation card, used when no LLM is available.

    This card goes through the same parser, so its labels must come from
    `LABELS`; otherwise a fallback run silently produces default parameters.
    """
    t, lb = texts(lang), labels(lang)
    # The internal values "sft" and "lora" are the backend schema; the card
    # writes "FFT" and "LoRA", which is what the parser reads.
    method_label = "FFT" if str(ctx.get("rec_method", "")).lower() == "sft" else "LoRA"
    values = {
        "model": str(ctx["rec_model"]),
        "method": method_label,
        "dataset": (
            t["card_dataset_value"].format(name=ctx["dataset_name"], count=ctx["dataset_count"])
            if str(ctx.get("dataset_name") or "").strip()
            else t["none"]
        ),
        "epochs": str(ctx["rec_epochs"]),
        "batch": str(ctx["rec_batch"]),
        "lr": str(ctx["rec_lr"]),
    }
    lines = [t["heading"]] + [f"- **{lb[row]}**: {values[row]}" for row in ROWS]
    return "\n".join(lines)


__all__ = [
    "CARD_ONLY_LABELS",
    "ROWS",
    "TEXTS",
    "build_prompt",
    "fallback_card",
    "format_block",
    "labels",
    "texts",
]
