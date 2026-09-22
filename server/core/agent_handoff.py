"""Handoff messages between pipeline stages (pure module).

A handoff message is not decoration — it is the *instruction* the next agent
receives, and it may carry a tool call the agent must make. Tool-call
fragments in these strings are machine contracts and must survive any
rewording (``TOOL_TOKENS`` is checked by tests).
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet

from server.core.lang import EN

#: Tool-call fragments that must stay intact in the messages below.
TOOL_TOKENS: FrozenSet[str] = frozenset({"start_training_job"})


def _upload_docs_next(folder: str) -> str:
    return (
        f"Documents were uploaded to the '{folder}' folder. RAG indexing has started in the "
        f"background — check its progress and, once it finishes, prepare the KBD "
        f"knowledge-boundary analysis."
    )


def _upload_qa_next(folder: str) -> str:
    return (
        f"A QA dataset (JSON/JSONL) was uploaded to the '{folder}' folder and is ready for "
        f"training as-is. Check the available models and GPU status, then recommend a "
        f"training setup for this dataset."
    )


def _retrieval_next() -> str:
    return "RAG indexing is complete. Measure the base model's knowledge boundary using the indexed documents."


def _retrieval_announce() -> str:
    return "RAG indexing complete → start KBD analysis?"


def _tuning_next(task_id: str) -> str:
    return (
        f"Training job '{task_id}' is complete. Summarise the run for the user: final loss, "
        f"runtime, the output path, and anything notable in the training log."
    )


def _tuning_announce() -> str:
    return "Training complete."


def _skip_next() -> str:
    return (
        "Proceeding straight to training. Check the available models and GPU status, "
        "then recommend the best training setup."
    )


def _skip_announce() -> str:
    return "Start training right away?"


def _kbd_rag_next(coverage: object, model: str) -> str:
    _model_line = f"Model: {model} (the base model used for the KBD analysis).\n" if model else ""
    return (
        f"The KBD analysis found knowledge coverage of {coverage}%, which is high enough that "
        f"no additional training is required.\n"
        f"{_model_line}"
        "Retrieval alone is sufficient — the model can be used with RAG on the indexed documents."
    )


def _kbd_rag_announce(coverage: object) -> str:
    return f"KBD analysis complete — coverage {coverage}%, retrieval alone is sufficient (no training)."


def _kbd_tuning_next(model: str) -> str:
    _model_line = f"Base model: {model} (the same model used for the KBD analysis).\n" if model else ""
    return (
        f"The KBD analysis shows fine-tuning is needed, and a training dataset is available.\n"
        f"{_model_line}"
        f"Check this model and the GPU status, then start training."
    )


def _kbd_tuning_announce(coverage: object) -> str:
    return f"KBD analysis complete (coverage {coverage}%) → shall I start training?"


def _kbd_upload_next(coverage: object) -> str:
    return (
        f"The KBD analysis found knowledge coverage of {coverage}%, so fine-tuning is needed, "
        "but there is no training dataset yet. Ask the user to upload a QA dataset (JSON/JSONL) "
        "on the Upload page; training can start once it is registered."
    )


def _kbd_upload_announce(coverage: object) -> str:
    return f"KBD analysis complete (coverage {coverage}%) — fine-tuning is needed. Upload a QA dataset to continue."


_EN: Dict[str, Callable] = {
    "upload_docs_next": _upload_docs_next,
    "upload_qa_next": _upload_qa_next,
    "retrieval_next": _retrieval_next,
    "retrieval_announce": _retrieval_announce,
    "tuning_next": _tuning_next,
    "tuning_announce": _tuning_announce,
    "skip_next": _skip_next,
    "skip_announce": _skip_announce,
    "kbd_rag_next": _kbd_rag_next,
    "kbd_rag_announce": _kbd_rag_announce,
    "kbd_tuning_next": _kbd_tuning_next,
    "kbd_tuning_announce": _kbd_tuning_announce,
    "kbd_upload_next": _kbd_upload_next,
    "kbd_upload_announce": _kbd_upload_announce,
}

PACKS: Dict[str, Dict[str, Callable]] = {EN: _EN}


def pack(lang: str | None = None) -> Dict[str, Callable]:
    return _EN


def handoff(name: str, lang: str | None = None, /, *args, **kwargs) -> str:
    """Pick one message from the pack and render it."""
    return pack(lang)[name](*args, **kwargs)
