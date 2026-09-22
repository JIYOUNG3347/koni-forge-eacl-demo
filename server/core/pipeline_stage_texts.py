"""Language pack for pipeline stage messages.

The monitor view looks each message up in its dictionary, but these are
dynamic strings with values baked in, so they cannot be dictionary keys::

    KBD analysis complete — coverage 3% -> fine_tuning
    already indexed — 10 documents reused
    Orchestrator starting the pipeline. Dataset: dataset_20260908_083605
"""

from __future__ import annotations

from typing import Dict

from server.core.lang import EN

#: Stage messages. Values may contain ``str.format`` placeholders.
TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        "index_reuse_all": "Already indexed — reusing {reused} document(s)",
        "index_mixed": "{new} new / {reused} reused",
        "index_new": "Indexing {new} document(s)",
        "kbd_done": "KBD analysis complete — {detail}{recommendation}",
        "train_progress": "Training - step {step}/{total}",
        "train_loss_suffix": " - loss {loss}",
        "kbd_ready": (
            "Ready for KBD analysis - {collections} collection(s), {chunks} chunks indexed. "
            "The KBD Agent will measure the base model's knowledge boundary against these "
            "documents and recommend whether retrieval alone is enough or fine-tuning is needed."
        ),
        "qa_recommendation": (
            "For '{dataset}' ({chunks} chunks), {qa_per_file} Q&A pairs per file are recommended. "
            "The generated data can be used as an SFT training dataset."
        ),
    },
}


def texts(lang: object = None) -> Dict[str, str]:
    """Language pack."""
    return TEXTS[EN]


def stage_text(key: str, lang: object = None, **kwargs: object) -> str:
    """One stage message."""
    template = texts(lang).get(key)
    if template is None:
        return key
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template
