"""Two things here are machine contracts and are never translated."""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from server.core.lang import EN

SUGGESTED_CONFIRM_WORDS: Dict[str, Tuple[str, ...]] = {
    EN: ("yes", "go"),
}

#: Tool fragments that must survive into every translation.
TOOL_TOKENS: FrozenSet[str] = frozenset({"start_training_job"})

TEXTS: Dict[str, Dict[str, str]] = {
    EN: {
        "confirmed_below": (
            "The user confirmed the parameters below. "
            "Call start_training_job immediately, with no further analysis or questions."
        ),
        "confirmed_recommended": (
            "The user confirmed the recommended parameters. "
            "Call start_training_job immediately, with no further analysis or questions."
        ),
        "confirmed_below_recommended": (
            "The user confirmed the recommended parameters below. "
            "Call start_training_job immediately, with no further analysis or questions."
        ),
        "exact_params_label": "Exact parameters: ",
        "use_recommendation_as_is": (
            "Use the method, epochs, batch_size, learning_rate and base_model exactly as "
            "stated in the recommendation above."
        ),
        "recommendation_body": "Recommendation:",
        "goal_context": "\n[Pipeline goal] {value}",
        "confirm_hint": "To proceed, reply '{first}' or '{second}'.\n",
        "dataset_path": "Data path: {path}",
    },
}


def texts(lang: str | None = None) -> Dict[str, str]:
    """Phrase dictionary for the execution language."""
    return TEXTS[EN]


def confirm_hint(lang: str | None = None) -> str:
    """One line telling the user what to type.

    The suggested words come from :data:`SUGGESTED_CONFIRM_WORDS`, so only
    words the matcher accepts are ever suggested.
    """
    key = EN
    first, second = SUGGESTED_CONFIRM_WORDS[key]
    return texts(lang)["confirm_hint"].format(first=first, second=second)
