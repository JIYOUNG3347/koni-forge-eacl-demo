"""Detecting an execution confirmation, independent of output language.

``foundation.run`` lets a guarded tool run when the user replies with a
confirmation word *and* the previous assistant message was a confirmation
question. The phrase set for that second check lives here, and a static test
matches it against the real announcement strings, so wording and detector
cannot drift apart silently.
"""

from __future__ import annotations

from typing import Any

#: Phrases that mark an assistant message as asking for execution confirmation.
_CONFIRM_ASK: tuple = (
    "with these?",  # "... start training with these?" / "... generate data with these?"
    "shall i",  # "Shall I start training with these settings?"
    "proceed with",  # "Proceed with {tool_label}?"
    "would you like",
    "do you want",
    "say 'go'",
)

#: Words that count as a user confirmation.
CONFIRM_WORDS: frozenset = frozenset({"yes", "y", "ok", "okay", "go", "proceed", "start", "sure"})


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def is_confirmation_question(text: Any) -> bool:
    """Whether the previous assistant message asked for execution confirmation."""
    lowered = _text(text).lower()
    if not lowered:
        return False
    return any(kw in lowered for kw in _CONFIRM_ASK)


