"""Deterministic folder matching for chat messages.

The agent may miss a folder name (or the LLM call may fail); when the user
wrote the folder name verbatim we can still resolve it against the real
dataset list. This is a comparison against existing data, not a vocabulary.
"""

from __future__ import annotations

from typing import Optional, Sequence

#: Token budget for structured-decision LLM calls (JSON verdicts). Reasoning
#: models spend tokens on hidden thinking first, so a small budget yields an
#: empty answer.
DECISION_MAX_TOKENS = 1024


def find_folder_in_text(text: str, folders: Sequence[str]) -> Optional[str]:
    """Return the one existing folder name that appears verbatim in ``text``.

    Several matches are ambiguous (None), unless the longest match contains
    all the others (prefix relation), in which case the longest wins.
    """
    if not text:
        return None
    hits = [f for f in folders if f and f in text]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        longest = max(hits, key=len)
        if all(h in longest for h in hits):
            return longest
    return None
