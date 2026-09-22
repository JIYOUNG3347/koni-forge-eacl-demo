"""Recognise confirmation, skip, rejection and question intents.

Matching is phrase-based after normalisation, not exact equality: people write
``Yes, please`` and ``Go ahead``, so punctuation and fillers (``please``,
``just``) are stripped before comparing against the phrase sets.

A question mark cancels the confirmation and rejection tests — reading
``Should I continue?`` as approval would start training when the user only
asked.

``Option 1`` style answers are a machine contract and shared across languages.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, Pattern

from server.core.lang import EN

#: Full-width question mark, which some IMEs produce.
QUESTION_MARK_FULLWIDTH = "\uff1f"

#: Option answers. The UI renders "Option 1", so this is a machine contract.
OPTION_RE: Pattern[str] = re.compile(r"^option\s*\d")


_EN: Dict[str, FrozenSet[str]] = {
    "confirm_exact": frozenset(
        {
            "yes",
            "y",
            "yeah",
            "yep",
            "yup",
            "ok",
            "okay",
            "k",
            "go",
            "sure",
            "proceed",
            "proceeding",
            "continuing",
            "confirm",
            "confirmed",
            "continue",
            "start",
            "run",
            "approve",
            "approved",
            "accept",
            "agreed",
            "affirmative",
        }
    ),
    "confirm_phrases": frozenset(
        {
            "go ahead",
            "proceeding with the edited parameters",
            "go on",
            "keep going",
            "carry on",
            "move on",
            "next step",
            "do it",
            "run it",
            "start it",
            "send it",
            "ship it",
            "sounds good",
            "looks good",
            "that works",
            "lets go",
            "let us go",
            "go for it",
            "make it so",
            "all good",
            "no objection",
        }
    ),
    "skip_exact": frozenset(
        {
            "no",
            "n",
            "nope",
            "nah",
            "skip",
            "skipping",
            "pass",
            "later",
            "stop",
            "halt",
            "cancel",
            "cancelled",
            "abort",
            "decline",
            "declined",
            "reject",
        }
    ),
    "skip_substrings": frozenset(
        {
            "skip it",
            "skip this",
            "skip that",
            "skip the",
            "not now",
            "not yet",
            "no thanks",
            "no thank you",
            "maybe later",
            "do it later",
            "some other time",
            "leave it",
            "never mind",
            "nevermind",
        }
    ),
    "rejection_markers": frozenset(
        {
            "no thanks",
            "no thank you",
            "not now",
            "not yet",
            "maybe later",
            "later",
            "decline",
            "declined",
            "reject",
            "cancel",
            "no need",
            "do it later",
            "some other time",
            "never mind",
            "nevermind",
            "pass",
        }
    ),
    "question_markers": frozenset(
        {
            "what",
            "why",
            "how",
            "which",
            "when",
            "who",
            "where",
            "explain",
            "tell me",
            "describe",
            "show me",
            "clarify",
            "can you",
            "could you",
            "is it",
            "does it",
            "do i",
            "should i",
        }
    ),
}

PACKS: Dict[str, Dict[str, FrozenSet[str]]] = {EN: _EN}

#: Fillers removed during normalisation: politeness and emphasis that carry no meaning.
_EN_FILLERS: FrozenSet[str] = frozenset({"please", "pls", "just", "then", "now", "thanks", "thank", "you"})


def _has_question_mark(message: str) -> bool:
    return "?" in message or QUESTION_MARK_FULLWIDTH in message


def _pack(lang: object) -> Dict[str, FrozenSet[str]]:
    """Phrase pack for a language."""
    return _EN


def normalize_en(message: str) -> str:
    """Normalise a reply for phrase comparison.

    Strips punctuation, lowercases and collapses whitespace. An apostrophe is
    absorbed (``let's`` becomes ``lets``), which is the form the phrase sets use.
    """
    text = message.strip().lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_fillers(text: str) -> str:
    """The text with fillers removed. All-filler input is returned unchanged."""
    kept = [w for w in text.split() if w not in _EN_FILLERS]
    return " ".join(kept) if kept else text


def _en_is_confirmation(message: str) -> bool:
    if _has_question_mark(message):  # a question is not an approval
        return False
    text = normalize_en(message)
    if OPTION_RE.match(text):
        return True
    core = _strip_fillers(text)
    return core in _EN["confirm_exact"] or core in _EN["confirm_phrases"]


def _en_is_skip(message: str) -> bool:
    if _has_question_mark(message):
        return False
    text = normalize_en(message)
    core = _strip_fillers(text)
    if core in _EN["skip_exact"]:
        return True
    return any(s in text for s in _EN["skip_substrings"])


def _en_is_rejection(message: str) -> bool:
    if _has_question_mark(message):
        return False
    text = normalize_en(message)
    if _strip_fillers(text) in _EN["skip_exact"]:
        return True
    return any(marker in text for marker in _EN["rejection_markers"])


def is_confirmation(message: str, lang: object = None) -> bool:
    """Whether the message confirms."""
    return _en_is_confirmation(message)


def is_skip(message: str, lang: object = None) -> bool:
    """Whether the message asks to skip."""
    return _en_is_skip(message)


def is_rejection(message: str, lang: object = None) -> bool:
    """Whether the message declines a proposal. Distinct from skip."""
    return _en_is_rejection(message)


def is_question(message: str, lang: object = None) -> bool:
    """Whether the message is a question, so a clarification at a pending stage
    does not wake the specialist.

    A false positive only means "do not resume", which is safe; a miss runs the
    wrong thing. The markers are therefore generous.
    """
    msg = message.strip().lower()
    if "?" in msg or QUESTION_MARK_FULLWIDTH in msg:
        return True
    pack = _pack(lang)
    text = normalize_en(msg)
    return any(marker in text for marker in pack["question_markers"])
