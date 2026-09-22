"""Pick the training target a message refers to — the dataset and the model.

A name must be matched against the real list, never guessed from the shape of
the message. Taking the first token turns "Go ahead and train on ..." into a
dataset called ``Go``, and the LLM then reasons from its zero samples.

Datasets are named in full, so they match on the family prefixes derived from
the real list. Models are referred to by size or family ("the 0.5B", "Qwen"),
so they match on handles derived from each installed name.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from server.core.auto_intent import find_folder_in_text

#: A quoted name — the user marked it explicitly.
_QUOTED = re.compile(r'"([^"]+)"')


def dataset_names(datasets: Sequence[Mapping[str, Any]]) -> list:
    """``{"name": ...}`` records to a list of names. Empty names are dropped."""
    out = []
    for item in datasets or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            out.append(name)
    return out


def select_training_dataset(
    message: str,
    datasets: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """Pick a dataset name from the message that actually exists, else ``None``.

    Order:
      1. A quoted name, but only when it is in the real list. The old version
         never checked, so any string became a dataset name.
      2. A real name in the body (`find_folder_in_text`); several is ambiguous.

    The first token is never guessed at — that is what made `Go` and `Start`
    into dataset names.
    """
    names = dataset_names(datasets)
    if not names or not message:
        return None

    quoted = _QUOTED.search(message)
    if quoted:
        candidate = quoted.group(1).strip()
        if candidate in names:
            return candidate
        # A quoted value that does not exist is dropped; fall through to the body.

    return find_folder_in_text(message, names)


STATUS_FOUND = "found"  #: exactly one real name found
STATUS_ABSENT = "absent"  #: a dataset-shaped name was given but is not in the list
STATUS_AMBIGUOUS = "ambiguous"  #: two or more real names mentioned
STATUS_NONE = "none"  #: no dataset mentioned — the caller falls back to its default

#: Punctuation stripped from both ends of a token (never part of a name).
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>"
#: Dataset names are ASCII, so any non-ASCII run marks the end of a name.
_NON_ASCII_TAIL = re.compile(r"[^\x00-\x7f].*$")
#: A name family is the prefix up to the first digit (separator included), or
#: up to the first separator when there is no digit.
_FAMILY_BEFORE_DIGIT = re.compile(r"^([^\d]*?[_-])\d")
_FAMILY_FIRST_SEP = re.compile(r"^([^_-]+[_-])")


class TargetResolution:
    """Result of :func:`resolve_training_dataset`."""

    __slots__ = ("status", "name", "mentioned")

    def __init__(self, status: str, name: Optional[str] = None, mentioned: Optional[list] = None):
        self.status = status
        self.name = name
        self.mentioned = list(mentioned or [])

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"TargetResolution({self.status!r}, name={self.name!r}, mentioned={self.mentioned!r})"


def name_families(names: Sequence[str]) -> list:
    """Derive family prefixes from the real names, de-duplicated, longest first.

    ``dataset_20260908_102036`` → ``dataset_`` / ``agent-gen-1789617384`` → ``agent-gen-`` /
    ``dataset_x`` gives ``dataset_``. A name with no separator (``mydata``) has no family.
    """
    out = []
    for n in names:
        m = _FAMILY_BEFORE_DIGIT.match(n) or _FAMILY_FIRST_SEP.match(n)
        if m and m.group(1) not in out and m.group(1) != n:
            out.append(m.group(1))
    return sorted(out, key=len, reverse=True)


def _tokens(message: str) -> list:
    out = []
    for raw in message.split():
        tok = _NON_ASCII_TAIL.sub("", raw).strip(_EDGE_PUNCT)
        if tok:
            out.append(tok)
    return out


def missing_mentions(message: str, names: Sequence[str]) -> list:
    """Tokens that look like a dataset but are not in the list, in order.

    Families are derived from the real list, so an empty list matches nothing.
    That case means "there are no datasets at all", which is not this decision.
    """
    families = name_families(names)
    if not families or not message:
        return []
    known = set(names)
    out = []
    for tok in _tokens(message):
        if tok in known or tok in out:
            continue
        if any(tok.startswith(f) and len(tok) > len(f) for f in families):
            out.append(tok)
    return out


def resolve_training_dataset(
    message: str,
    datasets: Sequence[Mapping[str, Any]],
) -> TargetResolution:
    """Resolve the training target into one of four statuses.

    "found" outranks "absent": one exact real name is enough, and an odd token
    beside it must not block a valid request.

    """
    names = dataset_names(datasets)
    if not names or not message:
        return TargetResolution(STATUS_NONE)

    picked = select_training_dataset(message, datasets)
    if picked:
        return TargetResolution(STATUS_FOUND, picked)

    present = [n for n in names if n in message]
    if len(present) > 1:
        return TargetResolution(STATUS_AMBIGUOUS, None, present)

    missing = missing_mentions(message, names)
    if missing:
        return TargetResolution(STATUS_ABSENT, None, missing)

    return TargetResolution(STATUS_NONE)


# ---------------------------------------------------------------------------
# The base model
# ---------------------------------------------------------------------------

#: A size in a model name or a message: 0.5B, 135M, 7 b.
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])\b", re.I)

#: Word-ish pieces of a model name. Short ones ("hf", "it") match too loosely.
_MIN_HANDLE = 4


def model_handles(name: str) -> set:
    """Ways a person might name this model, lowercased.

    ``Qwen--Qwen2.5-0.5B-Instruct`` gives the full name, ``qwen2.5``, ``qwen``
    and the size ``0.5b``. Generic words like "instruct" are not handles: they
    appear in every name and would match everything.
    """
    low = name.lower()
    handles = {low}
    for num, unit in _SIZE.findall(low):
        handles.add(f"{num}{unit}")
    # Not on ".", so a version stays whole: qwen2.5, llama3.1.
    for piece in re.split(r"[-_/]+", low):
        if len(piece) >= _MIN_HANDLE and piece not in _GENERIC_PIECES and not _SIZE.fullmatch(piece):
            handles.add(piece)
    return handles


#: Pieces shared by most repo ids, so they identify nothing.
_GENERIC_PIECES = {"instruct", "chat", "base", "model", "huggingfacetb"}


def resolve_requested_model(message: str, model_names: Sequence[str]) -> Optional[str]:
    """The installed model the message asks for, or ``None``.

    ``None`` when nothing is named and when two are, because a guess here is
    what makes a stale model from earlier in the conversation win.
    """
    if not message or not model_names:
        return None
    low = " ".join(message.lower().split())
    # Normalise "0.5 B" and "135 M" so a spaced size still matches.
    low = _SIZE.sub(lambda m: f"{m.group(1)}{m.group(2)}", low)

    matched = [name for name in model_names if any(h in low for h in model_handles(name))]
    return matched[0] if len(matched) == 1 else None
