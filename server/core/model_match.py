"""Resolve a model name the user or LLM wrote to a real model folder.

Policy:
  1. Normalise (drop separators, case and dots) and look for an **exact** match.
  2. Otherwise a **suffix** match, for a name missing only its org prefix.
  3. Two or more candidates is **ambiguous**, so return ``None``.

``None`` means "could not decide", and callers must not pick one anyway: leave
it empty and let the user choose. Stopping beats silently training the wrong model.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# Separators that carry no meaning. Org, model and version are written many ways
# ("Qwen--Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5 7b instruct"),
# so all of them are stripped before comparing.
_STRIP = ("--", "/", "\\", "_", "-", " ", ".")


def normalize(name: str) -> str:
    """Normalise for comparison: ``"Qwen--Qwen3-4B"`` becomes ``"qwenqwen34b"``."""
    if not isinstance(name, str):
        return ""
    out = name.strip().lower()
    for token in _STRIP:
        out = out.replace(token, "")
    return out


def resolve_model_name(candidate: str, known: Iterable[str]) -> Optional[str]:
    """Resolve ``candidate`` to one of ``known``. ``None`` when undecidable.

    Args:
        candidate: the name as written ("Qwen3-4B", "Qwen/Qwen3-4B", ...)
        known: names of the model folders that actually exist

    Returns:
        The resolved folder name, or ``None`` when missing or ambiguous.

    A suffix match only covers a missing org prefix: "Qwen3-4B" matches
    "Qwen--Qwen3-4B" but not "Qwen--Qwen3-4B-Instruct", because anything longer
    is a different model.
    """
    target = normalize(candidate)
    if not target:
        return None
    pairs = [(normalize(k), k) for k in known if isinstance(k, str) and k.strip()]

    exact = [orig for norm, orig in pairs if norm == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # normalisation collision — do not pick arbitrarily

    tail = [orig for norm, orig in pairs if norm.endswith(target)]
    if len(tail) == 1:
        return tail[0]
    return None  # missing or ambiguous


def resolve_or_none(candidate: str, known: Iterable[str]) -> Optional[str]:
    """Pass an exact folder name through, otherwise :func:`resolve_model_name`."""
    known_list: List[str] = [k for k in known if isinstance(k, str)]
    if candidate in known_list:
        return candidate
    return resolve_model_name(candidate, known_list)


