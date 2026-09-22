"""Language pack for job progress messages."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

Builder = Callable[..., str]


# ---------------------------------------------------------------------------
# English builders. Word order differs by language, so formats are not shared.
# ---------------------------------------------------------------------------

_EN: Dict[str, Builder] = {
    "twist_running": lambda method: f"Running TWIST ({method})",
    "download_done": lambda name: f"'{name}' downloaded",
    "kbd_load_model": lambda model: f"Loading the {model} model",
    "kbd_probe_start": lambda total: f"Starting probes ({total} total)",
    "done": lambda: "Done",
    "user_stopped": lambda: "Stopped by the user",
}

# Name list, so the guard and the callers read the same thing.
NAMES = tuple(sorted(_EN))


def pack(lang: Optional[str]) -> Dict[str, Builder]:
    """Builder pack for a language."""
    return _EN


def progress(name: str, *args: Any, lang: Optional[str] = None) -> str:
    """Build a progress message.

    **Never raises.** One progress line must not kill a training run. On
    failure it returns the name itself (visible on screen) and logs a warning.
    """
    builder = pack(lang).get(name)
    if builder is None:
        logger.warning("[i18n] unknown progress message name=%r — showing the name as is", name)
        return name
    try:
        return builder(*args)
    except TypeError:
        # An argument count mismatch is a caller bug. Returning an empty string
        # would hide it from the screen and make the cause hard to find.
        logger.warning("[i18n] progress(%r) argument mismatch args=%r — showing the name as is", name, args)
        return name
