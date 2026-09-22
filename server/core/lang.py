"""Execution language for prompts, filters and judges.

This build ships English only. The resolver keeps its signature so callers
that pass explicit / session / config values keep working, but every path
resolves to ``EN``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger("core.lang")

EN = "en"
SUPPORTED = (EN,)

#: key in ``system_config.json`` / ``user_config.json``
CONFIG_KEY = "exec_lang"


def normalize(value: Any, *, source: str = "") -> Optional[str]:
    """Normalise a language code. Returns the code if supported, else ``None``."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in SUPPORTED:
        return text
    logger.warning(
        "[lang] unsupported language code %r%s — ignored (supported: %s)",
        value,
        f" ({source})" if source else "",
        ", ".join(SUPPORTED),
    )
    return None


def resolve_lang(
    explicit: Any = None,
    session: Any = None,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the execution language. Always ``EN`` in this build."""
    for value, source in ((explicit, "explicit"), (session, "session")):
        normalize(value, source=source)
    if config:
        normalize(config.get(CONFIG_KEY), source="config")
    return EN
