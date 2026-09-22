"""Language pack for the error messages users see."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

from server.core.lang import EN, normalize, resolve_lang

logger = logging.getLogger(__name__)

HEADER = "X-KONI-Lang"

Builder = Callable[..., str]


# Each language has its own builders; word order differs, so formats are not shared.

_EN: Dict[str, Builder] = {
    "model_load_failed": lambda e: f"Could not load the model: {e}",
    "job_name_exists": lambda name: f"A job named '{name}' already exists.",
    "dataset_not_found": lambda name: f"Dataset not found: {name}",
}

#: Name list, so the guard and the callers read the same thing.
NAMES = tuple(sorted(_EN))


def display_lang(
    header: Optional[str] = None,
    config: Optional[dict] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Language to use for an error message.

    Order: the display language from the header, then the execution language
    from the config or environment. An error is shown to a person, so the
    language they are currently looking at wins.

    An unknown value is ignored and falls through. The header comes from the
    client and cannot be trusted, and raising here would turn building an error
    response into a 500.
    """
    if header:
        normalized = normalize(header)
        if normalized == EN:
            return normalized
        logger.warning("[i18n] unknown %s header value %r — falling back to the execution language", HEADER, header)
    return resolve_lang(config=config, env=env)


def lang_from_request(request: Any = None, user_id: Optional[str] = None) -> str:
    """Display language taken from the request.

    A failure reading the header or config is ignored: raising here would turn
    building an error response into a 500, stacking one cause on another.
    """
    header = None
    try:
        header = request.headers.get(HEADER) if request is not None else None
    except Exception:  # noqa: BLE001
        pass

    config = None
    if user_id:
        try:
            from server.core.pipeline_config import load_pipeline_config

            config = load_pipeline_config(user_id)
        except Exception:  # noqa: BLE001
            config = None
    return display_lang(header, config)


def pack(lang: Optional[str]) -> Dict[str, Builder]:
    """Builder pack for a language."""
    return _EN


def error(name: str, *args: Any, lang: Optional[str] = None) -> str:
    """Build an error message. **Never raises.**

    An error while building an error stacks one cause on another. It returns
    the name itself, which is visible on screen, and logs a warning.
    """
    builder = pack(lang).get(name)
    if builder is None:
        logger.warning("[i18n] unknown error message name=%r — showing the name as is", name)
        return name
    try:
        return builder(*args)
    except TypeError:
        logger.warning("[i18n] error(%r) argument mismatch args=%r — showing the name as is", name, args)
        return name
