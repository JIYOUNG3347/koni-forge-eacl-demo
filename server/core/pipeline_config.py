"""Load the settings for an auto pipeline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from server.core.lang import CONFIG_KEY as _LANG_CONFIG_KEY
from server.core.lang import resolve_lang

# Same defaults as dispatcher._get_pipeline_config. If the two diverge, the two
# paths run with different settings, which is the problem this module exists to stop.


def merge_pipeline_config(
    system_cfg: Optional[Mapping[str, Any]],
    user_cfg: Optional[Mapping[str, Any]],
) -> dict:
    """Merge system_config and user_config into the pipeline settings; user wins.

    Produces the **same keys and defaults** as ``dispatcher._get_pipeline_config``,
    so both paths run a pipeline with identical settings.
    """
    merged: dict = dict(system_cfg or {})
    merged.update(dict(user_cfg or {}))

    def _int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    return {
        "pipeline_goal": merged.get("pipeline_goal", ""),
        # An unset or unsupported value falls back in resolve_lang, with a warning.
        _LANG_CONFIG_KEY: resolve_lang(config=merged),
    }


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def load_pipeline_config(user_id: str, storage_root: Optional[Path] = None) -> dict:
    """Read the pipeline settings from disk and merge them.

    A read failure proceeds with the defaults: a missing config file must not
    """
    if storage_root is None:
        from server.core.config import STORAGE_ROOT as _root

        storage_root = Path(_root)
    system_cfg = _read_json(storage_root / "system_config.json")
    user_cfg = _read_json(storage_root / "outputs" / user_id / "user_config.json")
    return merge_pipeline_config(system_cfg, user_cfg)
