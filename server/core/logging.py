"""
KONI-Forge Logging — TimedRotatingFileHandler with auto-detect writable path

Features:
- Daily log rotation (configurable interval)
- Configurable backup count
- sys_log() convenience function
- fmt_ts() timestamp formatter
- Auto-detect writable log path
"""

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from server.core.config import (
    LOG_BACKUP_COUNT,
    LOG_ROTATION_INTERVAL,
    LOGS_DIR,
    STORAGE_ROOT,
)

# ═══════════════════════════════════════════════════════════
# Log path detection
# ═══════════════════════════════════════════════════════════


def _detect_log_path() -> Path:
    """Auto-detect a writable log path."""
    candidates = [
        LOGS_DIR,
        STORAGE_ROOT / "logs",
        Path("./logs"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Test write
            test_file = candidate / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            return candidate
        except (OSError, PermissionError):
            continue
    # Fallback: current directory
    return Path(".")


LOG_PATH: Path = _detect_log_path()

# ═══════════════════════════════════════════════════════════
# Logger setup
# ═══════════════════════════════════════════════════════════

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Map interval strings to handler parameters
_INTERVAL_MAP = {
    "S": ("S", 1),
    "M": ("M", 1),
    "H": ("H", 1),
    "D": ("D", 1),
    "midnight": ("midnight", 1),
}


def _create_file_handler(
    log_file: str = "koni_forge.log",
) -> TimedRotatingFileHandler:
    """Create a TimedRotatingFileHandler."""
    filepath = LOG_PATH / log_file
    when, interval = _INTERVAL_MAP.get(LOG_ROTATION_INTERVAL, ("D", 1))

    handler = TimedRotatingFileHandler(
        filename=str(filepath),
        when=when,
        interval=interval,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        utc=False,
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.suffix = "%Y-%m-%d"
    return handler


def get_logger(name: str = "koni_forge") -> logging.Logger:
    """
    Get a configured logger with file rotation and console output.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # File handler with rotation
    try:
        file_handler = _create_file_handler()
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass  # Graceful fallback to console only

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    return logger


# Module-level default logger
_logger = get_logger()

# Attach handlers to both logger trees — `agent.*` (specialists) and `agents.*`
# (dispatcher / foundation) — so child loggers propagate instead of being
# dropped by a handler-less root.
for _parent_name in ("agent", "agents"):
    get_logger(_parent_name)


# ═══════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════


def sys_log(message: str, level: str = "INFO") -> None:
    """
    Log a system message at the specified level.

    Args:
        message: Log message text
        level: Log level — "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    _logger.log(log_level, message)


def fmt_ts(dt: Optional[datetime] = None) -> str:
    """Format a datetime as a human-readable timestamp string.
    Uses current time if dt is None.
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(_DATE_FORMAT)
