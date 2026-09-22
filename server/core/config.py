"""Runtime configuration — host.toml for limits and paths, env vars for the rest.

Hardware limits, storage paths and concurrency come from host.toml
(:mod:`server.core.host_config`). Secrets and service URLs come from the
environment.
"""

import os
from pathlib import Path
from typing import List

from server.core.host_config import HOST_CONFIG, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
VERSION = os.getenv("KONI_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Storage — host.toml [paths] is the single source
# ---------------------------------------------------------------------------
_storage_base = Path(HOST_CONFIG.paths.storage_base).expanduser()
if not _storage_base.is_absolute():
    _storage_base = (PROJECT_ROOT / _storage_base).resolve()
STORAGE_ROOT: Path = _storage_base

# Modules that read these from the environment must see the resolved values.
os.environ["STORAGE_BASE_PATH"] = str(STORAGE_ROOT)

MODELS_DIR: Path = STORAGE_ROOT / "models"
CORPUS_DIR: Path = STORAGE_ROOT / "corpus"
DATASETS_DIR: Path = STORAGE_ROOT / "datasets"
CHECKPOINTS_DIR: Path = STORAGE_ROOT / "checkpoints"
OUTPUTS_DIR: Path = STORAGE_ROOT / "outputs"
CHROMA_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(STORAGE_ROOT / "chroma")))
os.environ["CHROMA_PERSIST_DIR"] = str(CHROMA_DIR)
LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", str(STORAGE_ROOT / "logs")))
TEMP_DIR: Path = STORAGE_ROOT / "temp"
USERS_DIR: Path = STORAGE_ROOT / "users"
USERS_JSON: Path = Path(os.getenv("USERS_JSON", str(STORAGE_ROOT / "users.json")))
SYSTEM_CONFIG_PATH: Path = STORAGE_ROOT / "system_config.json"
ACTIVITY_LOG_PATH: Path = STORAGE_ROOT / "activity_log.json"

for _d in [
    STORAGE_ROOT,
    MODELS_DIR,
    CORPUS_DIR,
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    CHROMA_DIR,
    LOGS_DIR,
    TEMP_DIR,
    USERS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
INTERNAL_API_URL: str = os.getenv("INTERNAL_API_URL", f"http://localhost:{APP_PORT}")
INTERNAL_TOKEN: str = os.getenv("INTERNAL_TOKEN", "")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_cors_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS: List[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# LLM access
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Redis / Celery
# ---------------------------------------------------------------------------
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_TIMEOUT_SECONDS: int = int(os.getenv("TRAIN_TIMEOUT_SECONDS", "7200"))
SUBPROCESS_TIMEOUT_SECONDS: int = int(os.getenv("SUBPROCESS_TIMEOUT_SECONDS", "30"))
TRAIN_SEED: int = int(os.getenv("TRAIN_SEED", "42"))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
BCRYPT_COST_FACTOR: int = int(os.getenv("BCRYPT_COST_FACTOR", "12"))
LOGIN_MAX_FAILURES_IP: int = int(os.getenv("LOGIN_MAX_FAILURES_IP", "5"))
LOGIN_MAX_FAILURES_ACCOUNT: int = int(os.getenv("LOGIN_MAX_FAILURES_ACCOUNT", "10"))
LOGIN_LOCKOUT_SECONDS: int = int(os.getenv("LOGIN_LOCKOUT_SECONDS", "300"))

# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
KONI_ADMIN_ID: str = os.getenv("KONI_ADMIN_ID", "admin")
KONI_ADMIN_PW: str = os.getenv("KONI_ADMIN_PW", "")

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
JOB_HISTORY_MAX_COUNT: int = int(os.getenv("JOB_HISTORY_MAX_COUNT", "1000"))
JOB_HISTORY_RETENTION_DAYS: int = int(os.getenv("JOB_HISTORY_RETENTION_DAYS", "30"))
LOG_ROTATION_INTERVAL: str = os.getenv("LOG_ROTATION_INTERVAL", "D")
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "30"))
