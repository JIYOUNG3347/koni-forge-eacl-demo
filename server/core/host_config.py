"""``host.toml`` parser — operator limits for storage, training and concurrency.

The file is optional: without it the defaults below apply, which is what a
local single-machine run wants. Present but invalid stops the boot, so a typo
never silently falls back.

Search order:
  1. ``$KONI_HOST_CONFIG_PATH``   (explicit override)
  2. ``./host.toml``              (project root)
  3. ``/etc/koniforge/host.toml`` (system wide)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelLimitsSection(BaseModel):
    """Ceilings applied to every training job, whatever the agent recommends."""

    max_train_param_b: int = 14
    max_train_seq_length: int = 2048
    max_train_batch_size: int = 8

    @field_validator("max_train_param_b", "max_train_seq_length", "max_train_batch_size")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class ConcurrencySection(BaseModel):
    max_concurrent_train_jobs: int = 1
    max_concurrent_kbd_jobs: int = 1

    @field_validator("max_concurrent_train_jobs", "max_concurrent_kbd_jobs")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class PathsSection(BaseModel):
    storage_base: str = str(PROJECT_ROOT / "storage")


class HostConfig(BaseModel):
    model_limits: ModelLimitsSection = Field(default_factory=ModelLimitsSection)
    concurrency: ConcurrencySection = Field(default_factory=ConcurrencySection)
    paths: PathsSection = Field(default_factory=PathsSection)


def load(path: Path) -> HostConfig:
    """Parse one host.toml. Exits the process on a parse or validation error."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        _fatal(f"host.toml not found: {path}")
    except tomllib.TOMLDecodeError as e:
        _fatal(f"host.toml is not valid TOML: {e}")

    try:
        return HostConfig.model_validate(data)
    except Exception as e:
        _fatal(f"host.toml failed validation:\n{e}\nSee host.toml.example for the accepted keys.")


def find_and_load() -> HostConfig:
    """Load the first host.toml found, or the defaults when there is none."""
    override = os.getenv("KONI_HOST_CONFIG_PATH")
    if override:
        return load(Path(override))
    for candidate in (PROJECT_ROOT / "host.toml", Path("/etc/koniforge/host.toml")):
        if candidate.exists():
            return load(candidate)
    return HostConfig()


def _fatal(msg: str) -> NoReturn:
    print(f"\n[KONI-Forge] FATAL: {msg}\n", file=sys.stderr)
    sys.exit(1)


HOST_CONFIG: HostConfig = find_and_load()
