"""Shared test fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test env vars BEFORE importing any project modules
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # Use DB 15 for tests
os.environ.setdefault("INTERNAL_API_URL", "http://localhost:8000")
os.environ.setdefault("NETWORK_MODE", "closed")
os.environ.setdefault("BCRYPT_COST_FACTOR", "4")  # Fast for tests
os.environ.setdefault(
    "KONI_HOST_CONFIG_PATH",
    str(Path(__file__).parent / "fixtures" / "host.toml"),
)


@pytest.fixture
def tmp_storage(tmp_path):
    """Create temporary storage structure."""
    for d in ["corpus", "models", "raw_corpus", "outputs", "chroma"]:
        (tmp_path / d).mkdir()
    return tmp_path
