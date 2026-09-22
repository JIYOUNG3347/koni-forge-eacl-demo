"""These do not run in CI, which only runs ``pytest tests/unit/``.
Run them locally, opt-in, against a running stack:

    KONI_E2E=1 pytest tests/e2e/ -v

Environment:
    KONI_E2E=1              — the switch (everything skips without it)
    KONI_E2E_BASE_URL       — defaults to http://localhost:8000
    KONI_E2E_USER / _PASS   — when set, logs in and runs the authenticated scenarios
                              (without them those scenarios skip)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BASE_URL = os.getenv("KONI_E2E_BASE_URL", "http://localhost:8000")


def _enabled() -> bool:
    return os.getenv("KONI_E2E") == "1"


def pytest_collection_modifyitems(config, items):
    """Skip the e2e tests only. The hook sees every collected item, so a
    directory check is what keeps ``pytest tests`` from skipping the unit suite."""
    if _enabled():
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="e2e is opt-in — needs KONI_E2E=1 and a running stack")
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def http():
    httpx = pytest.importorskip("httpx")
    with httpx.Client(base_url=BASE_URL, timeout=15) as client:
        yield client


@pytest.fixture(scope="session")
def auth_headers(http):
    """Login token header. Skips the related tests when no credentials are given."""
    user, pw = os.getenv("KONI_E2E_USER"), os.getenv("KONI_E2E_PASS")
    if not user or not pw:
        pytest.skip("KONI_E2E_USER/_PASS not set — skipping authenticated scenarios")
    r = http.post("/api/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, f"login failed ({r.status_code}): {r.text[:200]}"
    return {"Authorization": f"Bearer {r.json()['token']}"}
