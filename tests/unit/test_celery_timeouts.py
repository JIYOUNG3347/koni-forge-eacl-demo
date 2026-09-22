"""The core invariant: visibility_timeout > task_time_limit.

Celery-free pure module, so it runs in a plain CI venv.
"""

import server.core.celery_timeouts as ct

DEFAULT_REDIS_VISIBILITY = 3600


def test_repro_2026_07_22_incident():
    # An 8-hour training run (TRAIN_TIMEOUT_SECONDS=28800) plus a 300s hard-kill margin.
    task_time_limit = 28800 + 300
    vt = ct.broker_visibility_timeout(task_time_limit)
    assert vt > task_time_limit, "redelivery before the hard kill means a duplicate run"
    assert vt > DEFAULT_REDIS_VISIBILITY




# ── Calculation ─────────────────────────────────────────────
def test_adds_margin():
    assert ct.broker_visibility_timeout(29100) == 29100 + ct.DEFAULT_MARGIN_SECONDS


def test_custom_margin():
    assert ct.broker_visibility_timeout(29100, margin=1200) == 30300


# ── Never below 3600, whatever the input ────────────────────
def test_never_below_minimum():
    for bad in (0, -1, 60, None, "abc", 3.5):
        vt = ct.broker_visibility_timeout(bad)
        assert vt >= ct.MIN_VISIBILITY_TIMEOUT
        assert vt > DEFAULT_REDIS_VISIBILITY, f"{bad!r} gave {vt}, below the default"


def test_minimum_exceeds_redis_default():
    # The constant itself must exceed the default to mean anything.
    assert ct.MIN_VISIBILITY_TIMEOUT > DEFAULT_REDIS_VISIBILITY








