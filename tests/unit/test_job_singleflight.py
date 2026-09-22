"""Redis-free pure module, so it runs in a plain CI venv."""

import server.core.job_singleflight as sf

NOW = 1_000_000.0
ME = "task-aaa"
OTHER = "task-bbb"


def test_repro_2026_07_22_duplicate_skipped():
    # With a live owner (a fresh heartbeat), the duplicate does not run.
    assert sf.should_run(OTHER, ME, NOW - 10, NOW) == sf.SKIP_DUPLICATE


def test_first_run_allowed():
    assert sf.should_run(None, ME, None, NOW) == sf.RUN
    assert sf.should_run("", ME, None, NOW) == sf.RUN


def test_reentry_by_same_owner_allowed():
    # The same instance re-entering is not a duplicate.
    assert sf.should_run(ME, ME, NOW - 10, NOW) == sf.RUN


# ── A dead owner is taken over, so nothing locks forever ────
def test_stale_owner_is_taken_over():
    assert sf.should_run(OTHER, ME, NOW - sf.DEFAULT_STALE_SECONDS - 1, NOW) == sf.RUN


def test_boundary_not_yet_stale():
    # Exactly at the boundary it still counts as alive (a duplicate run is worse).
    assert sf.should_run(OTHER, ME, NOW - sf.DEFAULT_STALE_SECONDS, NOW) == sf.SKIP_DUPLICATE


def test_missing_heartbeat_is_stale():
    # An unreadable heartbeat counts as stale, which beats a permanent lock.
    assert sf.should_run(OTHER, ME, None, NOW) == sf.RUN


def test_long_running_job_with_fresh_heartbeat_holds_lock():
    # An 8-hour job keeps ownership as long as it heartbeats.
    assert sf.should_run(OTHER, ME, NOW - 60, NOW) == sf.SKIP_DUPLICATE


# ── is_stale ────────────────────────────────────────────────
def test_is_stale_basic():
    assert sf.is_stale(NOW - 10, NOW) is False
    assert sf.is_stale(NOW - sf.DEFAULT_STALE_SECONDS - 1, NOW) is True


def test_is_stale_custom_window():
    assert sf.is_stale(NOW - 100, NOW, stale_seconds=50) is True
    assert sf.is_stale(NOW - 10, NOW, stale_seconds=50) is False


def test_is_stale_garbage_is_stale():
    assert sf.is_stale(None, NOW) is True
    assert sf.is_stale("abc", NOW) is True  # type: ignore[arg-type]
    assert sf.is_stale(NOW, "abc") is True  # type: ignore[arg-type]
