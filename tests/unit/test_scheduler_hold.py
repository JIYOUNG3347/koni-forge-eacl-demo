"""The scheduler hold: "do not start anything new right now".

Not a reordering — a blanket pause, which is what is actually needed before
maintenance or a demo. Running jobs are untouched; only new admissions are
blocked. The default is off, so with nobody holding it is a complete no-op.
"""

from pathlib import Path
from typing import Any

from server.core.gpu_admission import decide_admission, hold_message

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _decide(*, kind="training", active_leases=(), hold=None):
    return decide_admission(kind, "j1", active_leases, hold=hold)


# ── off by default ───────────────────────────────────────────────────
def test_no_hold_means_no_change():
    assert _decide().admit is True
    assert _decide(hold=None).admit is True
    assert _decide(hold={}).admit is True
    assert _decide(hold={"on": False}).admit is True


# ── a hold blocks new admissions ─────────────────────────────────────
def test_hold_blocks_new_admission():
    d = _decide(hold={"on": True, "by": "admin"})
    assert d.admit is False
    assert "holding new jobs" in d.reason


def test_hold_reason_names_who_and_why():
    """Without the who and why, users read a hold as an outage."""
    d = _decide(hold={"on": True, "by": "admin", "note": "preparing a demo"})
    assert "admin" in d.reason and "preparing a demo" in d.reason


def test_hold_is_checked_before_other_rules():
    """Another rule matching first would hide the hold behind its own reason."""
    lease = type("L", (), {"consumer_kind": "training", "indices": (0,), "meta": {"job_id": "other"}})()
    d = _decide(active_leases=[lease], hold={"on": True, "by": "admin"})
    assert "holding new jobs" in d.reason


def test_non_target_kinds_pass_through():
    """Blocking chat as well would stop the whole app."""
    assert _decide(kind="chat", hold={"on": True, "by": "admin"}).admit is True


# ── wording ──────────────────────────────────────────────────────────
def test_hold_message_empty_when_off():
    # "nonsense" is deliberately the wrong type: malformed must not read as held.
    cases: list[Any] = [None, {}, {"on": False}, {"on": False, "by": "admin"}, "nonsense"]
    for h in cases:
        assert hold_message(h) == "", h


def test_hold_message_without_who():
    assert hold_message({"on": True}) == "An administrator is holding new jobs"


# ── wiring ───────────────────────────────────────────────────────────
def test_both_admission_paths_pass_hold():
    """Wiring only one of the two paths would leak the hold through the other."""
    src = (PROJECT_ROOT / "server/core/gpu_admission.py").read_text(encoding="utf-8")
    assert src.count("hold=get_hold()") == 2




def test_queue_view_carries_the_same_wording():
    """The screen must not say something different from the decision."""
    src = (PROJECT_ROOT / "server/routers/system.py").read_text(encoding="utf-8")
    assert "hold_message(get_hold())" in src
