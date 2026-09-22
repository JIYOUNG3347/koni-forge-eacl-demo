"""The queue badge used to show only a count, so a user could not tell whose
job was running, how long was left, or why theirs had not started.
"""

from server.core.queue_view import (
    MAX_ROWS,
    build_gpu_rows,
    build_queue_view,
    build_rows,
    elapsed_seconds,
    format_elapsed,
    waiting_reason,
)

_NOW = 1_786_670_000.0


def _rec(**kw):
    base = {"kind": "training", "status": "STARTED", "job_id": "job-1", "user_id": "admin"}
    base.update(kw)
    return base


# ── classification ───────────────────────────────────────────────────
def test_splits_running_and_waiting():
    run, wait = build_rows([_rec(status="STARTED"), _rec(job_id="job-2", status="QUEUED")], _NOW)
    assert [r["job_id"] for r in run] == ["job-1"]
    assert [r["job_id"] for r in wait] == ["job-2"]
    assert run[0]["status"] == "running" and wait[0]["status"] == "waiting"


def test_unknown_status_is_excluded():
    """Completed and failed jobs must not appear in the queue."""
    run, wait = build_rows([_rec(status="SUCCESS"), _rec(status="FAILURE")], _NOW)
    assert run == [] and wait == []


def test_non_mapping_records_ignored():
    run, wait = build_rows([None, "x", 42, _rec()], _NOW)  # type: ignore[list-item]
    assert len(run) == 1


# ── wait reason: the decision wording is preserved ───────────────────
def test_scheduler_reason_is_preserved_verbatim():
    _, wait = build_rows([_rec(status="QUEUED", admission_reason="training is exclusive — GPU jobs active")], _NOW)
    assert wait[0]["reason"] == "training is exclusive — GPU jobs active"


def test_reason_derived_when_scheduler_silent():
    # Legacy mode has no decision reason — derive a short phrase from the state.
    _, wait = build_rows([_rec(status="STARTED"), _rec(job_id="b", status="QUEUED")], _NOW)
    assert wait[0]["reason"] == "starts when the current job finishes"
    _, only_wait = build_rows([_rec(status="QUEUED")], _NOW)
    assert only_wait[0]["reason"] == "waiting"


def test_waiting_reason_helper():
    assert waiting_reason({"admission_reason": "  because  "}, True) == "because"
    assert waiting_reason({"reason": "alternate key"}, True) == "alternate key"
    assert waiting_reason({}, False) == "waiting"


# ── elapsed time ─────────────────────────────────────────────────────
def test_elapsed_from_epoch_and_iso():
    assert elapsed_seconds(_NOW - 412, _NOW) == 412
    assert elapsed_seconds("2026-08-14T07:00:00+00:00", 1_786_690_800.0) is not None


def test_elapsed_invalid_is_none():
    assert elapsed_seconds(None, _NOW) is None
    assert elapsed_seconds("not-a-date", _NOW) is None


def test_format_elapsed():
    assert format_elapsed(45) == "45s"
    assert format_elapsed(412) == "6m"
    assert format_elapsed(3600) == "1h"
    assert format_elapsed(8040) == "2h 14m"
    assert format_elapsed(86400) == "1d"
    assert format_elapsed(277200) == "3d 5h"


def test_format_elapsed_edges():
    assert format_elapsed(None) == ""
    assert format_elapsed(0) == "0s"
    assert format_elapsed(-5) == "0s"  # clock skew must not break it


def test_format_elapsed_omits_zero_remainder():
    """No meaningless tail such as "2h 0m"."""
    assert format_elapsed(2 * 3600) == "2h"
    assert format_elapsed(2 * 86400) == "2d"
    assert format_elapsed(None) == ""


# ── device info ──────────────────────────────────────────────────────
def test_running_row_gets_gpu_from_lease():
    run, _ = build_rows([_rec()], _NOW, {"job-1": {"indices": (0, 1)}})
    assert run[0]["gpu_indices"] == [0, 1]


def test_gpu_rows_busy_flag():
    rows = build_gpu_rows(
        [
            {"index": 0, "memory_used_mb": 79000, "memory_total_mb": 81920, "utilization_pct": 97},
            {"index": 1, "memory_used_mb": 0, "memory_total_mb": 81920, "utilization_pct": 1},
        ]
    )
    assert rows[0]["busy"] is True and rows[1]["busy"] is False


def test_gpu_rows_accept_nested_physical():
    # The gpu/resources shape (nested under physical) is recognised too.
    rows = build_gpu_rows([{"index": 0, "physical": {"memory_used_mb": 2048, "memory_total_mb": 81920}}])
    assert rows[0]["memory_used_mb"] == 2048 and rows[0]["busy"] is True


# ── ordering and row cap ─────────────────────────────────────────────
def test_sorted_by_elapsed_desc():
    run, _ = build_rows([_rec(job_id="new", started_at=_NOW - 10), _rec(job_id="old", started_at=_NOW - 500)], _NOW)
    assert [r["job_id"] for r in run] == ["old", "new"]


def test_rows_are_capped():
    run, _ = build_rows([_rec(job_id=f"j{i}") for i in range(40)], _NOW)
    assert len(run) <= MAX_ROWS


# ── response assembly: existing keys are preserved ───────────────────
def test_existing_keys_preserved():
    """The badge reads only total and busy — losing these keys is a regression."""
    counts = {"total": 3, "train": 1, "eval": 2, "pipeline": 0, "busy": True}
    out = build_queue_view(counts=counts, running=[], waiting=[], gpus=[], mode="legacy")
    for k, v in counts.items():
        assert out[k] == v
    assert out["mode"] == "legacy"
    assert out["running"] == [] and out["waiting"] == [] and out["gpus"] == []


def test_view_does_not_mutate_counts():
    counts = {"total": 1, "busy": True}
    build_queue_view(counts=counts, running=[{"a": 1}], waiting=[], gpus=[], mode="managed")
    assert counts == {"total": 1, "busy": True}


def test_queue_reads_live_status_not_submission_meta():
    """``train:state:*`` is submission metadata written once by POST /start and
    never refreshed by the worker. Real progress goes to ``job:{job_id}`` via
    ``update_job()``, which is how /api/train/status/{job_id} already reads it.

    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "server/routers/system.py").read_text(encoding="utf-8")
    assert "def _live_status(" in src
    assert "from server.core.state import get_job" in src
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if not any(c in line for c in ("train_count += 1", "eval_count += 1")):
            continue
        window = "\n".join(lines[max(0, i - 8) : i])
        assert "_live_status(" in window, f"no _live_status before {line.strip()}"


def test_live_status_falls_back_on_lookup_failure():
    """A failed lookup must not empty the screen — fall back to submission metadata."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "server/routers/system.py").read_text(encoding="utf-8")
    body = src.split("def _live_status(")[1].split("def _build_queue_detail(")[0]
    assert "except Exception" in body and "return meta_status" in body
