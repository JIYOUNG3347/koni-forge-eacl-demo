"""A cancelled pipeline left training running::
    pipeline:control:pipe-…  = {"action": "cancel"}   <- the signal was there
    pipeline:state.status    = cancelled              <- the state changed too
    job:sft_agent_gen_…      STARTED  training, step 6/18

Two defects overlapped:

1. **Training was not in the revoke set.** Cancellation walks
   `pipeline:state.sub_tasks`, and only the data-generation specialist registered there.
2. **Even registered, that id would not work.** The user-facing `job_id` is not
   the celery task id — the chain UUIDs (unload, train, reload) live in
   `train:state:{job_id}.chain_task_ids`, which is what `/api/train/stop` uses.
"""

import json
from pathlib import Path

from server.core.job_revoke import (
    chain_ids,
    revoke_targets,
    with_sub_task,
)

_ROOT = Path(__file__).resolve().parents[2]

TRAIN_STATE = json.dumps(
    {
        "job_id": "sft_agent_gen_1788495520",
        "chain_task_ids": ["c261a24c-0bf4-4195-a3f5-9100528be545", "d0880a9e-b7b3-417c-9407-a8c026e0ab6b"],
    }
)


# ── What has to be revoked ──────────────────────────────────────────────────


def test_training_revokes_the_chain_not_the_job_name():
    assert revoke_targets("sft_agent_gen_1788495520", TRAIN_STATE) == [
        "c261a24c-0bf4-4195-a3f5-9100528be545",
        "d0880a9e-b7b3-417c-9407-a8c026e0ab6b",
    ]


def test_non_training_jobs_keep_the_old_behaviour():
    """For data generation the job_id is the task id, as before."""
    assert revoke_targets("gen-20260904041841-026571", None) == ["gen-20260904041841-026571"]


def test_broken_state_falls_back_to_the_job_id():
    """Giving up on a broken record would leave the GPU tied up."""
    for raw in ("not json", "", "null", json.dumps({"chain_task_ids": "a string"})):
        assert revoke_targets("job-x", raw) == ["job-x"]


def test_empty_chain_falls_back():
    assert revoke_targets("job-x", json.dumps({"chain_task_ids": []})) == ["job-x"]


def test_chain_ids_ignores_junk_entries():
    assert chain_ids({"chain_task_ids": ["a", "", "  ", None, 3, "b"]}) == ["a", "b"]
    assert chain_ids(None) == []
    assert chain_ids("a string") == []  # type: ignore[arg-type]


def test_no_job_id_is_not_a_revoke_target():
    for jid in ("", "   ", None):
        assert revoke_targets(jid, TRAIN_STATE) == []  # type: ignore[arg-type]


# ── Expanding the whole set ─────────────────────────────────────────────────








# ── Registration rule ───────────────────────────────────────────────────────


def test_registration_is_idempotent():
    assert with_sub_task({"sub_tasks": ["a"]}, "a")["sub_tasks"] == ["a"]
    assert with_sub_task({"sub_tasks": ["a"]}, "b")["sub_tasks"] == ["a", "b"]


def test_registration_keeps_other_fields_and_ignores_blanks():
    out = with_sub_task({"run_id": "r1", "status": "running"}, "   ")
    assert out["run_id"] == "r1" and out["status"] == "running"
    assert out["sub_tasks"] == []


def test_registration_does_not_mutate_the_input():
    original = {"sub_tasks": ["a"]}
    with_sub_task(original, "b")
    assert original["sub_tasks"] == ["a"]


# ── Wiring guard ────────────────────────────────────────────────────────────
