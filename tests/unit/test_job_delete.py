"""Stopping a training run right after starting it used to make deletion fail::

    delete failed: No training artifacts found for 'train-20260820001900-0ae862'

`DELETE /api/train/jobs/{job_id}` required artifacts on disk. A job stopped or
refused early (REVOKED, batch guard) never creates a folder, so it matched
nothing, returned 404 and never reached the Redis cleanup — so it stayed in
the list forever.
"""

from pathlib import Path

from server.core.job_delete import not_found_detail, should_delete

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_disk_targets_always_delete():
    assert should_delete(has_disk_targets=True, has_state=False, path_mode=False) is True
    assert should_delete(has_disk_targets=True, has_state=False, path_mode=True) is True


def test_state_only_job_is_deletable():
    """The user is deleting "that job", not "that folder"."""
    assert should_delete(has_disk_targets=False, has_state=True, path_mode=False) is True


def test_path_mode_requires_the_path():
    """A named path that does not exist is a wrong request — never silently succeed."""
    assert should_delete(has_disk_targets=False, has_state=True, path_mode=True) is False


def test_nothing_anywhere_is_not_found():
    assert should_delete(has_disk_targets=False, has_state=False, path_mode=False) is False


def test_messages_distinguish_the_cause():
    """Naming which of the two was missing tells the user what to do next."""
    assert "path" in not_found_detail("j1", path_mode=True)
    assert "j1" in not_found_detail("j1", path_mode=False)


def test_router_checks_state_before_404():
    src = (PROJECT_ROOT / "server/routers/train.py").read_text(encoding="utf-8")
    assert "should_delete(" in src
    block = src.split("should_delete(")[0][-700:]
    assert "train:state:" in block and "job:" in block  # state is checked first


def test_router_still_cleans_redis_after_deleting():
    """Redis cleanup must run even when there are no artifacts to delete."""
    src = (PROJECT_ROOT / "server/routers/train.py").read_text(encoding="utf-8")
    after = src.split("should_delete(")[1]
    assert 'delete(f"train:state:{job_id}")' in after
    assert 'delete(f"job:{job_id}")' in after
