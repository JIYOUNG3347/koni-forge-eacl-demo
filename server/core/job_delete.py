"""Whether a job delete may proceed.

A job stopped or refused early never writes a folder, so requiring artifacts on
disk would leave it in the list with no way to remove it.

Rules:
  * job_id mode: proceed when Redis still has a record, even with nothing on
    disk — the user is deleting "that job", not "that folder".
"""

from __future__ import annotations


def should_delete(*, has_disk_targets: bool, has_state: bool, path_mode: bool) -> bool:
    """Whether the delete should proceed.

    Args:
        has_disk_targets: is there at least one folder to delete?
        has_state: is there still a job record in Redis (train:state / job:)?
        path_mode: is this a partial delete of one specific path?
    """
    if has_disk_targets:
        return True
    if path_mode:
        # A path was named and it does not exist — the request itself is wrong.
        return False
    return has_state


def not_found_detail(job_id: str, path_mode: bool) -> str:
    """Failure message, naming which of the two was missing."""
    if path_mode:
        return "The specified path was not found."
    return f"No training record found for '{job_id}'."
