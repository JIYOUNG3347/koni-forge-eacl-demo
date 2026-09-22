"""Decide which celery task ids to revoke when a job is cancelled.

The user-facing ``job_id`` is not the celery task id. The UUIDs of the chain
``submit_gpu_task`` builds live in ``train:state:{job_id}.chain_task_ids``, and
that is what ``POST /api/train/stop`` revokes. Keeping the rule here stops
pipeline cancellation and training stop from diverging.
"""

from __future__ import annotations

import json
from typing import Any, List, Mapping, Optional


def chain_ids(state: Optional[Mapping[str, Any]]) -> List[str]:
    """Celery task ids to revoke, read from the submission metadata.

    A malformed or empty value returns an empty list, so the caller can fall
    back to "try the id itself".
    """
    if not isinstance(state, Mapping):
        return []
    raw = state.get("chain_task_ids")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]


def revoke_targets(job_id: str, state_raw: Optional[str]) -> List[str]:
    """What to revoke in order to stop this sub-task.

    A training job gives the chain UUIDs; anything else (or missing metadata)
    gives the `job_id` itself, which is right for paths like data generation

    Args:
        job_id: the user-facing job name.
        state_raw: the raw ``train:state:{job_id}`` string, or None.
    """
    jid = (job_id or "").strip()
    if not jid:
        return []
    try:
        state = json.loads(state_raw) if state_raw else None
    except (ValueError, TypeError):
        state = None
    ids = chain_ids(state)
    return ids if ids else [jid]


def with_sub_task(state: Optional[Mapping[str, Any]], job_id: str) -> dict:
    """A copy of the pipeline state with this sub-task registered.

    Empty and duplicate values are skipped, so nothing is revoked twice.

    """
    out = dict(state or {})
    jid = (job_id or "").strip()
    subs = out.get("sub_tasks")
    subs = list(subs) if isinstance(subs, (list, tuple)) else []
    if jid and jid not in subs:
        subs.append(jid)
    out["sub_tasks"] = subs
    return out
