"""
KONI-Forge Job State — Redis-based job tracking

Features:
- Jobs stored as Redis hashes (job:{job_id})
- SYSTEM_STATE as Redis key
- Dict-like proxy for backward compatibility
- Job history cleanup (MAX_COUNT, RETENTION_DAYS)
- Atomic operations via Redis pipelines
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

from server.core.config import (
    REDIS_URL,
)

# ═══════════════════════════════════════════════════════════
# Redis connection
# ═══════════════════════════════════════════════════════════

_redis_client: Optional[redis.Redis] = None

_JOB_PREFIX = "job:"
_JOB_INDEX_KEY = "job_index"


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def set_redis_client(client: redis.Redis) -> None:
    """Override Redis client (for testing)."""
    global _redis_client
    _redis_client = client


# ═══════════════════════════════════════════════════════════
# Job state operations
# ═══════════════════════════════════════════════════════════


def create_job(
    job_type: str,
    user_id: str,
    params: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> str:
    """
    Create a new job entry in Redis.
    Returns job_id.
    """
    r = _get_redis()
    if job_id is None:
        job_id = str(uuid.uuid4())[:12]

    now = time.time()
    job_data = {
        "job_id": job_id,
        "type": job_type,
        "user_id": user_id,
        "status": "PENDING",
        "progress": "0",
        "message": "",
        "created_at": str(now),
        "updated_at": str(now),
        "params": json.dumps(params or {}),
        "result": "",
        "error": "",
    }

    pipe = r.pipeline()
    pipe.hset(f"{_JOB_PREFIX}{job_id}", mapping=job_data)
    pipe.zadd(_JOB_INDEX_KEY, {job_id: now})
    pipe.execute()

    return job_id


def update_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """Atomically update job fields."""
    r = _get_redis()
    key = f"{_JOB_PREFIX}{job_id}"
    updates: Dict[str, str] = {"updated_at": str(time.time())}

    if status is not None:
        updates["status"] = status
    if progress is not None:
        updates["progress"] = str(progress)
    if message is not None:
        updates["message"] = message
    if result is not None:
        updates["result"] = json.dumps(result) if not isinstance(result, str) else result
    if error is not None:
        updates["error"] = error

    r.hset(key, mapping=updates)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job data as dict. Returns None if not found."""
    r = _get_redis()
    data = r.hgetall(f"{_JOB_PREFIX}{job_id}")
    if not data:
        return None
    # Parse nested JSON fields
    if "params" in data:
        try:
            data["params"] = json.loads(data["params"])
        except (json.JSONDecodeError, TypeError):
            pass
    if "result" in data and data["result"]:
        try:
            data["result"] = json.loads(data["result"])
        except (json.JSONDecodeError, TypeError):
            pass
    # Coerce numeric fields
    if "progress" in data:
        try:
            data["progress"] = int(data["progress"])
        except (ValueError, TypeError):
            data["progress"] = 0
    return data


def delete_job(job_id: str) -> bool:
    """Delete a job from Redis."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.delete(f"{_JOB_PREFIX}{job_id}")
    pipe.zrem(_JOB_INDEX_KEY, job_id)
    results = pipe.execute()
    return bool(results[0])


# ═══════════════════════════════════════════════════════════
# Job history cleanup
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# System state (singleton Redis key)
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# Dict-like proxy (backward compatibility)
# ═══════════════════════════════════════════════════════════


class JobStateProxy:
    """
    Dict-like access to job state for backward compatibility.

    Usage:
        state = JobStateProxy()
        state["job_id"] = {...}   # creates/updates job
        job = state["job_id"]     # gets job
        del state["job_id"]       # deletes job
    """

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        job = get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        return job

    def __setitem__(self, job_id: str, data: Dict[str, Any]) -> None:
        r = _get_redis()
        # Serialize complex fields
        flat = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v) if v is not None else ""
        flat["updated_at"] = str(time.time())
        r.hset(f"{_JOB_PREFIX}{job_id}", mapping=flat)
        r.zadd(_JOB_INDEX_KEY, {job_id: time.time()})

    def __delitem__(self, job_id: str) -> None:
        if not delete_job(job_id):
            raise KeyError(f"Job not found: {job_id}")

    def __contains__(self, job_id: str) -> bool:
        r = _get_redis()
        return r.exists(f"{_JOB_PREFIX}{job_id}") > 0

    def get(self, job_id: str, default: Any = None) -> Any:
        job = get_job(job_id)
        return job if job is not None else default

    def keys(self) -> List[str]:
        r = _get_redis()
        return r.zrevrange(_JOB_INDEX_KEY, 0, -1)


