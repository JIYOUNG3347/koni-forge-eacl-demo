"""How the boundary agent downloads a model it needs.

POST /api/models/download/hf returns a job id, then GET
/api/models/download/{job_id}/status is polled until it finishes. httpx and
asyncio.sleep are mocked to cover success, dispatch failure, job failure and
the poll timeout.
"""

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import modules.agents.specialists.boundary_specialist as mod
from modules.agents.specialists.boundary_specialist import BoundarySpecialist


def _resp(status, data):
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=data)
    r.text = json.dumps(data)
    return r


def _client(post_resp=None, get_resp=None):
    c = AsyncMock()
    c.__aenter__.return_value = c
    c.__aexit__.return_value = False
    if post_resp is not None:
        c.post = AsyncMock(return_value=post_resp)
    if get_resp is not None:
        c.get = AsyncMock(return_value=get_resp)
    return c


def _run(spec, model_id):
    return json.loads(asyncio.run(spec._download_model({"model_id": model_id})))


def test_hf_download_success():
    spec = BoundarySpecialist()
    c = _client(
        post_resp=_resp(200, {"job_id": "dl-1", "status": "queued"}),
        get_resp=_resp(200, {"status": "SUCCESS", "progress": 100, "message": "", "result": {}, "error": ""}),
    )
    with patch.object(mod.httpx, "AsyncClient", lambda *a, **k: c), patch("asyncio.sleep", new=AsyncMock()):
        out = _run(spec, "Qwen/Qwen2.5-3B-Instruct")
    assert out.get("success") is True
    assert out["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    # Dispatched to the real HF download endpoint.
    assert "/api/models/download/hf" in c.post.call_args.args[0]
    assert c.post.call_args.kwargs["json"] == {"model_id": "Qwen/Qwen2.5-3B-Instruct"}
    # Polling uses the job-based path.
    assert "/api/models/download/dl-1/status" in c.get.call_args.args[0]


def test_missing_job_id_errors():
    spec = BoundarySpecialist()
    c = _client(post_resp=_resp(200, {"status": "queued"}))  # no job_id
    with patch.object(mod.httpx, "AsyncClient", lambda *a, **k: c), patch("asyncio.sleep", new=AsyncMock()):
        out = _run(spec, "Qwen/Qwen2.5-7B-Instruct")
    assert "job_id" in out["error"]


def test_download_suggestions_use_full_repo_ids():
    # A larger model recommended by KBD needs a full HF repo id (org/model) to download.
    src = inspect.getsource(BoundarySpecialist)
    assert "google/gemma-3-1b-it" in src
    assert "Qwen/Qwen2.5-3B-Instruct" in src
    assert "Qwen/Qwen2.5-7B-Instruct" in src


def test_no_dead_fetch_endpoint():
    src = inspect.getsource(BoundarySpecialist._download_model)
    for dead in ("/api/models/fetch/local", "/api/models/fetch/hub", "/api/models/fetch/status"):
        assert dead not in src, f"dead endpoint still referenced: {dead}"
    assert "/api/models/download/hf" in src
