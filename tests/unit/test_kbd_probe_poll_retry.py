"""During a long KBD probe poll, uvicorn can close the keepalive connection and
the next poll fails with httpx.RemoteProtocolError("Server disconnected").
The probe result is already stored as SUCCESS in the worker job state, so the
poll must retry rather than mistake a finished analysis for a failure.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import modules.agents.specialists.boundary_specialist as mod
from modules.agents.specialists.boundary_specialist import BoundarySpecialist


def _resp(status, data):
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=data)
    r.text = json.dumps(data)
    return r


def _client(post_resp, get_side_effect):
    c = AsyncMock()
    c.__aenter__.return_value = c
    c.__aexit__.return_value = False
    c.post = AsyncMock(return_value=post_resp)
    c.get = AsyncMock(side_effect=get_side_effect)
    return c


def _run(spec):
    return json.loads(
        asyncio.run(
            spec._run_knowledge_probes(
                {
                    "model_name": "Qwen/Qwen2.5-3B-Instruct",
                    "probes": [{"question": "Q", "ground_truth": "A", "category": "c"}],
                }
            )
        )
    )


_SUCCESS = {"status": "SUCCESS", "progress": 100, "result": {"results": [{"index": 0}], "total": 1, "errors": 0}}


def test_poll_retries_transient_disconnect_then_succeeds():
    # The first poll disconnects and the second succeeds, so the retry returns the result.
    spec = BoundarySpecialist()
    c = _client(
        post_resp=_resp(200, {"job_id": "kbd-1"}),
        get_side_effect=[
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            _resp(200, _SUCCESS),
        ],
    )
    with patch.object(mod.httpx, "AsyncClient", lambda *a, **k: c), patch("asyncio.sleep", new=AsyncMock()):
        out = _run(spec)
    assert "error" not in out
    assert out["total"] == 1 and len(out["results"]) == 1


def test_transient_error_caught_in_source():
    import inspect

    src = inspect.getsource(BoundarySpecialist._run_knowledge_probes)
    # Check at source level that a transient transport error is retried.
    assert "httpx.TransportError" in src
