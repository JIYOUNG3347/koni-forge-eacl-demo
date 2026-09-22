"""How the boundary agent drives a probe run.

One POST /api/hf/probe, then polling GET /api/hf/probe/{job_id} until it
reports SUCCESS, FAILURE or the poll times out. openai and fastapi may be
missing in CI, hence the stub-load pattern.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

BOUNDARY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "modules" / "agents" / "specialists" / "boundary_specialist.py"
)


def _load_boundary_module():
    """Load boundary_specialist.py with its dependencies mocked.

    foundation (AgentBase and friends) is stubbed — only _run_knowledge_probes is tested.
    """
    saved = {}
    mocks_needed = {}

    # foundation module stub
    mock_foundation = types.ModuleType("modules.agents.foundation")

    class _AgentBase:
        name = ""

        def __init__(self, *args, **kwargs):
            self.conversation = []
            self._last_probe_results = []
            self._last_probe_model = ""

        async def complete_text(self, *args, **kwargs):
            return ""

        def _rt(self, name, **fields):
            from server.core.agent_runtime_texts import text

            return text(name, "ko", **fields)

        @property
        def _lang(self):
            """On the base class this reads `system_config`. Here it is pinned,
            for the same reason as `_rt`: this test looks at the probe branches,
            not at language resolution.

            """
            return "ko"

    class _AgentEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _AgentPhase:
        IDLE = "idle"
        THINKING = "thinking"
        TOOL_USE = "tool_use"
        RESPONDING = "responding"

    class _ChainSignal:
        def __init__(self, stage_completed="", metadata=None):
            self.stage_completed = stage_completed
            self.metadata = metadata or {}

    mock_foundation.AgentBase = _AgentBase  # type: ignore[attr-defined]
    mock_foundation.AgentEvent = _AgentEvent  # type: ignore[attr-defined]
    mock_foundation.AgentPhase = _AgentPhase  # type: ignore[attr-defined]
    mock_foundation.ChainSignal = _ChainSignal  # type: ignore[attr-defined]
    mocks_needed["modules.agents.foundation"] = mock_foundation

    # Parent package stubs
    if "modules" not in sys.modules:
        mocks_needed["modules"] = types.ModuleType("modules")
    if "modules.agents" not in sys.modules:
        mocks_needed["modules.agents"] = types.ModuleType("modules.agents")
    if "modules.agents.specialists" not in sys.modules:
        mocks_needed["modules.agents.specialists"] = types.ModuleType("modules.agents.specialists")

    # Register foundation on the parent package so the relative import works.
    parent_pkg = sys.modules.get("modules.agents") or types.ModuleType("modules.agents")
    parent_pkg.foundation = mock_foundation  # type: ignore[attr-defined]
    mocks_needed["modules.agents"] = parent_pkg

    for name, mod in mocks_needed.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        spec = importlib.util.spec_from_file_location(
            "modules.agents.specialists.boundary_specialist",
            str(BOUNDARY_PATH),
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


PROBES = [
    {"question": "Q1", "ground_truth": "A1", "category": "general"},
    {"question": "Q2", "ground_truth": "A2", "category": "general"},
]


def _mock_response(status_code: int, json_data: dict):
    """Mock response for httpx.AsyncClient."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data)
    resp.text = json.dumps(json_data)
    return resp


def _mock_async_client(post_returns, get_returns_seq):
    """httpx.AsyncClient context manager mock — one post, several gets."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=post_returns)
    client.get = AsyncMock(side_effect=get_returns_seq)
    return client


class TestRunKnowledgeProbesDispatchAndPoll:
    def test_single_post_dispatch_and_poll_until_success(self):
        mod = _load_boundary_module()
        spec = mod.BoundarySpecialist()

        post_resp = _mock_response(200, {"job_id": "kbd-probe-1", "status": "queued"})
        get_responses = [
            _mock_response(200, {"status": "STARTED", "progress": 10, "message": "probe started"}),
            _mock_response(200, {"status": "STARTED", "progress": 50, "message": "probe 1/2"}),
            _mock_response(
                200,
                {
                    "status": "SUCCESS",
                    "progress": 100,
                    "message": "done",
                    "result": {
                        "model": "qwen-7b",
                        "results": [
                            {
                                "index": 0,
                                "question": "Q1",
                                "ground_truth": "A1",
                                "category": "general",
                                "model_answer": "ans1",
                            },
                            {
                                "index": 1,
                                "question": "Q2",
                                "ground_truth": "A2",
                                "category": "general",
                                "model_answer": "ans2",
                            },
                        ],
                        "total": 2,
                        "errors": 0,
                    },
                },
            ),
        ]

        client = _mock_async_client(post_resp, get_responses)
        with _patch_httpx_client(mod, client), _patch_sleep(mod):
            result_str = asyncio.run(
                spec._run_knowledge_probes(
                    {
                        "model_name": "qwen-7b",
                        "probes": PROBES,
                    }
                )
            )

        result = json.loads(result_str)
        assert result["model"] == "qwen-7b"
        assert result["total"] == 2
        assert len(result["results"]) == 2
        # One dispatch for the whole run, not one per probe.
        client.post.assert_called_once()
        post_url = client.post.call_args.args[0]
        assert post_url.endswith("/api/hf/probe")
        # GET polls until SUCCESS.
        assert client.get.call_count >= 2
        # The result is cached on the instance for score_probe_responses.
        assert spec._last_probe_results == result["results"]
        assert spec._last_probe_model == "qwen-7b"

    def test_failure_status_returns_error(self):
        mod = _load_boundary_module()
        spec = mod.BoundarySpecialist()

        post_resp = _mock_response(200, {"job_id": "kbd-fail", "status": "queued"})
        get_responses = [
            _mock_response(
                200,
                {
                    "status": "FAILURE",
                    "progress": 30,
                    "message": "",
                    "error": "OOM",
                },
            ),
        ]

        client = _mock_async_client(post_resp, get_responses)
        with _patch_httpx_client(mod, client), _patch_sleep(mod):
            result_str = asyncio.run(
                spec._run_knowledge_probes(
                    {
                        "model_name": "qwen-7b",
                        "probes": PROBES,
                    }
                )
            )

        result = json.loads(result_str)
        assert "error" in result
        assert "OOM" in result["error"]

    def test_empty_probes_returns_error(self):
        mod = _load_boundary_module()
        spec = mod.BoundarySpecialist()

        result_str = asyncio.run(
            spec._run_knowledge_probes(
                {
                    "model_name": "qwen-7b",
                    "probes": [],
                }
            )
        )
        result = json.loads(result_str)
        assert "error" in result

    def test_empty_model_name_returns_error(self):
        mod = _load_boundary_module()
        spec = mod.BoundarySpecialist()

        result_str = asyncio.run(
            spec._run_knowledge_probes(
                {
                    "model_name": "",
                    "probes": PROBES,
                }
            )
        )
        result = json.loads(result_str)
        assert "error" in result


# ── Helpers: patch httpx.AsyncClient and asyncio.sleep ──


class _PatchCtx:
    def __init__(self, target_module, attr_path: str, replacement):
        self.target = target_module
        self.attr_path = attr_path
        self.replacement = replacement
        self.saved = None

    def __enter__(self):
        # attr_path is a dotted path such as "httpx.AsyncClient".
        parts = self.attr_path.split(".")
        obj = self.target
        for p in parts[:-1]:
            obj = getattr(obj, p)
        self.saved = getattr(obj, parts[-1])
        setattr(obj, parts[-1], self.replacement)
        self.holder = obj
        self.last_part = parts[-1]
        return self

    def __exit__(self, *args):
        setattr(self.holder, self.last_part, self.saved)


def _patch_httpx_client(mod, client):
    """Replace boundary_specialist's httpx.AsyncClient(...) with client."""
    fake_httpx = types.SimpleNamespace(AsyncClient=lambda *a, **kw: client)
    return _PatchCtx(mod, "httpx", fake_httpx)


def _patch_sleep(mod):
    """Make asyncio.sleep return immediately, to speed the test up."""

    async def _instant_sleep(*args, **kwargs):
        return None

    return _PatchCtx(mod, "asyncio.sleep", _instant_sleep)
