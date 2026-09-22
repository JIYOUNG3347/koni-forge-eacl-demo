"""Read-only end-to-end smoke tests — nothing here consumes a GPU."""

from __future__ import annotations


def test_openapi_serves_current_surface(http):
    """The app starts, dead endpoints are gone, and the new read model is there."""
    r = http.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/agent/abort" not in paths, "the dead /abort route is back, or an old process is serving"
    assert "/api/agent/orchestration/trace" in paths
    assert "/api/system/gpu/resources" in paths


def test_gpu_resources_read_model(http, auth_headers):
    """Device inventory read model."""
    r = http.get("/api/system/gpu/resources", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["topology"]["device_count"] >= 1
    assert isinstance(data["gpus"], list) and len(data["gpus"]) == data["topology"]["device_count"]
    for gpu in data["gpus"]:
        assert "index" in gpu and "active_consumers" in gpu


def test_orchestration_trace_read_model(http, auth_headers):
    """Traverse read model: session, mode and event schema."""
    r = http.get("/api/agent/orchestration/trace", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert set(data) >= {"session_id", "session_mode", "orchestrator_mode", "events"}
    assert data["orchestrator_mode"] in ("legacy", "graph")
    assert isinstance(data["events"], list)


def test_agent_status_alive(http, auth_headers):
    r = http.get("/api/agent/status", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("active", "error")
    if body.get("status") == "active":
        names = {a["name"] for a in body.get("agents") or []}
        assert names == {"retrieval", "boundary", "tuning"}
