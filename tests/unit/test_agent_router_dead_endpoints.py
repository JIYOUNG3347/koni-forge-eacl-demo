from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_ROUTER = PROJECT_ROOT / "server" / "routers" / "agent.py"


def test_no_phantom_dispatcher_methods_in_routers():
    """The router must not call dispatcher methods that do not exist."""
    for path in (PROJECT_ROOT / "server" / "routers").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "handle_message" not in text, f"{path.name}: references the nonexistent dispatcher.handle_message"
        assert ".abort()" not in text, f"{path.name}: references the nonexistent dispatcher.abort"


def test_dead_endpoints_removed():
    text = _AGENT_ROUTER.read_text(encoding="utf-8")
    assert 'websocket("/chat")' not in text, "the dead WS /chat endpoint is back"
    assert '"/abort"' not in text, "the falsely-succeeding POST /abort endpoint is back"
    assert "AbortRequest" not in text, "the dead AbortRequest schema is back"


def test_live_endpoints_still_present():
    """Live endpoints must remain (a guard against over-deletion)."""
    text = _AGENT_ROUTER.read_text(encoding="utf-8")
    for route in ('"/status"', '"/reset"', '"/dispatch"', '"/pipeline"'):
        assert route in text, f"the live endpoint {route} disappeared"
