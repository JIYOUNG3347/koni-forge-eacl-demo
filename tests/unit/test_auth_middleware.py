"""The worker reaches the API with an X-Internal-Token header, never by IP.

Covered:
- a correct INTERNAL_TOKEN bypasses as user_id="system"
- a wrong or missing token falls through to normal auth (401 without a token)
- an unset INTERNAL_TOKEN blocks the bypass entirely
- a private client IP earns no privilege on its own
"""

import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

API_PATH = Path(__file__).resolve().parent.parent.parent / "api.py"


def _load_auth_middleware(internal_token: str = "", user_id_validator=None):
    """Extract just the AuthMiddleware class from api.py.

    FastAPI and redis are stubbed so this runs in CI.

    user_id_validator: replaces server.core.users.validate_user_id.
        None uses a default validator allowing alphanumerics, _ and -.
    """
    mocks = {}

    mock_fastapi = types.ModuleType("fastapi")
    mock_fastapi.FastAPI = MagicMock()  # type: ignore[attr-defined]
    mock_fastapi.Request = object  # type: ignore[attr-defined]

    mock_fr = types.ModuleType("fastapi.responses")
    mock_fr.FileResponse = MagicMock()  # type: ignore[attr-defined]
    mock_fr.JSONResponse = MagicMock(  # type: ignore[attr-defined]
        side_effect=lambda status_code, content: {"status_code": status_code, "content": content}
    )
    mocks["fastapi"] = mock_fastapi
    mocks["fastapi.responses"] = mock_fr

    mock_cors = types.ModuleType("fastapi.middleware.cors")
    mock_cors.CORSMiddleware = MagicMock()  # type: ignore[attr-defined]
    mocks["fastapi.middleware.cors"] = mock_cors

    mock_gzip = types.ModuleType("fastapi.middleware.gzip")
    mock_gzip.GZipMiddleware = MagicMock()  # type: ignore[attr-defined]
    mocks["fastapi.middleware.gzip"] = mock_gzip

    mock_static = types.ModuleType("fastapi.staticfiles")
    mock_static.StaticFiles = MagicMock()  # type: ignore[attr-defined]
    mocks["fastapi.staticfiles"] = mock_static

    mock_starlette = types.ModuleType("starlette")
    mock_starlette_mw = types.ModuleType("starlette.middleware")
    mock_starlette_base = types.ModuleType("starlette.middleware.base")

    class _BaseHTTPMiddleware:
        def __init__(self, app):
            self.app = app

    mock_starlette_base.BaseHTTPMiddleware = _BaseHTTPMiddleware  # type: ignore[attr-defined]
    mocks["starlette"] = mock_starlette
    mocks["starlette.middleware"] = mock_starlette_mw
    mocks["starlette.middleware.base"] = mock_starlette_base

    mock_server = types.ModuleType("server")
    mock_server_core = types.ModuleType("server.core")
    mock_config = types.ModuleType("server.core.config")
    mock_config.ACTIVITY_LOG_PATH = Path("/tmp/activity.json")  # type: ignore[attr-defined]
    mock_config.CORS_ORIGINS = ["http://localhost:5173"]  # type: ignore[attr-defined]
    mock_config.REDIS_URL = "redis://localhost:6379/0"  # type: ignore[attr-defined]
    mock_config.VERSION = "test"  # type: ignore[attr-defined]
    mock_logging = types.ModuleType("server.core.logging")
    mock_logging.sys_log = MagicMock()  # type: ignore[attr-defined]
    # The static cache policy is a pure module, so the real implementation is used.
    from server.core import static_cache as real_static_cache

    mock_static_cache = types.ModuleType("server.core.static_cache")
    mock_static_cache.GZIP_MIN_SIZE = real_static_cache.GZIP_MIN_SIZE  # type: ignore[attr-defined]
    mock_static_cache.cache_control_for = real_static_cache.cache_control_for  # type: ignore[attr-defined]
    mock_routers = types.ModuleType("server.routers")
    mock_routers.ALL_ROUTERS = []  # type: ignore[attr-defined]
    mock_users = types.ModuleType("server.core.users")
    if user_id_validator is None:
        import re as _re

        _pat = _re.compile(r"^[A-Za-z0-9_-]+$")

        def user_id_validator(uid):  # noqa: E306
            return bool(_pat.match(uid))

    mock_users.validate_user_id = user_id_validator  # type: ignore[attr-defined]
    mocks["server"] = mock_server
    mocks["server.core"] = mock_server_core
    mocks["server.core.config"] = mock_config
    mocks["server.core.logging"] = mock_logging
    mocks["server.core.static_cache"] = mock_static_cache
    mocks["server.core.users"] = mock_users
    mocks["server.routers"] = mock_routers

    saved = {}
    for k, v in mocks.items():
        saved[k] = sys.modules.get(k)
        sys.modules[k] = v

    try:
        with patch.dict(os.environ, {"INTERNAL_TOKEN": internal_token}, clear=False):
            spec = importlib.util.spec_from_file_location("_api_test", API_PATH)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        for k, original in saved.items():
            if original is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = original

    return mod.AuthMiddleware


def _make_request(headers: dict, path: str = "/api/system/settings", client_host: str = "1.2.3.4"):
    """Build a mock Request."""
    req = MagicMock()
    req.url.path = path
    _h = {k.lower(): v for k, v in headers.items()}
    req.headers.get = lambda k, default=None: _h.get(k.lower(), default)
    req.query_params.get = MagicMock(return_value="")
    req.client = MagicMock()
    req.client.host = client_host
    req.state = MagicMock()
    req.state.user_id = None
    req.state.user_role = None
    return req


def _run(coro):
    """asyncio.run() wrapper, so a coroutine runs without pytest-asyncio."""
    return asyncio.run(coro)


class TestXInternalTokenBypass:
    """Internal service auth via the X-Internal-Token header."""

    def test_valid_token_sets_system_user(self, monkeypatch):
        """A correct INTERNAL_TOKEN gives user_id='system', user_role='admin'."""
        token = "test_token_abc123"
        monkeypatch.setenv("INTERNAL_TOKEN", token)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-Internal-Token": token})
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"
        call_next.assert_called_once()

    def test_wrong_token_falls_through_to_normal_auth(self, monkeypatch):
        """A wrong token means no bypass, so a missing Bearer is a 401."""
        monkeypatch.setenv("INTERNAL_TOKEN", "expected_secret")

        AuthMiddleware = _load_auth_middleware("expected_secret")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-Internal-Token": "wrong_token"})
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "system"
        assert result.get("status_code") == 401

    def test_empty_env_token_blocks_bypass(self, monkeypatch):
        """An empty INTERNAL_TOKEN means no header can bypass."""
        monkeypatch.setenv("INTERNAL_TOKEN", "")

        AuthMiddleware = _load_auth_middleware("")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-Internal-Token": "any_value"})
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "system"
        assert result.get("status_code") == 401

    def test_no_token_header_falls_through(self, monkeypatch):
        """Without the header, normal auth applies and a missing Bearer is a 401."""
        monkeypatch.setenv("INTERNAL_TOKEN", "some_token")

        AuthMiddleware = _load_auth_middleware("some_token")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({})  # no headers
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "system"
        assert result.get("status_code") == 401


class TestClientIpIsNotPrivilege:
    """The client IP grants nothing; only the header does."""

    def test_a_private_ip_without_a_token_requires_auth(self, monkeypatch):
        """client.host=172.17.0.1, no header, so 401."""
        monkeypatch.setenv("INTERNAL_TOKEN", "real_internal_token")

        AuthMiddleware = _load_auth_middleware("real_internal_token")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({}, client_host="172.17.0.1")
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "admin"
        assert result.get("status_code") == 401

    def test_localhost_without_a_token_requires_auth(self, monkeypatch):
        """client.host=127.0.0.1, no header, so 401."""
        monkeypatch.setenv("INTERNAL_TOKEN", "real_internal_token")

        AuthMiddleware = _load_auth_middleware("real_internal_token")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({}, client_host="127.0.0.1")
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "admin"
        assert result.get("status_code") == 401

    def test_a_valid_token_bypasses_from_any_ip(self, monkeypatch):
        """The worker case: a correct X-Internal-Token is what counts."""
        token = "real_internal_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-Internal-Token": token}, client_host="172.17.0.2")
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"


class TestExemptPaths:
    """Auth-exempt paths pass without a token."""

    def test_health_endpoint_passes_without_auth(self):
        """/api/system/health needs no auth."""
        AuthMiddleware = _load_auth_middleware("")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({}, path="/api/system/health")
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        call_next.assert_called_once()

    def test_login_endpoint_passes_without_auth(self):
        """/api/auth/login needs no auth."""
        AuthMiddleware = _load_auth_middleware("")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({}, path="/api/auth/login")
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        call_next.assert_called_once()


def _install_users_stub(monkeypatch, validator=None):
    """Keep the server.core.users.validate_user_id stub alive until dispatch.

    The middleware imports validate_user_id lazily, so the one-shot stub in
    _load_auth_middleware is already gone by the time dispatch runs. The real
    server.core.users needs bcrypt, which CI does not have, so this keeps the
    stub in place with monkeypatch for the whole test.
    """
    if validator is None:
        import re as _re

        _pat = _re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

        def validator(uid):  # noqa: E306
            return bool(_pat.match(uid))

    users_stub = types.ModuleType("server.core.users")
    users_stub.validate_user_id = validator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "server.core.users", users_stub)
    # server and server.core may need stubbing too (the loader may have restored them empty).
    if "server" not in sys.modules:
        monkeypatch.setitem(sys.modules, "server", types.ModuleType("server"))
    if "server.core" not in sys.modules:
        monkeypatch.setitem(sys.modules, "server.core", types.ModuleType("server.core"))


class TestXOnBehalfOfImpersonation:
    """Training started by an agent is recorded under the real user, so it passes
    the user_id filter on /api/train/active and /api/train/checkpoints.
    """

    def test_valid_token_and_valid_user_impersonates(self, monkeypatch):
        """A valid token plus a valid X-On-Behalf-Of gives that user_id and role='user'."""
        token = "real_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)
        _install_users_stub(monkeypatch)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request(
            {
                "X-Internal-Token": token,
                "X-On-Behalf-Of": "jychoi",
            }
        )
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "jychoi"
        # Impersonation never grants admin (defence in depth).
        assert req.state.user_role == "user"
        call_next.assert_called_once()

    def test_valid_token_no_impersonation_header_falls_back_to_system(self, monkeypatch):
        """A valid token with no X-On-Behalf-Of keeps the existing system/admin behaviour."""
        token = "real_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-Internal-Token": token})
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"

    def test_no_token_with_impersonation_header_is_ignored(self, monkeypatch):
        """X-On-Behalf-Of without a token is an impersonation attempt: ignored, 401.

        The security point: a request that did not enter the internal-token branch
        never has X-On-Behalf-Of trusted.
        """
        monkeypatch.setenv("INTERNAL_TOKEN", "real_token")

        AuthMiddleware = _load_auth_middleware("real_token")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request({"X-On-Behalf-Of": "admin"})  # no token
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        # A 401 response, and user_id is not set to admin.
        assert getattr(req.state, "user_id", None) != "admin"
        assert result.get("status_code") == 401

    def test_wrong_token_with_impersonation_header_is_ignored(self, monkeypatch):
        """A wrong token means the header is ignored and normal auth applies (401)."""
        monkeypatch.setenv("INTERNAL_TOKEN", "real_token")

        AuthMiddleware = _load_auth_middleware("real_token")
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request(
            {
                "X-Internal-Token": "wrong",
                "X-On-Behalf-Of": "jychoi",
            }
        )
        call_next = AsyncMock(return_value=MagicMock())

        result = _run(middleware.dispatch(req, call_next))

        assert getattr(req.state, "user_id", None) != "jychoi"
        assert result.get("status_code") == 401

    def test_invalid_user_id_falls_back_to_system(self, monkeypatch):
        """A valid token whose X-On-Behalf-Of fails validation falls back to system.

        validate_user_id rejects it, so no impersonation happens and user_id='system'.
        This keeps an agent-started training run from dying with a 401.
        """
        token = "real_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)
        # Keep a validator stub that rejects every user_id alive until dispatch.
        _install_users_stub(monkeypatch, validator=lambda uid: False)

        AuthMiddleware = _load_auth_middleware(token, user_id_validator=lambda uid: False)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request(
            {
                "X-Internal-Token": token,
                "X-On-Behalf-Of": "../etc/passwd",
            }
        )
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"

    def test_empty_impersonation_header_falls_back_to_system(self, monkeypatch):
        """An empty X-On-Behalf-Of falls back to system (treated as absent)."""
        token = "real_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request(
            {
                "X-Internal-Token": token,
                "X-On-Behalf-Of": "   ",  # whitespace only, empty after strip
            }
        )
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"

    def test_system_impersonation_target_is_noop(self, monkeypatch):
        """X-On-Behalf-Of=='system' is a no-op, identical to the plain system branch."""
        token = "real_token"
        monkeypatch.setenv("INTERNAL_TOKEN", token)

        AuthMiddleware = _load_auth_middleware(token)
        middleware = AuthMiddleware(app=MagicMock())

        req = _make_request(
            {
                "X-Internal-Token": token,
                "X-On-Behalf-Of": "system",
            }
        )
        call_next = AsyncMock(return_value=MagicMock())

        _run(middleware.dispatch(req, call_next))

        assert req.state.user_id == "system"
        assert req.state.user_role == "admin"
