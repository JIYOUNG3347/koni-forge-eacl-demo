"""KONI-Forge API v1 — FastAPI main application.

Clean Slate rewrite: 10 domain routers, auth middleware, CORS,
static files, SPA fallback. No legacy /v1/* routes.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from server.core.config import (
    ACTIVITY_LOG_PATH,
    CORS_ORIGINS,
    REDIS_URL,
    VERSION,
)
from server.core.logging import sys_log
from server.core.static_cache import GZIP_MIN_SIZE, cache_control_for
from server.routers import ALL_ROUTERS

# ═══════════════════════════════════════════════════════════════════════════
# Auth Middleware
# ═══════════════════════════════════════════════════════════════════════════

# Paths exempt from authentication
_AUTH_EXEMPT = {
    "/api/system/health",
    "/api/system/ready",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/rag/index/execute",  # worker -> app ChromaDB write
}

_AUTH_EXEMPT_PREFIXES = (
    "/assets",
    "/static",
    "/favicon",
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token authentication middleware.

    Validates session tokens via Redis. Sets request.state.user_id
    and request.state.user_role for downstream handlers.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for exempt paths
        if path in _AUTH_EXEMPT:
            return await call_next(request)

        if path.startswith(_AUTH_EXEMPT_PREFIXES):
            return await call_next(request)

        # Skip auth for SPA fallback (non-API, non-asset paths)
        if not path.startswith("/api/"):
            return await call_next(request)

        internal_token = request.headers.get("X-Internal-Token", "")
        expected_token = os.environ.get("INTERNAL_TOKEN", "")
        if internal_token and expected_token and internal_token == expected_token:
            # X-On-Behalf-Of is trusted only when the internal token is valid, so
            # agent-started work is recorded under the real user and shows up in
            # the training UI. A request without a valid token never reaches this
            # branch, so the header cannot be used to impersonate from outside.
            on_behalf_of = request.headers.get("X-On-Behalf-Of", "").strip()
            if on_behalf_of and on_behalf_of != "system":
                try:
                    from server.core.users import validate_user_id

                    if validate_user_id(on_behalf_of):
                        request.state.user_id = on_behalf_of
                        # The impersonated identity is pinned to "user" so it never
                        # gains more privilege than the original. Matching user_id
                        # alone is enough to pass the /api/train filters.
                        request.state.user_role = "user"
                        return await call_next(request)
                except Exception as e:
                    sys_log(
                        f"[Auth] X-On-Behalf-Of validation error: {e}",
                        level="WARNING",
                    )
            request.state.user_id = "system"
            request.state.user_role = "admin"
            return await call_next(request)

        # Extract Bearer token (header or query param for SSE)
        auth_header: Optional[str] = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
        else:
            # Fallback: query param (for EventSource/SSE which can't set headers)
            token = request.query_params.get("token", "")

        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        # Validate session in Redis
        try:
            from server.core.users import validate_session

            session = validate_session(token)
        except Exception as e:
            sys_log(f"[Auth] Session validation error: {e}", level="WARNING")
            return JSONResponse(
                status_code=503,
                content={"detail": "Authentication service unavailable"},
            )

        if session is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired session"},
            )

        # Set user info on request state
        request.state.user_id = session.get("user_id", "unknown")
        request.state.user_role = session.get("role", "user")

        return await call_next(request)


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

_cleanup_task: Optional[asyncio.Task] = None


async def _orchestrator_cleanup_loop():
    """Periodic cleanup of stale pipeline states and expired jobs."""
    while True:
        try:
            await asyncio.sleep(300)  # every 5 minutes

            def _run_cleanup():
                import redis as _redis

                r = _redis.from_url(REDIS_URL, decode_responses=True)
                cursor = 0
                cleaned = 0
                while True:
                    cursor, keys = r.scan(cursor, match="pipeline:state:*", count=50)
                    for key in keys:
                        try:
                            state = json.loads(r.get(key) or "{}")
                            status = state.get("status", "")
                            if status in ("completed", "failed", "cancelled"):
                                ttl = r.ttl(key)
                                if ttl == -1:  # no expiry set
                                    r.expire(key, 86400 * 7)  # 7 days
                                    cleaned += 1
                        except Exception:
                            continue
                    if cursor == 0:
                        break
                return cleaned

            cleaned = await asyncio.to_thread(_run_cleanup)
            if cleaned:
                sys_log(f"[Cleanup] Set TTL on {cleaned} stale pipeline states")
        except asyncio.CancelledError:
            break
        except Exception as e:
            sys_log(f"[Cleanup] Error: {e}", level="WARNING")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    global _cleanup_task

    # ── Startup ──
    sys_log("[Startup] KONI-Forge API v1 starting...")

    # SQLite DB — create tables if not exist
    try:
        from server.core.db import init_db

        await init_db()
        sys_log("[Startup] SQLite DB initialised")
    except Exception as e:
        sys_log(f"[Startup] SQLite DB init failed: {e}", level="WARNING")

    # Clear activity log
    try:
        ACTIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVITY_LOG_PATH.write_text("[]", encoding="utf-8")
        sys_log("[Startup] Activity log cleared")
    except Exception as e:
        sys_log(f"[Startup] Activity log clear failed: {e}", level="WARNING")

    # Start orchestrator cleanup loop
    _cleanup_task = asyncio.create_task(_orchestrator_cleanup_loop())
    sys_log("[Startup] Orchestrator cleanup loop started")

    sys_log(f"[Startup] KONI-Forge API v1 ready (version={VERSION})")

    yield

    # ── Shutdown ──
    sys_log("[Shutdown] KONI-Forge API v1 shutting down...")

    # Cancel cleanup task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass

    # Cleanup: revoke running jobs (best-effort)
    try:
        import redis

        r = redis.from_url(REDIS_URL, decode_responses=True)
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="train:state:*", count=50)
            for key in keys:
                try:
                    state = json.loads(r.get(key) or "{}")
                    if state.get("status") in ("running", "queued"):
                        job_id = state.get("job_id")
                        if job_id:
                            from celery.result import AsyncResult

                            AsyncResult(job_id).revoke(terminate=True, signal="SIGTERM")
                            sys_log(f"[Shutdown] Revoked job: {job_id}")
                except Exception:
                    continue
            if cursor == 0:
                break
    except Exception as e:
        sys_log(f"[Shutdown] Job cleanup error: {e}", level="WARNING")

    sys_log("[Shutdown] KONI-Forge API v1 stopped")


# ═══════════════════════════════════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="KONI-Forge API v1",
    version=VERSION,
    description="MultiAgent LLMOps Platform — Clean Slate Rewrite",
    lifespan=lifespan,
)

# ── Middleware ──

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
app.add_middleware(AuthMiddleware)


# ── Routers ──

for router in ALL_ROUTERS:
    app.include_router(router)


# ── Static files ──

_STATIC_DIR = Path(__file__).parent / "static"
_ASSETS_DIR = _STATIC_DIR / "assets"


class _CachedStaticFiles(StaticFiles):
    """Add Cache-Control to static file responses.

    Vite stamps a content hash into file names, so a hashed file is cached
    immutably — no revalidation round trip on every refresh. The decision
    lives in server.core.static_cache, which is framework-free.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = cache_control_for(path)
        return response


if _ASSETS_DIR.exists():
    # gzip only inside /assets. As global middleware it would buffer SSE
    # (text/event-stream) responses and cut the agent chat and event streams.
    app.mount(
        "/assets",
        GZipMiddleware(
            _CachedStaticFiles(directory=str(_ASSETS_DIR)),
            minimum_size=GZIP_MIN_SIZE,
        ),
        name="assets",
    )


# ── SPA fallback ──

_INDEX_HTML = _STATIC_DIR / "index.html"


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(request: Request, full_path: str):
    """Catch-all route for SPA (React Router).

    Serves index.html for any path that doesn't match an API route
    or static file. This enables client-side routing.
    """
    # Don't serve index.html for API routes (should have been handled above)
    if full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # Serve specific static files
    static_path = _STATIC_DIR / full_path
    if static_path.exists() and static_path.is_file():
        return FileResponse(str(static_path))

    # SPA fallback
    if _INDEX_HTML.exists():
        return FileResponse(str(_INDEX_HTML))

    return JSONResponse(status_code=404, content={"detail": "Not found"})
