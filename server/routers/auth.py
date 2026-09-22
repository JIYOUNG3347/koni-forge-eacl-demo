"""Authentication router — /api/auth

POST /login  — bcrypt verify, Redis session, rate limit check
POST /logout — invalidate session
GET  /me     — return current user info from request.state
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server.core.logging import sys_log

router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: str
    role: str
    message: str = "login successful"


class UserInfo(BaseModel):
    user_id: str
    role: str


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request):
    """Authenticate user with bcrypt, issue Redis session token.

    Uses server.core.users.authenticate() which handles:
    - Rate limiting (IP + account based)
    - bcrypt verification with SHA-256 migration
    - Redis session creation
    """
    from server.core.users import authenticate

    # Extract client IP for rate limiting
    client_ip = request.client.host if request.client else None

    result = authenticate(req.username, req.password, ip=client_ip)

    if not result.get("success"):
        error_msg = result.get("error", "Invalid credentials")
        # Distinguish rate limit from auth failure
        if "Too many failed logins" in error_msg:
            raise HTTPException(status_code=429, detail=error_msg)
        raise HTTPException(status_code=401, detail=error_msg)

    sys_log(f"[Auth] User '{req.username}' logged in (role={result['role']})")
    return LoginResponse(
        token=result["token"],
        user_id=result["user_id"],
        role=result["role"],
    )


# ---------------------------------------------------------------------------
# POST /logout
# ---------------------------------------------------------------------------


@router.post("/logout")
async def logout(request: Request):
    """Invalidate the current session."""
    from server.core.users import invalidate_session

    auth_header: Optional[str] = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token provided")

    token = auth_header.split(" ", 1)[1]
    deleted = invalidate_session(token)
    if deleted:
        sys_log("[Auth] Session invalidated")
    return {"message": "logged out"}


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------


@router.get("/me", response_model=UserInfo)
async def me(request: Request):
    """Return current user info from request.state (set by AuthMiddleware)."""
    user_id = getattr(request.state, "user_id", None)
    user_role = getattr(request.state, "user_role", None)

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return UserInfo(
        user_id=user_id,
        role=user_role or "user",
    )
