"""
KONI-Forge User Management — bcrypt hashing, Redis sessions, rate limiting

Security features:
- bcrypt password hashing (configurable cost factor)
- SHA-256 → bcrypt migration (dual verify, re-hash on success)
- Redis-backed sessions with TTL
- Login rate limiting (IP + account)
- user_id whitelist regex validation
"""

import hashlib
import json
import re
import secrets
import time
from typing import Any, Dict, Optional

import bcrypt
import redis

from server.core.config import (
    BCRYPT_COST_FACTOR,
    KONI_ADMIN_ID,
    KONI_ADMIN_PW,
    LOGIN_LOCKOUT_SECONDS,
    LOGIN_MAX_FAILURES_ACCOUNT,
    LOGIN_MAX_FAILURES_IP,
    REDIS_URL,
    SESSION_TTL_SECONDS,
    USERS_JSON,
)

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SESSION_PREFIX = "session:"
_RATE_IP_PREFIX = "login_fail:ip:"
_RATE_ACCOUNT_PREFIX = "login_fail:account:"

# ═══════════════════════════════════════════════════════════
# Redis connection (lazy singleton)
# ═══════════════════════════════════════════════════════════

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Get or create Redis connection."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def set_redis_client(client: redis.Redis) -> None:
    """Override Redis client (for testing with fakeredis)."""
    global _redis_client
    _redis_client = client


# ═══════════════════════════════════════════════════════════
# User ID validation
# ═══════════════════════════════════════════════════════════


def validate_user_id(user_id: str) -> bool:
    """Validate user_id against whitelist regex. Prevents path traversal."""
    return bool(USER_ID_PATTERN.match(user_id))


# ═══════════════════════════════════════════════════════════
# Password hashing
# ═══════════════════════════════════════════════════════════


def hash_password(password: str) -> str:
    """Hash password with bcrypt using configured cost factor."""
    salt = bcrypt.gensalt(rounds=BCRYPT_COST_FACTOR)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def _verify_bcrypt(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _verify_sha256(password: str, hashed: str) -> bool:
    """Verify password against legacy SHA-256 hash (no salt)."""
    sha_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return sha_hash == hashed


def verify_password(password: str, stored_hash: str) -> tuple[bool, bool]:
    """
    Verify password with dual-mode support (bcrypt + legacy SHA-256).

    Returns:
        (is_valid, needs_rehash): is_valid=True if password matches,
        needs_rehash=True if stored as SHA-256 and should be migrated.
    """
    # Try bcrypt first (new format starts with $2b$ or $2a$)
    if stored_hash.startswith(("$2b$", "$2a$")):
        return _verify_bcrypt(password, stored_hash), False

    # Legacy SHA-256 (64-char hex string)
    if len(stored_hash) == 64 and all(c in "0123456789abcdef" for c in stored_hash):
        valid = _verify_sha256(password, stored_hash)
        return valid, valid  # needs_rehash=True only if valid

    return False, False


# ═══════════════════════════════════════════════════════════
# Users file I/O
# ═══════════════════════════════════════════════════════════


def _load_users() -> Dict[str, Any]:
    """Load users from JSON file."""
    if USERS_JSON.exists():
        try:
            return json.loads(USERS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_users(users: Dict[str, Any]) -> None:
    """Atomically save users to JSON file."""
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = USERS_JSON.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(USERS_JSON)


def _ensure_admin() -> None:
    """Ensure admin account exists with env-configured credentials."""
    if not KONI_ADMIN_ID or not KONI_ADMIN_PW:
        return
    users = _load_users()
    if KONI_ADMIN_ID not in users:
        users[KONI_ADMIN_ID] = {
            "password": hash_password(KONI_ADMIN_PW),
            "role": "admin",
            "created_at": time.time(),
        }
        _save_users(users)


# Initialize admin on import
_ensure_admin()


# ═══════════════════════════════════════════════════════════
# Rate limiting
# ═══════════════════════════════════════════════════════════


def _check_rate_limit(ip: Optional[str], user_id: str) -> Optional[str]:
    """
    Check login rate limits. Returns error message if locked out, None if OK.
    Uses Redis counters with TTL for IP and account-based limiting.
    """
    r = _get_redis()

    # Check IP-based limit
    if ip:
        ip_key = f"{_RATE_IP_PREFIX}{ip}"
        ip_count = r.get(ip_key)
        if ip_count and int(ip_count) >= LOGIN_MAX_FAILURES_IP:
            ttl = r.ttl(ip_key)
            return f"Too many failed logins from this IP. Try again in {ttl}s."

    # Check account-based limit
    acct_key = f"{_RATE_ACCOUNT_PREFIX}{user_id}"
    acct_count = r.get(acct_key)
    if acct_count and int(acct_count) >= LOGIN_MAX_FAILURES_ACCOUNT:
        ttl = r.ttl(acct_key)
        return f"Too many failed logins for this account. Try again in {ttl}s."

    return None


def _record_failure(ip: Optional[str], user_id: str) -> None:
    """Record a login failure for rate limiting."""
    r = _get_redis()
    pipe = r.pipeline()

    if ip:
        ip_key = f"{_RATE_IP_PREFIX}{ip}"
        pipe.incr(ip_key)
        pipe.expire(ip_key, LOGIN_LOCKOUT_SECONDS)

    acct_key = f"{_RATE_ACCOUNT_PREFIX}{user_id}"
    pipe.incr(acct_key)
    pipe.expire(acct_key, LOGIN_LOCKOUT_SECONDS)

    pipe.execute()


def _clear_failures(ip: Optional[str], user_id: str) -> None:
    """Clear login failure counters on successful login."""
    r = _get_redis()
    pipe = r.pipeline()

    if ip:
        pipe.delete(f"{_RATE_IP_PREFIX}{ip}")
    pipe.delete(f"{_RATE_ACCOUNT_PREFIX}{user_id}")

    pipe.execute()


# ═══════════════════════════════════════════════════════════
# Session management (Redis-backed)
# ═══════════════════════════════════════════════════════════


def _create_session(user_id: str, role: str) -> str:
    """Create a new session in Redis, return session token."""
    token = secrets.token_urlsafe(48)
    r = _get_redis()
    session_data = json.dumps(
        {
            "user_id": user_id,
            "role": role,
            "created_at": time.time(),
        }
    )
    r.setex(f"{_SESSION_PREFIX}{token}", SESSION_TTL_SECONDS, session_data)
    return token


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a session token. Returns session data dict or None.
    """
    if not token:
        return None
    r = _get_redis()
    data = r.get(f"{_SESSION_PREFIX}{token}")
    if data:
        return json.loads(data)
    return None


def invalidate_session(token: str) -> bool:
    """Delete a session from Redis. Returns True if existed."""
    r = _get_redis()
    return bool(r.delete(f"{_SESSION_PREFIX}{token}"))


# ═══════════════════════════════════════════════════════════
# Core user operations
# ═══════════════════════════════════════════════════════════


def authenticate(
    user_id: str,
    password: str,
    ip: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Authenticate user. Returns dict with 'success', 'token', 'role', or 'error'.

    Features:
    - Rate limiting (IP + account)
    - SHA-256 → bcrypt auto-migration
    - Redis session creation
    """
    if not validate_user_id(user_id):
        return {"success": False, "error": "Invalid user ID."}

    # Rate limit check
    rate_error = _check_rate_limit(ip, user_id)
    if rate_error:
        return {"success": False, "error": rate_error}

    users = _load_users()
    user = users.get(user_id)

    if not user:
        _record_failure(ip, user_id)
        return {"success": False, "error": "Incorrect user ID or password."}

    stored_hash = user.get("password", "")
    is_valid, needs_rehash = verify_password(password, stored_hash)

    if not is_valid:
        _record_failure(ip, user_id)
        return {"success": False, "error": "Incorrect user ID or password."}

    # SHA-256 → bcrypt migration on successful login
    if needs_rehash:
        user["password"] = hash_password(password)
        _save_users(users)

    # Clear rate limit counters
    _clear_failures(ip, user_id)

    # Create Redis session
    role = user.get("role", "user")
    token = _create_session(user_id, role)

    return {
        "success": True,
        "token": token,
        "user_id": user_id,
        "role": role,
    }


