"""Small TTL cache for list handlers that would otherwise scan the disk per request.
Redis-free. Simple TTL plus optional signature invalidation: with a signature,
a cheap value (such as the root directory mtime) forces a recompute as soon as
it changes, so new output appears without waiting for the TTL.

Thread safe. compute() runs outside the lock so a slow scan does not block
other keys (a miss can rarely compute twice, which is fine at a short TTL).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


class TTLCache:
    def __init__(self, ttl_s: float):
        self._ttl = ttl_s
        self._lock = threading.Lock()
        # key -> (timestamp, signature, value)
        self._store: dict[Any, tuple[float, Any, Any]] = {}

    def get_or_compute(
        self,
        key: Any,
        compute: Callable[[], Any],
        signature: Optional[Callable[[], Any]] = None,
    ) -> Any:
        sig = signature() if signature is not None else None
        now = time.monotonic()
        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                ts, cached_sig, value = hit
                if (now - ts) < self._ttl and cached_sig == sig:
                    return value
        value = compute()
        with self._lock:
            self._store[key] = (time.monotonic(), sig, value)
        return value

    def invalidate(self, key: Any = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)
