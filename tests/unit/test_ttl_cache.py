"""TTLCache is pure (threading and time only), so it imports in CI without fastapi.
The clock is monkeypatched on time.monotonic for a deterministic check.
"""

import server.core.ttl_cache as tc
from server.core.ttl_cache import TTLCache


def test_caches_within_ttl_then_recomputes(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(tc.time, "monotonic", lambda: clock[0])
    calls = []

    def compute():
        calls.append(1)
        return {"v": len(calls)}

    c = TTLCache(10.0)
    assert c.get_or_compute("k", compute) == {"v": 1}
    clock[0] += 5
    assert c.get_or_compute("k", compute) == {"v": 1}  # within the TTL, so cached
    assert len(calls) == 1
    clock[0] += 6  # 11s total, past the 10s TTL
    assert c.get_or_compute("k", compute) == {"v": 2}  # expired, recomputed
    assert len(calls) == 2


def test_signature_change_invalidates_within_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(tc.time, "monotonic", lambda: clock[0])
    calls = []
    sig = [1]

    def compute():
        calls.append(1)
        return len(calls)

    c = TTLCache(100.0)  # a long TTL still invalidates on a signature change
    c.get_or_compute("k", compute, signature=lambda: sig[0])
    c.get_or_compute("k", compute, signature=lambda: sig[0])  # same signature, cached
    assert len(calls) == 1
    sig[0] = 2  # simulate new output (the root mtime changed)
    c.get_or_compute("k", compute, signature=lambda: sig[0])
    assert len(calls) == 2  # recomputed at once


def test_keys_isolated(monkeypatch):
    monkeypatch.setattr(tc.time, "monotonic", lambda: 0.0)
    calls = {"a": 0, "b": 0}
    c = TTLCache(10.0)
    c.get_or_compute("a", lambda: calls.__setitem__("a", calls["a"] + 1))
    c.get_or_compute("b", lambda: calls.__setitem__("b", calls["b"] + 1))
    c.get_or_compute("a", lambda: calls.__setitem__("a", calls["a"] + 1))  # cache hit for a
    assert calls == {"a": 1, "b": 1}


def test_invalidate(monkeypatch):
    monkeypatch.setattr(tc.time, "monotonic", lambda: 0.0)
    calls = []
    c = TTLCache(100.0)
    c.get_or_compute("k", lambda: calls.append(1))
    c.invalidate("k")
    c.get_or_compute("k", lambda: calls.append(1))  # recomputed after invalidation
    assert len(calls) == 2
    c.invalidate()  # clear everything
    c.get_or_compute("k", lambda: calls.append(1))
    assert len(calls) == 3
