import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";

import { usePolling } from "../hooks/usePolling";

function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { value: hidden, configurable: true });
}

describe("usePolling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setHidden(false);
  });
  afterEach(() => {
    vi.useRealTimers();
    setHidden(false);
  });

  it("runs immediately then every intervalMs", () => {
    const cb = vi.fn();
    renderHook(() => usePolling(cb, 5000));
    expect(cb).toHaveBeenCalledTimes(1); // once immediately
    vi.advanceTimersByTime(15000);
    expect(cb).toHaveBeenCalledTimes(4); // +3
  });

  it("skips while document.hidden (paused in a background tab)", () => {
    setHidden(true);
    const cb = vi.fn();
    renderHook(() => usePolling(cb, 5000));
    expect(cb).not.toHaveBeenCalled(); // the immediate run is skipped too
    vi.advanceTimersByTime(20000);
    expect(cb).not.toHaveBeenCalled(); // and so is the interval tick
  });

  it("clears interval on unmount (no accumulation)", () => {
    const cb = vi.fn();
    const { unmount } = renderHook(() => usePolling(cb, 5000));
    cb.mockClear();
    unmount();
    vi.advanceTimersByTime(20000);
    expect(cb).not.toHaveBeenCalled();
  });

  it("does not poll when disabled", () => {
    const cb = vi.fn();
    renderHook(() => usePolling(cb, 5000, false));
    vi.advanceTimersByTime(20000);
    expect(cb).not.toHaveBeenCalled();
  });
});

describe("useTrainActive single poller fan-out", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setHidden(false);
  });
  afterEach(() => {
    vi.useRealTimers();
    setHidden(false);
    vi.restoreAllMocks();
  });

  it("calls /api/train/active once per tick however many subscribers there are", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ active: [{ job_id: "t1", status: "RUNNING" }], total: 1 }),
    });
    vi.doMock("../lib/apiFetch", () => ({ apiFetch: fetchMock }));
    const { useTrainActive } = await import("../hooks/useTrainActive");

    // Two subscribers (separate renders) share the same singleton poller.
    const a = renderHook(() => useTrainActive());
    const b = renderHook(() => useTrainActive());
    // One tick immediately on mount.
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1); // still once with two subscribers
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetchMock).toHaveBeenCalledTimes(2); // and once on the next tick
    a.unmount();
    b.unmount();
  });
});
