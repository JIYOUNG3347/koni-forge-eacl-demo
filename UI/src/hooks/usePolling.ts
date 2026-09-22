import { useEffect, useRef } from "react";

export function usePolling(callback: () => void | Promise<void>, intervalMs: number, enabled = true): void {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const run = () => {
      if (cancelled || document.hidden) return;
      void cbRef.current();
    };
    run(); // once immediately
    const id = window.setInterval(run, intervalMs);
    const onVisible = () => { if (!document.hidden) run(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [intervalMs, enabled]);
}
