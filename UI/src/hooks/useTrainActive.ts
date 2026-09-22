import { useSyncExternalStore } from "react";

import { apiFetch } from "../lib/apiFetch";

export interface TrainActiveJob {
  id: string;
  status: string;
  progress: number;
  message: string;
  method: string;
  model: string;
  dataset: string;
  created_at: number;
}

interface Snapshot {
  active: TrainActiveJob[];
  total: number;
  loading: boolean;
}

const EMPTY: Snapshot = { active: [], total: 0, loading: true };
let snapshot: Snapshot = EMPTY;

const subscribers = new Set<() => void>();
let intervalId: number | null = null;
const POLL_MS = 5000;

function emit(): void {
  for (const cb of subscribers) cb();
}

async function tick(): Promise<void> {
  if (document.hidden) return;
  try {
    const r = await apiFetch("/api/train/active");
    if (!r.ok) return;
    const d = await r.json();
    const raw: Array<Record<string, unknown>> = d?.active ?? [];
    const active: TrainActiveJob[] = raw.map((j) => ({
      id: String(j.job_id),
      status: String(j.status ?? "STARTED"),
      progress: typeof j.progress === "number" ? j.progress : 0,
      message: typeof j.message === "string" ? j.message : "",
      method: String(j.method ?? ""),
      model: String(j.model_name ?? ""),
      dataset: String(j.dataset_name ?? ""),
      created_at: j.created_at ? new Date(j.created_at as string).getTime() : Date.now(),
    }));
    snapshot = { active, total: Number(d?.total ?? active.length), loading: false };
    emit();
  } catch {
    /* ignore — retried on the next tick */
  }
}

function onVisible(): void {
  if (!document.hidden) void tick();
}

function start(): void {
  if (intervalId != null) return;
  void tick();
  intervalId = window.setInterval(() => { void tick(); }, POLL_MS);
  document.addEventListener("visibilitychange", onVisible);
}

function stop(): void {
  if (intervalId != null) {
    window.clearInterval(intervalId);
    intervalId = null;
  }
  document.removeEventListener("visibilitychange", onVisible);
}

function subscribe(cb: () => void): () => void {
  subscribers.add(cb);
  if (subscribers.size === 1) start();
  return () => {
    subscribers.delete(cb);
    if (subscribers.size === 0) stop();
  };
}

/** Non-reactive current snapshot, readable outside a render without fetching. */
export function getTrainActiveSnapshot(): Snapshot {
  return snapshot;
}

/** Reactive subscription — N components share the same 5s poller. */
export function useTrainActive(): Snapshot {
  return useSyncExternalStore(subscribe, getTrainActiveSnapshot, getTrainActiveSnapshot);
}
