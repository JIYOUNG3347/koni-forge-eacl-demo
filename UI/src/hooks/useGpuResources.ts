import { useEffect, useRef, useState } from "react";

import { apiFetch } from "../lib/apiFetch";

/** Device metrics from the read model. null when unavailable. */
interface PhysicalMetrics {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  utilization_pct: number;
  temperature_c: number;
}

interface ActiveConsumer {
  kind: string;
  workload_class?: string;
}

interface GpuEntry {
  index: number;
  roles: string[];
  hosted_classes: string[];
  physical: PhysicalMetrics | null;
  active_consumers: ActiveConsumer[];
}

export interface TrainCapacityOption {
  method: string;
  mode: string;
  num_gpus: number;
  label: string;
  max_params_b: number;
  basis_gb: number;
  single_gpu_max_b: number;
  advice: string;
}

export interface TrainCapacity {
  gpu_free_gb: number[];
  p2p: boolean | null;
  options: TrainCapacityOption[];
}

export interface GpuResourcesResponse {
  topology: { device_count: number; training_index: number; inference_index: number; device_indices: number[] };
  gpus: GpuEntry[];
  unbound_consumers: ActiveConsumer[];
  coverage?: unknown;
  generated_at?: string;
  train_capacity?: TrainCapacity;
}

/** Per-device view, shaped for consumers. */
export interface GpuView {
  index: number;
  name: string;
  totalMb: number;
  usedMb: number;
  freeMb: number;
  util: number;
  temp: number;
  available: boolean;
  /** Workload classes currently occupying it. Empty means idle. */
  consumers: string[];
}

export interface GpuResourcesView {
  gpus: GpuView[];
  count: number;
  totalUsedMb: number;
  totalVramMb: number;
  avgUtil: number;
  available: boolean;
  loading: boolean;
  trainCapacity: TrainCapacity | null;
}

/** Read model response to the normalised view. Pure, and therefore testable. */
export function normalizeGpuResources(resp: GpuResourcesResponse | null): Omit<GpuResourcesView, "loading"> {
  const entries = resp?.gpus ?? [];
  const gpus: GpuView[] = entries.map((g) => {
    const p = g.physical;
    const consumers = Array.from(
      new Set((g.active_consumers ?? []).map((c) => c.workload_class || c.kind).filter(Boolean)),
    );
    return {
      index: g.index,
      name: p?.name ?? "GPU",
      totalMb: p?.memory_total_mb ?? 0,
      usedMb: p?.memory_used_mb ?? 0,
      freeMb: p?.memory_free_mb ?? 0,
      util: p?.utilization_pct ?? 0,
      temp: p?.temperature_c ?? 0,
      available: p != null,
      consumers,
    };
  });
  const withPhysical = gpus.filter((g) => g.available);
  const totalUsedMb = withPhysical.reduce((a, g) => a + g.usedMb, 0);
  const totalVramMb = withPhysical.reduce((a, g) => a + g.totalMb, 0);
  const avgUtil = withPhysical.length
    ? Math.round(withPhysical.reduce((a, g) => a + g.util, 0) / withPhysical.length)
    : 0;
  return {
    gpus,
    count: gpus.length,
    totalUsedMb,
    totalVramMb,
    avgUtil,
    available: withPhysical.length > 0,
    // Passed through unchanged, preserving the read model schema.
    // Empty options give null — an empty array would read as "no devices".
    trainCapacity: resp?.train_capacity?.options?.length ? resp.train_capacity : null,
  };
}

/** Poll /api/system/gpu/resources every pollMs. */
export function useGpuResources(pollMs = 4000): GpuResourcesView {
  const [resp, setResp] = useState<GpuResourcesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    const load = async () => {
      try {
        const r = await apiFetch("/api/system/gpu/resources");
        if (!r.ok) return;
        const data = (await r.json()) as GpuResourcesResponse;
        if (alive.current) setResp(data);
      } catch {
        // Ignore and retry on the next poll.
      } finally {
        if (alive.current) setLoading(false);
      }
    };
    void load();
    const iv = setInterval(load, pollMs);
    return () => {
      alive.current = false;
      clearInterval(iv);
    };
  }, [pollMs]);

  return { ...normalizeGpuResources(resp), loading };
}
