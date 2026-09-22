/**
 * pipelineStore — training job list shared by the header badge and views.
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { apiFetch } from "../lib/apiFetch";
import { userScopedStorage } from "../utils/storage";

export interface PipelineJob {
  id: string;
  method: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "paused";
  progress: number;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  result?: unknown;
  error?: string;
}

interface PipelineState {
  jobs: PipelineJob[];
  runningJob: PipelineJob | null;

  fetchJobs: () => Promise<void>;
  stopJob: (jobId: string) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;
  setJobs: (jobs: PipelineJob[]) => void;
}

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set, get) => ({
      jobs: [],
      runningJob: null,

      fetchJobs: async () => {
        try {
          const res = await apiFetch("/api/train/checkpoints");
          if (res.ok) {
            const data = await res.json();
            const jobs: PipelineJob[] = (data.tasks ?? data.jobs ?? data ?? []).map(
              (j: Record<string, unknown>) => ({
                id: j.id as string,
                method: (j.method ?? j.type ?? "") as string,
                status: (j.status ?? "queued") as PipelineJob["status"],
                progress: (j.progress ?? 0) as number,
                created_at: (j.created_at ?? Date.now()) as number,
                started_at: j.started_at as number | undefined,
                completed_at: j.completed_at as number | undefined,
                result: j.result,
                error: j.error as string | undefined,
              }),
            );
            const runningJob = jobs.find((j) => j.status === "running") ?? null;
            set({ jobs, runningJob });
          }
        } catch {
          // Network error — keep existing state
        }
      },

      stopJob: async (jobId) => {
        await apiFetch(`/api/train/stop/${jobId}`, { method: "POST" });
        await get().fetchJobs();
      },

      deleteJob: async (jobId) => {
        await apiFetch(`/api/train/status/${jobId}`, { method: "DELETE" });
        set((s) => ({
          jobs: s.jobs.filter((j) => j.id !== jobId),
          runningJob: s.runningJob?.id === jobId ? null : s.runningJob,
        }));
      },

      setJobs: (jobs) => {
        const runningJob = jobs.find((j) => j.status === "running") ?? null;
        set({ jobs, runningJob });
      },
    }),
    {
      name: "koni-job-state",
      storage: createJSONStorage(() => userScopedStorage),
      partialize: (state) => ({ runningJob: state.runningJob }),
    },
  ),
);
