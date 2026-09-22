/**
 * systemStore — Zustand store for GPU info, settings and base models.
 */

import { create } from "zustand";
import { apiFetch } from "../lib/apiFetch";
import { t } from "../i18n";

export interface GpuInfo {
  name: string;
  vram_total_gb: number;
  vram_used_gb: number;
  vram_free_gb: number;
  utilization: number;
  temperature: number;
}

export interface SystemSettings {
  agentAutonomy: "auto" | "guided";
}

export interface ModelInfo {
  name: string;
  path?: string; // local dir for base HF models (/api/models/base)
  size_gb: number;
  type: "huggingface";
  loaded: boolean;
}

interface SystemState {
  gpuInfo: GpuInfo | null;
  systemSettings: SystemSettings;
  models: ModelInfo[];
  settingsLoading: boolean;

  fetchGpu: () => Promise<void>;
  fetchSettings: () => Promise<void>;
  updateSettings: (patch: Partial<SystemSettings>) => Promise<void>;
  fetchModels: () => Promise<void>;
}

const DEFAULT_SETTINGS: SystemSettings = {
  agentAutonomy: "guided",
};

export const useSystemStore = create<SystemState>((set, get) => ({
  gpuInfo: null,
  systemSettings: { ...DEFAULT_SETTINGS },
  models: [],
  settingsLoading: true,

  fetchGpu: async () => {
    try {
      const res = await apiFetch("/api/system/gpu");
      if (res.ok) {
        const data = await res.json();
        set({ gpuInfo: data });
      }
    } catch {
      // Silently fail
    }
  },

  fetchSettings: async () => {
    set({ settingsLoading: true });
    try {
      const res = await apiFetch("/api/system/settings");
      if (res.ok) {
        const data = await res.json();
        set({
          systemSettings: {
            agentAutonomy: data.agent_autonomy ?? data.agentAutonomy ?? "guided",
          },
        });
      }
    } catch {
      // Keep defaults
    } finally {
      set({ settingsLoading: false });
    }
  },

  updateSettings: async (patch) => {
    const current = get().systemSettings;
    set({ systemSettings: { ...current, ...patch } });

    const payload: Record<string, unknown> = {};
    if (patch.agentAutonomy !== undefined) payload.agent_autonomy = patch.agentAutonomy;

    const res = await apiFetch("/api/system/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      set({ systemSettings: current });
      const err = await res.json().catch(() => ({}));
      throw new Error((err as Record<string, string>).detail ?? t("Could not save the settings"));
    }
  },

  fetchModels: async () => {
    try {
      const res = await apiFetch("/api/models/base");
      if (res.ok) {
        const data = await res.json();
        set({ models: data.models ?? data ?? [] });
      }
    } catch {
      // Silently fail
    }
  },
}));
