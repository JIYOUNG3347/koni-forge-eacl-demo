import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";

import TrainView from "../views/guided/TrainView";
import { apiFetch } from "../lib/apiFetch";

const CHECKPOINTS = {
  checkpoints: [
    { job_id: "my-sft-full", name: "final", path: "/storage/outputs/u/completed/my-sft-full", method: "sft", size_mb: 1900, is_lora: false, location: "completed", job_status: null, created_at: "2026-06-15T00:00:00" },
    { job_id: "my-lora", name: "final", path: "/storage/outputs/u/completed/my-lora", method: "lora", size_mb: 50, is_lora: true, location: "completed", job_status: null, created_at: "2026-06-15T00:00:00" },
    { job_id: "stopped-one", name: "final", path: "/storage/outputs/u/completed/stopped-one", method: "sft", size_mb: 1900, is_lora: false, location: "completed", job_status: "stopped", created_at: "2026-06-15T00:00:00" },
  ],
};

vi.mock("../stores/dataStore", async (importActual) => {
  const actual = (await importActual()) as Record<string, unknown>;
  return {
    ...actual,
    useDataStore: (selector: (s: Record<string, unknown>) => unknown) =>
      selector({ trainingDatasets: [], fetchDatasets: vi.fn() }),
  };
});
vi.mock("../hooks/useGpuResources", () => ({
  useGpuResources: () => ({ count: 2, gpus: [] }),
}));
vi.mock("../hooks/usePolling", () => ({ usePolling: () => {} }));
vi.mock("../hooks/useTrainActive", () => ({
  useTrainActive: () => ({ active: [] }),
  getTrainActiveSnapshot: () => ({ active: [] }),
}));
vi.mock("../lib/apiFetch", () => ({
  apiFetch: vi.fn(async (url: string) => {
    if (url === "/api/train/checkpoints")
      return { ok: true, json: async () => CHECKPOINTS };
    if (url === "/api/models/base")
      return { ok: true, json: async () => ({ models: [{ name: "gemma-3-1b", size_gb: 1 }] }) };
    return { ok: true, json: async () => ({}) };
  }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("model selector", () => {
  it("fetches both /api/train/checkpoints and /api/models/base on mount", async () => {
    render(<TrainView />);
    const mock = vi.mocked(apiFetch);
    await waitFor(() =>
      expect(mock.mock.calls.some((c) => c[0] === "/api/train/checkpoints")).toBe(true),
    );
    expect(mock.mock.calls.some((c) => c[0] === "/api/models/base")).toBe(true);
  });

  it("renders the default SFT tab without crashing", () => {
    render(<TrainView />);
    // The default SFT mode renders.
    expect(screen.getByText("Full Fine-tuning")).toBeInTheDocument();
  });
});
