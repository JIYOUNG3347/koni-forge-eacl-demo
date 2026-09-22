import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";

// Tests DataCountBadge inside DatasetEntry, through DataView.
import { DatasetEntry } from "../views/guided/DataView";
import { DatasetDetailModal } from "../components/modals/DatasetDetailModal";

vi.mock("../i18n", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../i18n")>();
  const id = (k: string) => k;
  const fmt = (k: string, ...a: unknown[]) => mod.formatArgs(k, a);
  return { ...mod, t: id, useT: () => id, tp: fmt, useTp: () => fmt };
});

/* ── store mocks ── */
const fetchProvenance = vi.fn();
const fetchStorageInfo = vi.fn();
const fetchDatasetFiles = vi.fn();
const fetchLineage = vi.fn();
const startPreprocessing = vi.fn();
const fetchGenerationParams = vi.fn();
const previewDataset = vi.fn();
vi.mock("../stores/dataStore", () => ({
  useDataStore: (selector: (s: Record<string, unknown>) => unknown) =>
    selector({ fetchProvenance, fetchStorageInfo, fetchDatasetFiles, fetchLineage, startPreprocessing, fetchGenerationParams, previewDataset }),
}));
vi.mock("../stores/systemStore", () => ({
  useSystemStore: (selector: (s: Record<string, unknown>) => unknown) =>
    selector({ systemSettings: { openaiApiKey: "" } }),
}));

beforeEach(() => {
  fetchProvenance.mockReset();
  fetchStorageInfo.mockReset();
  fetchDatasetFiles.mockReset();
  fetchLineage.mockReset();
  startPreprocessing.mockReset();
  fetchGenerationParams.mockReset();
  previewDataset.mockReset();
  fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
  fetchStorageInfo.mockResolvedValue({
    dataset_id: "x", folder_path: "/storage/corpus/x",
    disk_size_bytes: 1024, file_count: 2, last_modified: null,
  });
  fetchDatasetFiles.mockResolvedValue({ dataset_id: "x", files: [], total: 0 });
  fetchLineage.mockResolvedValue({ dataset_id: "x", stages: [] });
  startPreprocessing.mockResolvedValue({ job_id: "j1", status: "queued", method: "dlmax_algo" });
  fetchGenerationParams.mockResolvedValue(null);
  previewDataset.mockResolvedValue({ items: [], total: 0 });
});

/* ── shared DatasetItem factory ── */
const mkItem = (over: Record<string, unknown> = {}) => ({
  id: "ds_x",
  name: "ds_x",
  type: "training" as const,
  size: 1024,
  format: "JSONL",
  created_at: Date.now(),
  status: "ready" as const,
  qa_count: 100,
  refined_applied: false,
  ...over,
});

/* ═══════════════════════════════════════════
   DataCountBadge label (through DatasetEntry)
═══════════════════════════════════════════ */
describe("DataCountBadge label", () => {
  it("data_format=sft → 'SFT 100'", () => {
    render(<DatasetEntry item={mkItem({ data_format: "sft" }) as never} />);
    expect(screen.getByText("SFT 100")).toBeInTheDocument();
  });

  it("data_format=null (carry-over) defaults to 'SFT 100'", () => {
    render(<DatasetEntry item={mkItem({ data_format: null }) as never} />);
    expect(screen.getByText("SFT 100")).toBeInTheDocument();
  });

});

/* ═══════════════════════════════════════════
   DatasetDetailModal — preference branch
═══════════════════════════════════════════ */
const sftItem = mkItem({ data_format: "sft", type: "training" });

describe("DatasetDetailModal — SFT behaviour unchanged", () => {
  it("shows the source file section button and the QA count metadata cell", async () => {
    render(<DatasetDetailModal item={sftItem as never} open onClose={() => {}} />);
    // SFT metadata grid: QA count, size, status (chunks are raw only).
    await waitFor(() => expect(screen.getByText("Source file")).toBeInTheDocument());
    expect(screen.getByText("QA count")).toBeInTheDocument();
    // No preference cell.
    expect(screen.queryByText("Preference")).not.toBeInTheDocument();
  });

});
