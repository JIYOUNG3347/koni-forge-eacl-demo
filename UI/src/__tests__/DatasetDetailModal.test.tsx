import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";

import { DatasetDetailModal } from "../components/modals/DatasetDetailModal";

vi.mock("../i18n", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../i18n")>();
  const id = (k: string) => k;
  const fmt = (k: string, ...a: unknown[]) => mod.formatArgs(k, a);
  return { ...mod, t: id, useT: () => id, tp: fmt, useTp: () => fmt };
});

const fetchProvenance = vi.fn();
const fetchStorageInfo = vi.fn();
const fetchDatasetFiles = vi.fn();
const previewDataset = vi.fn();
vi.mock("../stores/dataStore", () => ({
  useDataStore: (selector: (s: Record<string, unknown>) => unknown) =>
    selector({ fetchProvenance, fetchStorageInfo, fetchDatasetFiles, previewDataset }),
}));

let openaiApiKey = "";
vi.mock("../stores/systemStore", () => ({
  useSystemStore: (selector: (s: Record<string, unknown>) => unknown) =>
    selector({ systemSettings: { openaiApiKey } }),
}));

// A processed dataset that can still be refined (not yet refined, not running).
const processedItem = {
  id: "gen_20260615",
  name: "gen_20260615",
  type: "training",
  size: 2048,
  format: "JSON",
  created_at: 1780000000000,
  status: "ready",
  qa_count: 38,
  refined_applied: false,
} as never;

const baseItem = {
  id: "dataset_20260612_111300",
  name: "dataset_20260612_111300",
  type: "raw",
  size: 1024,
  format: "PDF",
  created_at: 1780000000000,
  status: "ready",
  qa_count: 38,
  indexing: { status: "completed", chunk_count: 142 },
} as never;

const NO_FILES_MSG = /This dataset has no source files, or no provenance was recorded/;

beforeEach(() => {
  fetchProvenance.mockReset();
  fetchStorageInfo.mockReset();
  fetchDatasetFiles.mockReset();
  previewDataset.mockReset();
  previewDataset.mockResolvedValue({ items: [], total: 0 });
  openaiApiKey = "";
  // Default empty storage response; the storage-specific test overrides it.
  fetchStorageInfo.mockResolvedValue({
    dataset_id: "x",
    folder_path: "/storage/raw_corpus/x",
    disk_size_bytes: 0,
    file_count: 0,
    last_modified: null,
  });
  // Default empty folder response; the folder-specific test overrides it.
  fetchDatasetFiles.mockResolvedValue({ dataset_id: "x", files: [], total: 0 });
});

describe("DatasetDetailModal", () => {
  it("shows the folder label, files and metadata when provenance exists", async () => {
    fetchProvenance.mockResolvedValue({
      dataset_id: "dataset_20260612_111300",
      available: true,
      provenance: {
        schema_version: 1,
        source_folder: "dataset_20260612_111300",
        source_folder_label: "AI paper collection",
        source_files: [
          { name: "a.pdf", size: 1024 },
          { name: "b.pdf", size: 2048 },
        ],
        created_at: null,
        generator: {},
        pipeline_steps: [],
      },
    });

    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);

    // The source file section starts collapsed, so click the header to expand.
    await waitFor(() => expect(screen.getByText("Source file")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Source file").closest("button")!);
    await waitFor(() => expect(screen.getByText("a.pdf")).toBeInTheDocument());
    expect(screen.getByText("b.pdf")).toBeInTheDocument();
    expect(screen.getByText("AI paper collection")).toBeInTheDocument();
    // Metadata stats: 142 chunks and 2 source files.
    expect(screen.getByText("142")).toBeInTheDocument();
    expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1);
  });

  it("lists folder files directly when provenance is absent (no pre-v2 message)", async () => {
    // The reported case: a raw_corpus dataset with no provenance and two PDFs on disk.
    const rawItem = { ...(baseItem as object), id: "dataset_20260615_042628", indexing: undefined } as never;
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    fetchDatasetFiles.mockResolvedValue({
      dataset_id: "x",
      total: 2,
      files: [
        { name: "economics-for-undergraduates.pdf", size: 622000, mtime: "2026-06-15T04:26:28" },
        { name: "ai-in-primary-education.pdf", size: 791000, mtime: "2026-06-15T04:26:28" },
      ],
    });

    render(<DatasetDetailModal item={rawItem} open onClose={() => {}} />);

    // Wait for the file-count stat cell, which signals the data has loaded.
    await waitFor(() => expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1));
    // The source file section starts collapsed; expand it to see the names.
    fireEvent.click(screen.getByText("Source file").closest("button")!);
    await waitFor(() =>
      expect(screen.getByText("economics-for-undergraduates.pdf")).toBeInTheDocument(),
    );
    expect(screen.getByText("ai-in-primary-education.pdf")).toBeInTheDocument();
    // The misleading "created before v2" / "None" note must not appear.
    expect(screen.queryByText(/Created before v2/)).not.toBeInTheDocument();
    expect(screen.queryByText(NO_FILES_MSG)).not.toBeInTheDocument();
    expect(fetchDatasetFiles).toHaveBeenCalledWith("dataset_20260615_042628");
  });

  it("shows an explicit note and '—' metadata when neither provenance nor folder exists", async () => {
    const rawItem = { ...(baseItem as object), indexing: undefined } as never;
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    fetchDatasetFiles.mockResolvedValue({ dataset_id: "x", files: [], total: 0 });

    render(<DatasetDetailModal item={rawItem} open onClose={() => {}} />);

    // Wait for the storage section, which signals the modal has loaded.
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    // The source file section starts collapsed; expand it to read the note.
    fireEvent.click(screen.getByText("Source file").closest("button")!);
    await waitFor(() => expect(screen.getByText(NO_FILES_MSG)).toBeInTheDocument());
    // The "—" metadata cell is present.
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("a.pdf")).not.toBeInTheDocument();
  });

  it("a raw modal has no QA label (the QA preview is for non-raw only)", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("QA preview")).not.toBeInTheDocument();
  });

  it("the JSONL download button is gone (no footer)", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("Download JSONL")).not.toBeInTheDocument();
  });

  it("there is no chunk preview section (the chunk count is a raw metadata stat)", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("Chunk preview")).not.toBeInTheDocument();
    // The chunk count appears in the raw metadata.
    expect(screen.getByText("142")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    const onClose = vi.fn();
    render(<DatasetDetailModal item={baseItem} open onClose={onClose} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose on Escape", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    const onClose = vi.fn();
    render(<DatasetDetailModal item={baseItem} open onClose={onClose} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    fireEvent.keyDown(document.body, { key: "Escape", code: "Escape" });
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("renders four storage rows (path, disk, files, modified) and a copy button", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    fetchStorageInfo.mockResolvedValue({
      dataset_id: "x",
      folder_path: "/storage/raw_corpus/dataset_xxx",
      disk_size_bytes: 5 * 1024 * 1024,
      file_count: 3,
      last_modified: "2026-06-15T02:35:00",
    });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);

    await waitFor(() =>
      expect(screen.getByText("/storage/raw_corpus/dataset_xxx")).toBeInTheDocument(),
    );
    expect(screen.getByText("Storage information")).toBeInTheDocument();
    expect(screen.getAllByText("5 MB").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getAllByText(/2026/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByRole("button", { name: "Copy path" })).toBeInTheDocument();
  });

  it("calls clipboard.writeText when the copy button is clicked", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    fetchStorageInfo.mockResolvedValue({
      dataset_id: "x",
      folder_path: "/storage/corpus/ds_1",
      disk_size_bytes: 100,
      file_count: 1,
      last_modified: "2026-06-15T02:35:00",
    });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("/storage/corpus/ds_1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Copy path" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("/storage/corpus/ds_1"));
  });

  it("shows '—' for the path and hides the copy button when folder_path is missing", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    fetchStorageInfo.mockResolvedValue({
      dataset_id: "x",
      folder_path: "",
      disk_size_bytes: 0,
      file_count: 0,
      last_modified: null,
    });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.getByText("No files")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy path" })).not.toBeInTheDocument();
  });

  it("shows a vN chip in the header when there are multiple versions", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    const v2 = { ...(baseItem as object), name: "dataset_v2", version: 2 } as never;
    render(<DatasetDetailModal item={v2} open totalVersions={3} onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.getByText("v2")).toBeInTheDocument();
  });

  it("hides the chip for a single version", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    const v1 = { ...(baseItem as object), version: 1 } as never;
    render(<DatasetDetailModal item={v1} open totalVersions={1} onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("v1")).not.toBeInTheDocument();
  });

  it("renders nothing when item is null", () => {
    render(<DatasetDetailModal item={null} open={false} onClose={() => {}} />);
    expect(screen.queryByText("dataset_20260612_111300")).not.toBeInTheDocument();
    expect(fetchProvenance).not.toHaveBeenCalled();
  });

  it("the source file section opens collapsed", async () => {
    fetchProvenance.mockResolvedValue({
      dataset_id: "x",
      available: true,
      provenance: {
        schema_version: 1,
        source_folder: "x",
        source_folder_label: null,
        source_files: [{ name: "hidden.pdf", size: 100 }],
        created_at: null,
        generator: {},
        pipeline_steps: [],
      },
    });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    // After the modal loads, the files must not be visible yet (collapsed).
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("hidden.pdf")).not.toBeInTheDocument();
    // Clicking the header reveals them.
    fireEvent.click(screen.getByText("Source file").closest("button")!);
    await waitFor(() => expect(screen.getByText("hidden.pdf")).toBeInTheDocument());
  });

  it("SFT QA preview shows the first item and pages forward", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    previewDataset.mockResolvedValue({
      items: [
        { question: "What is the capital of Korea?", answer: "Seoul" },
        { question: "What is AI?", answer: "Artificial intelligence" },
      ],
      total: 2,
    });
    render(<DatasetDetailModal item={processedItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("What is the capital of Korea?")).toBeInTheDocument());
    expect(screen.getByText("Seoul")).toBeInTheDocument();
    expect(screen.getByText("QA preview")).toBeInTheDocument();
    // Clicking next shows the second item.
    fireEvent.click(screen.getByText("Next →"));
    await waitFor(() => expect(screen.getByText("What is AI?")).toBeInTheDocument());
    expect(screen.getByText("Artificial intelligence")).toBeInTheDocument();
    expect(previewDataset).toHaveBeenCalledWith("gen_20260615", 20);
  });

  it("a raw dataset has no QA preview and never calls previewDataset", async () => {
    fetchProvenance.mockResolvedValue({ dataset_id: "x", available: false, reason: "pre-v2" });
    render(<DatasetDetailModal item={baseItem} open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Storage information")).toBeInTheDocument());
    expect(screen.queryByText("QA preview")).not.toBeInTheDocument();
    expect(previewDataset).not.toHaveBeenCalled();
  });
});
