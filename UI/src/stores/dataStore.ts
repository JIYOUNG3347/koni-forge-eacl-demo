/**
 * dataStore — Zustand store for dataset management.
 *
 * State: rawDatasets[], trainingDatasets[], uploadProgress
 * Actions: fetchDatasets, uploadFiles, deleteDataset, deleteDatasetFile, preview/provenance readers
 */

import { create } from "zustand";
import { apiFetch } from "../lib/apiFetch";
import { t } from "../i18n";

/**
 * Upload result. `agentHandoff` is the instruction the server composed for the
 * agent panel (null when the server did not provide one — the UI never makes
 * one up).
 */
export interface UploadResult {
  folder: string;
  agentHandoff: string | null;
}

export interface DatasetIndexingStatus {
  status: "queued" | "indexing" | "completed" | "failed";
  chunk_count: number;
  job_id?: string | null;
  progress?: number;
  message?: string;
  error?: string;
  files_processed?: number;
}

export interface QaPreviewItem {
  question?: string;
  answer?: string;
  category?: string;
  [key: string]: unknown;
}

export interface DatasetFileEntry {
  name: string;
  kind: "data" | "meta";
  active: boolean;
  size: number;
}

export interface DatasetItem {
  id: string;
  name: string;
  type: "raw" | "training";
  size: number;
  format: string;
  created_at: number;
  status: "ready" | "processing" | "error";
  indexing?: DatasetIndexingStatus;
  qa_count?: number | null;
  meta?: Record<string, unknown>;
  base_id?: string;
  version?: number;
  file_entries?: DatasetFileEntry[];
}

export interface DatasetGroup {
  base_id: string;
  versions: DatasetItem[];
}
export function groupByBaseId(items: DatasetItem[]): DatasetGroup[] {
  const map = new Map<string, DatasetItem[]>();
  for (const it of items) {
    const key = it.base_id ?? it.name;
    (map.get(key) ?? map.set(key, []).get(key)!).push(it);
  }
  const groups: DatasetGroup[] = [];
  for (const [base_id, versions] of map) {
    versions.sort((a, b) => (b.version ?? 1) - (a.version ?? 1));
    groups.push({ base_id, versions });
  }
  // Group order: newest (highest version) by creation date, descending.
  groups.sort((a, b) => (b.versions[0]?.created_at ?? 0) - (a.versions[0]?.created_at ?? 0));
  return groups;
}

export interface ProvenanceSourceFile {
  name: string;
  size: number;
  mtime?: string | null;
}
export interface ProvenanceData {
  schema_version: number;
  source_folder: string;
  source_folder_label?: string | null;
  source_files: ProvenanceSourceFile[];
  created_at?: string | null;
  generator: Record<string, unknown>;
  pipeline_steps: Record<string, unknown>[];
}
export interface ProvenanceResponse {
  dataset_id: string;
  available: boolean;
  reason?: string;
  provenance?: ProvenanceData;
}

export interface ChunkPreview {
  chunk_index: number | null;
  page_num: number | null;
  doc_name: string | null;
  text: string;
  truncated: boolean;
}
export interface ChunksResponse {
  dataset_id: string;
  chunks: ChunkPreview[];
  total: number;
}

export interface DatasetFile {
  name: string;
  size: number;
  mtime?: string | null;
}
export interface DatasetFilesResponse {
  dataset_id: string;
  files: DatasetFile[];
  total: number;
}

export interface StorageInfoResponse {
  dataset_id: string;
  folder_path: string;
  disk_size_bytes: number;
  file_count: number;
  last_modified: string | null;
}

interface DataState {
  rawDatasets: DatasetItem[];
  trainingDatasets: DatasetItem[];
  uploadProgress: number;
  isUploading: boolean;
  isLoadingDatasets: boolean;

  fetchDatasets: () => Promise<void>;
  uploadFiles: (
    files: File[],
    target?: "raw" | "processed" | "training",
    folderName?: string | null,
  ) => Promise<UploadResult>;
  deleteDataset: (datasetId: string, category?: "raw" | "training") => Promise<void>;
  deleteDatasetFile: (datasetId: string, fileName: string) => Promise<{ trainingReady: boolean }>;
  previewDataset: (datasetId: string, maxItems?: number) => Promise<{
    items: QaPreviewItem[];
    total: number;
    sourceFile?: string | null;
  }>;
  fetchProvenance: (datasetId: string) => Promise<ProvenanceResponse>;
  fetchChunks: (datasetId: string, limit?: number, offset?: number) => Promise<ChunksResponse>;
  fetchStorageInfo: (datasetId: string) => Promise<StorageInfoResponse>;
  fetchDatasetFiles: (datasetId: string) => Promise<DatasetFilesResponse>;
}

export const useDataStore = create<DataState>((set, get) => ({
  rawDatasets: [],
  trainingDatasets: [],
  uploadProgress: 0,
  isUploading: false,
  isLoadingDatasets: false,

  fetchDatasets: async () => {
    set({ isLoadingDatasets: true });
    try {
      const res = await apiFetch("/api/data/datasets");
      if (res.ok) {
        const data = await res.json();

        // API returns { raw: [...], processed: [...], training: [...] }
        // Each item has: id, category, name, file_count, files[], size_mb, created_at
        const mapItems = (items: any[], type: DatasetItem["type"]): DatasetItem[] =>
          (items ?? []).map((d: any) => ({
            id: d.id ?? d.name,
            name: d.name ?? d.id,
            type,
            size: Math.round((d.size_mb ?? 0) * 1024 * 1024),
            format: d.files?.[0]?.split(".").pop()?.toUpperCase() ?? `${d.file_count ?? 0} files`,
            created_at: d.created_at ? new Date(d.created_at).getTime() : Date.now(),
            status: "ready" as const,
            indexing: d.indexing
              ? {
                  status: d.indexing.status,
                  chunk_count: d.indexing.chunk_count ?? 0,
                  job_id: d.indexing.job_id ?? null,
                  progress: d.indexing.progress,
                  message: d.indexing.message,
                  error: d.indexing.error,
                  files_processed: d.indexing.files_processed,
                }
              : undefined,
            base_id: typeof d.base_id === "string" ? d.base_id : (d.name ?? d.id),
            version: typeof d.version === "number" ? d.version : 1,
            qa_count: typeof d.qa_count === "number" ? d.qa_count : null,
            file_entries: Array.isArray(d.file_entries)
              ? (d.file_entries as any[]).map((f) => ({
                  name: String(f?.name ?? ""),
                  kind: f?.kind === "meta" ? ("meta" as const) : ("data" as const),
                  active: Boolean(f?.active),
                  size: typeof f?.size === "number" ? f.size : 0,
                }))
              : undefined,
          }));

        set({
          rawDatasets: mapItems(data.raw, "raw"),
          trainingDatasets: mapItems(data.training, "training"),
          isLoadingDatasets: false,
        });
      } else {
        set({ isLoadingDatasets: false });
      }
    } catch {
      // Keep existing state on network error
      set({ isLoadingDatasets: false });
    }
  },

  uploadFiles: async (files, target = "raw", folderName = null) => {
    set({ isUploading: true, uploadProgress: 0 });
    let uploadedFolder = "";
    let agentHandoff: string | null = null;
    try {
      const formData = new FormData();
      files.forEach((f) => {
        // Folder uploads (webkitdirectory) carry a path prefix in
        // webkitRelativePath ("subdir/file.pdf"); send only the basename so the
        // backend writes into the dataset dir, not a non-existent subdir.
        const rel = (f as { webkitRelativePath?: string }).webkitRelativePath;
        const baseName = rel ? rel.split("/").pop() || f.name : f.name;
        formData.append("files", f, baseName);
      });
      formData.append("target", target);
      if (folderName) formData.append("folder_name", folderName);

      const xhr = new XMLHttpRequest();
      const token = localStorage.getItem("kf-token") ?? "";

      await new Promise<void>((resolve, reject) => {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            set({ uploadProgress: Math.round((e.loaded / e.total) * 100) });
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const resp = JSON.parse(xhr.responseText);
              uploadedFolder = resp.folder ?? "";
              agentHandoff = typeof resp.agent_handoff === "string" ? resp.agent_handoff : null;
            } catch {
        /* ignore */
      }
            resolve();
          } else {
            let errMsg = "Upload failed";
            try {
              const body = JSON.parse(xhr.responseText);
              if (body.detail) errMsg = String(body.detail);
            } catch {
        /* ignore */
      }
            reject(new Error(errMsg));
          }
        };
        xhr.onerror = () => reject(new Error(t("Network error")));
        xhr.open("POST", "/api/data/upload");
        if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
        xhr.send(formData);
      });

      await get().fetchDatasets();
    } finally {
      set({ isUploading: false, uploadProgress: 0 });
    }
    return { folder: uploadedFolder, agentHandoff };
  },

  deleteDatasetFile: async (datasetId, fileName) => {
    const res = await apiFetch(
      `/api/data/datasets/${encodeURIComponent(datasetId)}/files/${encodeURIComponent(fileName)}`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as Record<string, string>).detail ?? t("Failed to delete file"));
    }
    const data = (await res.json().catch(() => ({}))) as { training_ready?: boolean };
    await get().fetchDatasets();
    return { trainingReady: Boolean(data.training_ready) };
  },

  deleteDataset: async (datasetId, category) => {
    // category pins the deletion to the bucket the user pointed at.
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    await apiFetch(`/api/data/datasets/${datasetId}${qs}`, { method: "DELETE" });
    // Optimistic update: only remove from the targeted category list so
    // the other copies (if any) stay visible until the next fetch
    // reconciles. Without category we fall back to filtering all three
    // (legacy semantics — the backend's first-match also touches one
    // copy, which the cache invalidate will reveal on the next poll).
    set((s) => {
      if (category === "raw") return { rawDatasets: s.rawDatasets.filter((d) => d.id !== datasetId) };
      if (category === "training") return { trainingDatasets: s.trainingDatasets.filter((d) => d.id !== datasetId) };
      return {
        rawDatasets: s.rawDatasets.filter((d) => d.id !== datasetId),
        trainingDatasets: s.trainingDatasets.filter((d) => d.id !== datasetId),
      };
    });
  },

  previewDataset: async (datasetId, maxItems = 20) => {
    const res = await apiFetch(`/api/data/datasets/${encodeURIComponent(datasetId)}/preview?max_items=${maxItems}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? t("Could not load the preview"));
    }
    const data = (await res.json()) as {
      items?: QaPreviewItem[];
      total?: number;
      source_file?: string | null;
    };
    return {
      items: data.items ?? [],
      total: data.total ?? (data.items?.length ?? 0),
      sourceFile: data.source_file ?? null,
    };
  },

  fetchProvenance: async (datasetId) => {
    const res = await apiFetch(`/api/data/datasets/${encodeURIComponent(datasetId)}/provenance`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? t("Could not load the lineage"));
    }
    return (await res.json()) as ProvenanceResponse;
  },

  fetchChunks: async (datasetId, limit = 10, offset = 0) => {
    const res = await apiFetch(
      `/api/data/datasets/${encodeURIComponent(datasetId)}/chunks?limit=${limit}&offset=${offset}`,
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? t("Could not load the chunks"));
    }
    return (await res.json()) as ChunksResponse;
  },

  fetchStorageInfo: async (datasetId) => {
    const res = await apiFetch(`/api/data/datasets/${encodeURIComponent(datasetId)}/storage`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? t("Could not load the storage information"));
    }
    return (await res.json()) as StorageInfoResponse;
  },

  fetchDatasetFiles: async (datasetId) => {
    const res = await apiFetch(`/api/data/datasets/${encodeURIComponent(datasetId)}/files`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error((err as { detail?: string }).detail ?? t("Could not load the source files"));
    }
    return (await res.json()) as DatasetFilesResponse;
  },

}));
