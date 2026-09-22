/**
 * DataView — document / QA-dataset upload and the dataset list.
 *
 * Upload 1: documents (PDF/DOCX/PPTX/TXT/MD) — indexed for RAG and KBD
 * Upload 2: training data (JSON/JSONL) — used directly for training
 */

import React, { useEffect, useCallback, useState, useRef } from "react";
import {
  Upload, UploadCloud, FolderUp, FileText, File, FileCode, FileSpreadsheet,
  X, Loader2, Trash2, ChevronDown, ChevronRight,
  Sparkles, Database, CheckCircle2, AlertTriangle, Clock,
  Info, Search,
} from "lucide-react";

const GEN_SUPPORTED_EXTS = [".pdf", ".hwp", ".hwpx", ".docx", ".pptx", ".md", ".txt"];
import { useDataStore, groupByBaseId, type DatasetItem, type DatasetFileEntry } from "../../stores/dataStore";
import { DatasetDetailModal } from "../../components/modals/DatasetDetailModal";
import { useAgentStore } from "../../stores/agentStore";
import { useSystemStore } from "../../stores/systemStore";
import { toast } from "sonner";
import { tp, useT } from "../../i18n";
import { formatDateTime, formatNumber } from "../../i18n/locale";

/* ── Types ── */
interface SelectedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
}

/* ── Helpers ── */
function formatSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024, sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function formatDate(ts: number): string {
  return formatDateTime(ts, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function getFileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase();
  if (ext === "pdf") return <FileText className="w-4 h-4 text-neutral-500" />;
  if (ext === "json" || ext === "jsonl") return <FileCode className="w-4 h-4 text-neutral-500" />;
  if (ext === "csv" || ext === "xlsx") return <FileSpreadsheet className="w-4 h-4 text-neutral-500" />;
  return <File className="w-4 h-4 text-neutral-400" />;
}

/* ── RAG indexing badge ── */
function IndexingBadge({ indexing }: { indexing: NonNullable<DatasetItem["indexing"]> }) {
  const t = useT();
  if (indexing.status === "completed") {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-neutral-900 text-white text-[9px] font-semibold">
        <CheckCircle2 className="w-2.5 h-2.5" />
        {tp("RAG · {0} chunks", formatNumber(indexing.chunk_count))}
      </span>
    );
  }
  if (indexing.status === "indexing") {
    const pct = indexing.progress ?? 0;
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-neutral-900 text-white text-[9px] font-semibold">
        <Loader2 className="w-2.5 h-2.5 animate-spin" />
        {tp("Indexing for RAG{0}", pct ? ` · ${pct}%` : "")}
      </span>
    );
  }
  if (indexing.status === "queued") {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-neutral-200 text-neutral-600 text-[9px] font-semibold">
        <Clock className="w-2.5 h-2.5" />
        {t("Awaiting RAG")}
      </span>
    );
  }
  // failed
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-red-50 text-red-600 text-[9px] font-semibold" title={indexing.error ?? ""}>
      <AlertTriangle className="w-2.5 h-2.5" />
      {t("RAG indexing failed")}
    </span>
  );
}

/* ── Data count badge ── */
function DataCountBadge({ count }: { count: number }) {
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded-md text-[9px] font-semibold bg-neutral-100 text-neutral-700">
      SFT {formatNumber(count)}
    </span>
  );
}

function DatasetFiles({
  entries, onDeleteFile,
}: {
  entries: DatasetFileEntry[];
  onDeleteFile: (name: string) => void;
}) {
  const t = useT();
  if (entries.length === 0) {
    return <p className="px-4 py-3 text-[11px] text-neutral-400">{t("No files")}</p>;
  }
  return (
    <div className="bg-neutral-50/60 border-t border-neutral-100 px-4 py-1.5">
      {entries.map((f) => (
        <div key={f.name}
          className="grid grid-cols-[10px_minmax(0,1fr)_84px_28px] items-center gap-3 py-1.5 border-t border-neutral-100 first:border-t-0">
          <span
            aria-hidden
            className={`w-[7px] h-[7px] rounded-full justify-self-center box-border ${
              f.active
                ? "bg-neutral-900 ring-2 ring-neutral-900/15"
                : "border-[1.5px] border-neutral-300"
            }`}
          />
          <span
            className={`text-[11px] font-mono truncate ${f.active ? "text-neutral-900 font-semibold" : "text-neutral-500"}`}
            title={f.active ? t("Used for training") : f.name}
          >
            {f.name}
          </span>
          <span className="text-[10.5px] text-neutral-400 tabular-nums">{formatSize(f.size)}</span>
          <button onClick={() => onDeleteFile(f.name)}
            title={t("Delete this file only")}
            aria-label={t("Delete this file only")}
            className="inline-flex items-center justify-center w-6 h-6 rounded-md text-neutral-400 hover:text-white hover:bg-red-500 transition-colors">
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      ))}
    </div>
  );
}

/* ── Dataset list item ── */
/** Column definitions — header and rows read the same constant, or the table breaks. */
export const DATA_COLS =
  "grid grid-cols-[minmax(0,1fr)_84px_92px_110px_128px_112px] items-center gap-3";

const STAGE_LABEL: Record<string, string> = {
  raw: "① Source",
  training: "② Datasets",
};

export function DatasetEntry({
  item, onDelete, onOpenDetail, onDeleteFile, extraAction, indent,
}: {
  item: DatasetItem; onDelete?: (id: string) => void;
  onDeleteFile?: (item: DatasetItem, fileName: string) => void;
  onOpenDetail?: (item: DatasetItem) => void;
  extraAction?: React.ReactNode;
  /** Indent a child row of a version group. */
  indent?: boolean;
}) {
  const t = useT();

  const isRaw = item.type === "raw";
  // File drawer, collapsed by default. Raw source documents have no drawer.
  const entries = item.file_entries ?? [];
  const canExpand = !!onDeleteFile && !isRaw && entries.length > 0;
  const [filesOpen, setFilesOpen] = useState(false);

  return (
    <div className="border-t border-neutral-50">
    <div className={`${DATA_COLS} px-4 py-2.5 hover:bg-neutral-50/70 transition-colors group`}>
      {/* Name */}
      <div className={`flex items-center gap-2.5 min-w-0 ${indent ? "pl-6" : ""}`}>
        {canExpand ? (
          <button onClick={() => setFilesOpen((v) => !v)}
            aria-expanded={filesOpen}
            title={filesOpen ? t("Collapse files") : t("Expand files")}
            className="w-7 h-7 rounded-lg bg-neutral-50 border border-neutral-100 flex items-center justify-center flex-shrink-0 text-neutral-400 hover:text-neutral-900 hover:border-neutral-300 transition-colors">
            <ChevronRight className={`w-3.5 h-3.5 transition-transform ${filesOpen ? "rotate-90" : ""}`} />
          </button>
        ) : (
        <div className="w-7 h-7 rounded-lg bg-neutral-50 border border-neutral-100 flex items-center justify-center flex-shrink-0">
          <FileText className="w-3.5 h-3.5 text-neutral-400" />
        </div>
        )}
        <div className="min-w-0">
          <p className="text-[12.5px] font-medium text-neutral-900 truncate" title={item.name}>{item.name}</p>
          <p className="text-[9.5px] text-neutral-400 mt-0.5">
            {t(STAGE_LABEL[item.type] ?? item.type)} · {item.format?.toUpperCase()}
          </p>
        </div>
      </div>

      {/* Size */}
      <span className="text-[11.5px] text-neutral-500 tabular-nums">{formatSize(item.size)}</span>

      {/* Count */}
      <span className="text-[11.5px] text-neutral-500 tabular-nums">
        {typeof item.qa_count === "number" && item.qa_count > 0 ? (
          <DataCountBadge count={item.qa_count} />
        ) : "—"}
      </span>

      {/* Status */}
      <div className="flex items-center gap-1 flex-wrap min-w-0">
        {item.indexing && <IndexingBadge indexing={item.indexing} />}
      </div>

      {/* Updated */}
      <span className="text-[11px] text-neutral-400 tabular-nums">{formatDate(item.created_at)}</span>

      {/* Actions */}
      <div className="flex items-center justify-end gap-1">
        {extraAction}
        {onOpenDetail && (
          <button onClick={() => onOpenDetail(item)}
            title={t("Details")}
            className="inline-flex items-center justify-center w-7 h-7 rounded-md text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 transition-colors">
            <Info className="w-3.5 h-3.5" />
          </button>
        )}
        {onDelete && (
          <button onClick={() => onDelete(item.id)}
            title={t("Delete")}
            className="inline-flex items-center justify-center w-7 h-7 rounded-md text-red-500 bg-red-50 hover:text-white hover:bg-red-500 transition-colors">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
    {canExpand && filesOpen && (
      <DatasetFiles entries={entries} onDeleteFile={(name) => onDeleteFile!(item, name)} />
    )}
    </div>
  );
}

/* ── Dataset table ──
   Three summary cards plus one table. The stages are numbers in the summary,
   and the list is one table with aligned columns to scan. */

/** Hold the same height as the real layout during the first load.
 *  Showing a "Loading" line and an empty table at the same time makes the
 *  screen contradict itself, and the content jumps when the line disappears. */
function DatasetListSkeleton() {
  const t = useT();
  return (
    <div className="space-y-6 animate-pulse" aria-busy="true" aria-label={t("Loading datasets")}>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-xl px-4 py-3 border border-neutral-100 bg-neutral-50">
            <div className="h-2.5 w-20 rounded bg-neutral-200" />
            <div className="h-6 w-10 rounded bg-neutral-200 mt-2.5" />
            <div className="h-2 w-24 rounded bg-neutral-100 mt-2" />
          </div>
        ))}
      </div>

      <div className="flex items-center gap-1.5">
        {[56, 48, 64, 56].map((w, i) => (
          <div key={i} className="h-7 rounded-full bg-neutral-100" style={{ width: w }} />
        ))}
        <div className="ml-auto h-7 w-[200px] rounded-full bg-neutral-100" />
      </div>

      <div className="bg-white border border-neutral-200/60 rounded-xl overflow-hidden shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
        <div className={`${DATA_COLS} px-4 py-2.5 bg-neutral-50/70`}>
          {[28, 24, 24, 24, 36, 24].map((w, i) => (
            <div key={i} className="h-2 rounded bg-neutral-200" style={{ width: w }} />
          ))}
        </div>
        {[0, 1, 2, 3].map((r) => (
          <div key={r} className={`${DATA_COLS} px-4 py-2.5 border-t border-neutral-50`}>
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-neutral-100 flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="h-2.5 rounded bg-neutral-100" style={{ width: `${70 - r * 12}%` }} />
                <div className="h-2 w-16 rounded bg-neutral-50 mt-1.5" />
              </div>
            </div>
            <div className="h-2.5 w-12 rounded bg-neutral-100" />
            <div className="h-2.5 w-10 rounded bg-neutral-100" />
            <div className="h-4 w-16 rounded-full bg-neutral-100" />
            <div className="h-2.5 w-14 rounded bg-neutral-100" />
            <div className="flex justify-end gap-1">
              <div className="w-7 h-7 rounded-md bg-neutral-100" />
              <div className="w-7 h-7 rounded-md bg-neutral-100" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

type StageKey = "all" | "raw" | "dataset";

function DatasetTable({
  items, expanded, onToggleExpand,
  onDelete, onOpenDetail, onDeleteFile,
  emptyText, refreshing,
}: {
  items: DatasetItem[];
  expanded: Set<string>;
  onToggleExpand: (baseId: string) => void;
  onDelete: (item: DatasetItem) => void;
  onOpenDetail: (item: DatasetItem) => void;
  onDeleteFile: (item: DatasetItem, fileName: string) => void;
  emptyText: string;
  /** A polling refresh is in flight, shown only as a top bar that takes no space. */
  refreshing?: boolean;
}) {
  const t = useT();
  const groups = groupByBaseId(items);
  const hasAny = groups.length > 0;

  return (
    <div className="relative bg-white border border-neutral-200/60 rounded-xl overflow-hidden shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
      {refreshing && (
        <div className="absolute inset-x-0 top-0 h-0.5 overflow-hidden" aria-hidden>
          <div className="h-full w-1/3 bg-neutral-300"
            style={{ animation: "pipeline-flow 1.4s ease-in-out infinite" }} />
        </div>
      )}
      <div className={`${DATA_COLS} px-4 py-2.5 bg-neutral-50/70 text-[9.5px] font-bold uppercase tracking-[0.1em] text-neutral-400`}>
        <span>{t("Name")}</span>
        <span>{t("Size")}</span>
        <span>{t("Count")}</span>
        <span>{t("Status")}</span>
        <span>{t("Update")}</span>
        <span className="text-right">{t("Actions")}</span>
      </div>

      {!hasAny ? (
        <div className="flex flex-col items-center justify-center py-14 gap-2 text-neutral-300">
          <Database className="w-5 h-5 opacity-40" />
          <p className="text-[12px] font-medium">{emptyText}</p>
        </div>
      ) : (
        <div>
          {groups.map((g) => {
            if (g.versions.length === 1) {
              const item = g.versions[0];
              return (
                <DatasetEntry
                  key={item.id}
                  item={item}
                  onDelete={() => onDelete(item)}
                  onOpenDetail={onOpenDetail}
                  onDeleteFile={onDeleteFile}
                />
              );
            }
            const open = expanded.has(g.base_id);
            const latest = g.versions[0];
            return (
              <React.Fragment key={g.base_id}>
                <DatasetEntry
                  item={latest}
                  onDelete={() => onDelete(latest)}
                  onOpenDetail={onOpenDetail}
                  onDeleteFile={onDeleteFile}
                  extraAction={
                    <button
                      onClick={() => onToggleExpand(g.base_id)}
                      title={open ? t("Collapse versions") : tp("Show {0} versions", g.versions.length)}
                      className="inline-flex items-center gap-1 h-7 px-2 rounded-md bg-neutral-100 text-[10.5px] font-bold text-neutral-600 hover:bg-neutral-200 transition-colors"
                    >
                      v{g.versions.length}
                      <ChevronDown className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} />
                    </button>
                  }
                />
                {open && g.versions.slice(1).map((v) => (
                  <DatasetEntry
                    key={v.id}
                    item={v}
                    indent
                    onDelete={() => onDelete(v)}
                      onOpenDetail={onOpenDetail}
                    onDeleteFile={onDeleteFile}
                  />
                ))}
              </React.Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════
   Main Component
   ══════════════════════════════════════════ */
export default function GuidedDataView() {
  const t = useT();
  const rawDatasets = useDataStore((s) => s.rawDatasets);
  const trainingDatasets = useDataStore((s) => s.trainingDatasets);
  const fetchDatasets = useDataStore((s) => s.fetchDatasets);
  const uploadStoreFiles = useDataStore((s) => s.uploadFiles);
  const deleteDataset = useDataStore((s) => s.deleteDataset);
  const deleteDatasetFile = useDataStore((s) => s.deleteDatasetFile);
  const uploadProgress = useDataStore((s) => s.uploadProgress);
  const isUploading = useDataStore((s) => s.isUploading);
  const isLoadingDatasets = useDataStore((s) => s.isLoadingDatasets);
  const dispatchMessage = useAgentStore((s) => s.dispatchMessage);
  const agentAutonomy = useSystemStore((s) => s.systemSettings.agentAutonomy);

  // Distinguish the first load from a polling refresh. fetchDatasets sets
  // isLoadingDatasets on every call (every 15s, or 5s while busy), so reading
  // that alone makes the loading indicator blink on an empty account.
  const [firstLoadDone, setFirstLoadDone] = useState(false);
  const sawLoadingRef = useRef(false);
  useEffect(() => {
    if (isLoadingDatasets) sawLoadingRef.current = true;
    else if (sawLoadingRef.current) setFirstLoadDone(true);
  }, [isLoadingDatasets]);

  // List filter — the summary cards and the tabs share this state.
  const [stageFilter, setStageFilter] = useState<StageKey>("all");
  const [query, setQuery] = useState("");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const toggleGroup = (baseId: string) =>
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(baseId)) next.delete(baseId);
      else next.add(baseId);
      return next;
    });

  // Shared delete handler for both categories. The category arg pins the
  // backend deletion to the bucket the user pointed at.
  // Dataset detail modal (open = item, closed = null).
  const [detailItem, setDetailItem] = useState<DatasetItem | null>(null);
  const openDetailBrowse = useCallback((it: DatasetItem) => setDetailItem(it), []);

  const handleDeleteDataset = useCallback(async (
    id: string,
    category: "raw" | "training",
  ) => {
    if (!window.confirm(tp("Delete the dataset '{0}'?", id))) return;
    try {
      await deleteDataset(id, category);
      toast.success(t("Deleted"), { description: id });
    } catch (e) {
      toast.error(tp("Delete failed: {0}", (e as Error).message));
    }
  }, [deleteDataset, t]);

  const handleDeleteFile = useCallback(async (item: DatasetItem, fileName: string) => {
    if (!window.confirm(tp("Delete the file '{0}'?", fileName))) return;
    try {
      const { trainingReady } = await deleteDatasetFile(item.id, fileName);
      toast.success(t("File deleted"), {
        description: trainingReady
          ? tp("{0} · the remaining data can still be trained on", fileName)
          : tp("{0} · no trainable data left in this dataset", fileName),
      });
    } catch (e) {
      toast.error(tp("Failed to delete file: {0}", (e as Error).message));
    }
  }, [deleteDatasetFile, t]);

  // Local file staging state (files are staged first, then uploaded on button click)
  const [genFiles, setGenFiles] = useState<SelectedFile[]>([]);
  const [trainFiles, setTrainFiles] = useState<SelectedFile[]>([]);
  const [genFolderName, setGenFolderName] = useState<string | null>(null);
  const [genExcluded, setGenExcluded] = useState<{ count: number; exts: string[] } | null>(null);
  const [genDragOver, setGenDragOver] = useState(false);
  const [trainDragOver, setTrainDragOver] = useState(false);
  const [openSection, setOpenSection] = useState<"gen" | "train" | null>("gen");
  const genOpen = openSection === "gen";
  const trainOpen = openSection === "train";

  const genFileRef = useRef<HTMLInputElement>(null);
  const genFolderRef = useRef<HTMLInputElement>(null);
  const trainFileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchDatasets(); }, [fetchDatasets]);

  // Background poll (15s).
  useEffect(() => {
    const id = window.setInterval(() => { void fetchDatasets(); }, 15000);
    return () => window.clearInterval(id);
  }, [fetchDatasets]);

  // Faster polling (5s) while indexing is in flight.
  useEffect(() => {
    const idxing = rawDatasets.some(
      (d) => d.indexing && (d.indexing.status === "indexing" || d.indexing.status === "queued"),
    );
    if (!idxing) return;
    const id = window.setInterval(() => { void fetchDatasets(); }, 5000);
    return () => window.clearInterval(id);
  }, [rawDatasets, fetchDatasets]);

  /* ── File selection handlers ── */
  const addFiles = (fileList: FileList | File[], target: "gen" | "train") => {
    const arr = Array.from(fileList);
    const toItem = (f: File, i: number): SelectedFile => ({
      id: `${Date.now()}_${i}_${Math.random().toString(36).slice(2, 6)}`,
      name: f.name, // basename for display (folder name tracked separately)
      size: f.size,
      type: f.type,
      file: f,
    });

    if (target === "gen") {
      // For a folder pick, the first segment of webkitRelativePath is the folder name.
      const relPath = arr
        .map((f) => (f as any).webkitRelativePath as string | undefined)
        .find(Boolean);
      const folderName = relPath ? relPath.split("/")[0] : null;

      // Only supported extensions are included; the rest are summarised as excluded.
      const accepted: SelectedFile[] = [];
      const excludedExts = new Set<string>();
      let excludedCount = 0;
      arr.forEach((f, i) => {
        const lower = f.name.toLowerCase();
        if (GEN_SUPPORTED_EXTS.some((e) => lower.endsWith(e))) {
          accepted.push(toItem(f, i));
        } else {
          excludedCount += 1;
          const dot = f.name.lastIndexOf(".");
          excludedExts.add(dot >= 0 ? f.name.slice(dot).toLowerCase() : t("(no extension)"));
        }
      });

      if (folderName) setGenFolderName(folderName);
      setGenExcluded(
        excludedCount > 0 ? { count: excludedCount, exts: Array.from(excludedExts).slice(0, 5) } : null,
      );
      if (accepted.length === 0) {
        toast.error(folderName ? tp("'{0}' contains no supported files.", folderName) : t("No supported files."));
      }
      setGenFiles((prev) => [...prev, ...accepted]);
    } else {
      // Training data: JSON and JSONL only.
      const valid = arr
        .filter((f) => f.name.endsWith(".json") || f.name.endsWith(".jsonl"))
        .map((f, i) => toItem(f, i));
      setTrainFiles((prev) => [...prev, ...valid]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, target: "gen" | "train") => {
    if (e.target.files) addFiles(e.target.files, target);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent, target: "gen" | "train") => {
    e.preventDefault();
    if (target === "gen") setGenDragOver(false); else setTrainDragOver(false);
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files, target);
  };

  const removeFile = (id: string, target: "gen" | "train") => {
    if (target === "gen") {
      setGenFiles((prev) => {
        const next = prev.filter((f) => f.id !== id);
        if (next.length === 0) { setGenFolderName(null); setGenExcluded(null); }
        return next;
      });
    } else {
      setTrainFiles((prev) => prev.filter((f) => f.id !== id));
    }
  };

  /* ── Upload handler (staged files → API) ── */
  const handleUpload = async (target: "gen" | "train") => {
    const files = target === "gen" ? genFiles : trainFiles;
    if (files.length === 0 || isUploading) return;

    // gen → "raw" (documents) · train → "processed" (QA dataset)
    const backendTarget: "raw" | "processed" = target === "gen" ? "raw" : "processed";

    let folder: string;
    let agentHandoff: string | null = null;
    try {
      ({ folder, agentHandoff } = await uploadStoreFiles(
        files.map((f) => f.file),
        backendTarget,
        target === "gen" ? genFolderName : null,
      ));
    } catch (e) {
      toast.error((e as Error).message);
      return;
    }

    // Clear staged files after upload
    if (target === "gen") { setGenFiles([]); setGenFolderName(null); setGenExcluded(null); }
    else setTrainFiles([]);

    // Guided mode: hand the upload over to the agent panel. The wording comes
    // from the server so the instruction matches the backend contract.
    if (folder && agentAutonomy === "guided" && agentHandoff) {
      void dispatchMessage(agentHandoff, target === "gen" ? "retrieval" : "tuning");
    }

  };

  /* ── Upload section renderer ── */
  const renderUploadSection = (target: "gen" | "train") => {
    const files = target === "gen" ? genFiles : trainFiles;
    const isDragOver = target === "gen" ? genDragOver : trainDragOver;
    const setDrag = target === "gen" ? setGenDragOver : setTrainDragOver;
    const inputRef = target === "gen" ? genFileRef : trainFileRef;
    const accept = target === "gen" ? ".pdf,.hwp,.hwpx,.docx,.pptx,.md,.txt" : ".json,.jsonl";
    const supportText = target === "gen" ? t("PDF, HWP, DOCX, TXT, MD and more") : t("JSON and JSONL only");

    return (
      <div className="px-5 pb-5 space-y-3">
        {/* Hidden inputs */}
        <input ref={inputRef} type="file" multiple className="hidden" accept={accept}
          onChange={(e) => handleFileChange(e, target)} />
        {target === "gen" && (
          <input ref={genFolderRef} type="file" className="hidden"
            {...{ webkitdirectory: "" } as any}
            onChange={(e) => handleFileChange(e, "gen")} />
        )}

        {/* Drop zone */}
        <div
          onClick={() => !isUploading && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => handleDrop(e, target)}
          className={`h-24 border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-1.5 cursor-pointer transition-all duration-200 ${
            isDragOver
              ? "border-neutral-900 bg-gradient-to-b from-neutral-100 to-neutral-50 shadow-inner"
              : "border-neutral-200 hover:border-neutral-400 hover:bg-gradient-to-b hover:from-neutral-50 hover:to-white"
          }`}
        >
          <div className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${isDragOver ? "bg-neutral-900" : "bg-neutral-100"}`}>
            <UploadCloud className={`w-4 h-4 transition-colors ${isDragOver ? "text-white" : "text-neutral-400"}`} />
          </div>
          <span className="text-[12px] font-medium text-neutral-500">
            {isDragOver ? t("Drop here") : t("Click or drag files here")}
          </span>
          <span className="text-[10px] text-neutral-300">{supportText}</span>
        </div>

        {/* File / folder select (generation only) — secondary button tone */}
        {target === "gen" && (
          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => genFileRef.current?.click()} disabled={isUploading}
              className="flex items-center justify-center gap-2 py-2 rounded-lg border border-neutral-200 text-[12px] font-medium text-neutral-600 hover:bg-neutral-50 hover:border-neutral-400 transition-all disabled:opacity-40">
              <FileText className="w-3.5 h-3.5" />
              {t("Choose a file")}
            </button>
            <button onClick={() => genFolderRef.current?.click()} disabled={isUploading}
              className="flex items-center justify-center gap-2 py-2 rounded-lg border border-neutral-200 text-[12px] font-medium text-neutral-600 hover:bg-neutral-50 hover:border-neutral-400 transition-all disabled:opacity-40">
              <FolderUp className="w-3.5 h-3.5" />
              {t("Select a folder")}
            </button>
          </div>
        )}

        {/* Notice about excluded unsupported files */}
        {target === "gen" && genExcluded && (
          <div className="flex items-center gap-1.5 text-[11px] text-neutral-400 animate-in fade-in duration-200">
            <Info className="w-3 h-3 shrink-0" />
            {tp("{0} excluded ({1})", genExcluded.count, genExcluded.exts.join(", "))}
          </div>
        )}

        {target === "train" && (
          <div className="rounded-lg border border-neutral-100 bg-neutral-50 px-3.5 py-2.5">
            <p className="mb-1.5 text-[9.5px] font-semibold uppercase tracking-wider text-neutral-400">{t("Supported formats")}</p>
            <ul className="space-y-1">
              <li className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span className="w-2 shrink-0 text-[11px] text-neutral-400">•</span>
                <code className="rounded bg-neutral-100 px-1 py-px font-mono text-[10.5px] text-neutral-700">{"{question, answer}"}</code>
                <span className="text-[10px] text-neutral-300">→</span>
                <span className="text-[11px] text-neutral-500">SFT</span>
              </li>
            </ul>
            <p className="mt-1.5 flex items-center gap-1 text-[10px] text-neutral-400">
              <Info className="h-3 w-3 shrink-0" />
              {t("Any other key structure is rejected")}
            </p>
          </div>
        )}

        {/* Staged file list */}
        {files.length > 0 && (
          <div className="space-y-1.5 animate-in fade-in slide-in-from-top-1 duration-200">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[12px] font-semibold text-neutral-500 whitespace-nowrap">{tp("Selected files ({0})", files.length)}</span>
              <span className="text-[11px] text-neutral-300 tabular-nums whitespace-nowrap">
                {formatSize(files.reduce((s, f) => s + f.size, 0))}
              </span>
            </div>
            <div className="max-h-36 overflow-y-auto rounded-lg border border-neutral-100 divide-y divide-neutral-50">
              {files.map((f) => (
                <div key={f.id} className="flex items-center gap-2 px-3 py-2 hover:bg-neutral-50/50 transition-colors">
                  {getFileIcon(f.name)}
                  <span className="truncate text-[12px] font-medium text-neutral-700 flex-1">{f.name}</span>
                  <span className="text-[10px] font-normal text-neutral-300 tabular-nums whitespace-nowrap">{formatSize(f.size)}</span>
                  <button onClick={() => removeFile(f.id, target)}
                    className="p-1 rounded text-neutral-300 hover:text-neutral-600 hover:bg-neutral-100">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload progress */}
        {isUploading && (
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-neutral-500 font-medium">{t("Uploading")}</span>
              <span className="text-neutral-900 font-bold tabular-nums">{uploadProgress}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-neutral-100 overflow-hidden">
              <div className="h-full rounded-full bg-neutral-900 transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}

        {/* Upload button */}
        <button
          onClick={() => handleUpload(target)}
          disabled={files.length === 0 || isUploading}
          className="w-full h-10 rounded-lg bg-neutral-900 hover:bg-neutral-800 text-white text-[13px] font-semibold transition-all disabled:opacity-40 flex items-center justify-center gap-2"
        >
          {isUploading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> {t("Uploading...")}</>
            : <><Upload className="w-4 h-4" /> {t("Upload")}{files.length > 0 ? tp(" ({0})", files.length) : ""}</>
          }
        </button>
      </div>
    );
  };

  const allItems: DatasetItem[] = [...rawDatasets, ...trainingDatasets];
  const matchesQuery = (name: string) =>
    !query.trim() || name.toLowerCase().includes(query.trim().toLowerCase());
  const inBucket = (d: DatasetItem) =>
    stageFilter === "all" || (stageFilter === "raw" ? d.type === "raw" : d.type === "training");
  const visibleItems = allItems
    .filter(inBucket)
    .filter((d) => matchesQuery(d.name))
    .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));

  const indexingCount = rawDatasets.filter((d) => d.indexing?.status === "indexing").length;

  const STAGE_CARDS = [
    {
      key: "raw" as StageKey, label: t("① Source documents"), icon: Sparkles,
      count: () => rawDatasets.length,
      note: () => (indexingCount > 0 ? tp("{0} indexing", indexingCount) : t("RAG · KBD source")),
    },
    {
      key: "dataset" as StageKey, label: t("② Datasets"), icon: Database,
      count: () => trainingDatasets.length,
      note: () => t("Use for training directly"),
    },
  ];

  const STAGE_TABS: { key: StageKey; label: string; count: () => number }[] = [
    { key: "all", label: t("All"), count: () => allItems.length },
    { key: "raw", label: t("Source"), count: () => rawDatasets.length },
    { key: "dataset", label: t("Datasets"), count: () => trainingDatasets.length },
  ];

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Page Header */}
      <div>
        <p className="text-[10px] font-bold text-neutral-300 uppercase tracking-[0.2em] mb-2">Data Agent</p>
        <h2 className="text-[28px] font-black text-neutral-900 tracking-tight">{t("Data management")}</h2>
        <p className="text-[13px] text-neutral-400 mt-1">{t("Upload documents and run each step manually.")}</p>
      </div>

      {/* ── Upload section 1: for generation ── */}
      <div className="bg-white border border-neutral-100 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] overflow-hidden">
        <button onClick={() => setOpenSection(genOpen ? null : "gen")}
          className="w-full flex items-center justify-between px-6 py-4 hover:bg-neutral-50/50 transition-colors">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-neutral-50 border border-neutral-100 flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5 text-neutral-500" />
            </div>
            <div className="text-left">
              <span className="text-[13px] font-semibold text-neutral-700">{t("Want to generate data from documents?")}</span>
              <span className="text-[11px] text-neutral-300 ml-2">PDF, DOCX, HWP, TXT, MD</span>
            </div>
          </div>
          <ChevronDown className={`w-4 h-4 text-neutral-300 transition-transform duration-200 ${genOpen ? "rotate-180" : ""}`} />
        </button>
        {genOpen && (
          <div className="border-t border-neutral-100">
            {renderUploadSection("gen")}
          </div>
        )}
      </div>

      {/* ── Upload section 2: for training ── */}
      <div className="bg-white border border-neutral-100 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.04)] overflow-hidden">
        <button onClick={() => setOpenSection(trainOpen ? null : "train")}
          className="w-full flex items-center justify-between px-6 py-4 hover:bg-neutral-50/50 transition-colors">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-neutral-50 border border-neutral-100 flex items-center justify-center">
              <Database className="w-3.5 h-3.5 text-neutral-400" />
            </div>
            <div className="text-left">
              <span className="text-[13px] font-semibold text-neutral-600">{t("Already have a generated QA dataset?")}</span>
              <span className="text-[11px] text-neutral-300 ml-2">{t("JSON, JSONL — uploaded as awaiting preprocessing / TWIST")}</span>
            </div>
          </div>
          <ChevronDown className={`w-4 h-4 text-neutral-300 transition-transform duration-200 ${trainOpen ? "rotate-180" : ""}`} />
        </button>
        {trainOpen && (
          <div className="border-t border-neutral-100">
            {renderUploadSection("train")}
          </div>
        )}
      </div>

      {/* ── Dataset list — summary plus one table ── */}
      {!firstLoadDone ? <DatasetListSkeleton /> : <>

      {/* Per-stage summary; clicking one narrows the table below */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {STAGE_CARDS.map((c) => {
          const on = stageFilter === c.key;
          const Icon = c.icon;
          return (
            <button key={c.key} onClick={() => setStageFilter(on ? "all" : c.key)}
              className={`text-left rounded-xl px-4 py-3 border transition-colors ${
                on ? "bg-neutral-900 border-neutral-900" : "bg-neutral-50 border-neutral-100 hover:bg-neutral-100"
              }`}>
              <span className={`flex items-center gap-1.5 text-[10.5px] font-bold ${on ? "text-white/60" : "text-neutral-500"}`}>
                <Icon className="w-3 h-3" />
                {c.label}
              </span>
              <p className={`text-[24px] font-black tracking-[-0.04em] tabular-nums mt-1.5 ${on ? "text-white" : "text-neutral-900"}`}>
                {c.count()}
              </p>
              <p className={`text-[10.5px] mt-0.5 ${on ? "text-white/45" : "text-neutral-400"}`}>{c.note()}</p>
            </button>
          );
        })}
      </div>

      {/* Filter tabs and search */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {STAGE_TABS.map((t) => (
          <button key={t.key} onClick={() => setStageFilter(t.key)}
            className={`px-3 py-1.5 rounded-full text-[11.5px] font-bold border transition-colors ${
              stageFilter === t.key
                ? "bg-neutral-900 text-white border-neutral-900"
                : "bg-neutral-50 text-neutral-500 border-neutral-100 hover:bg-neutral-100"
            }`}>
            {t.label} {t.count()}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 px-3 py-1.5 rounded-full border border-neutral-200 min-w-[200px]">
          <Search className="w-3.5 h-3.5 text-neutral-400 flex-shrink-0" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("Search by name")}
            className="flex-1 min-w-0 text-[11.5px] outline-none placeholder:text-neutral-300"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-neutral-300 hover:text-neutral-600">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <DatasetTable
        items={visibleItems}
        expanded={expandedGroups}
        onToggleExpand={toggleGroup}
        onDelete={(item) => handleDeleteDataset(item.id, item.type)}
        onOpenDetail={openDetailBrowse}
        onDeleteFile={handleDeleteFile}
        emptyText={query ? tp('No dataset matches "{0}"', query) : t("No datasets")}
        refreshing={isLoadingDatasets}
      />

      </>}

      <DatasetDetailModal
        item={detailItem}
        open={detailItem !== null}
        clickedVersion={detailItem?.version}
        totalVersions={(() => {
          if (!detailItem) return 1;
          // How many generations share this base_id within the same section.
          const list = detailItem.type === "raw" ? rawDatasets : trainingDatasets;
          const base = detailItem.base_id ?? detailItem.name;
          return list.filter((d) => (d.base_id ?? d.name) === base).length;
        })()}
        onClose={() => setDetailItem(null)}
      />

    </div>
  );
}
