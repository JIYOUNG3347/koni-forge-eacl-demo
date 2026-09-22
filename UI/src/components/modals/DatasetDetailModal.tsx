/**
 * DatasetDetailModal — dataset details: metadata, source files, storage info
 * and a QA preview. Source files come from provenance when available,
 * otherwise from a direct folder read.
 */
import { useEffect, useState } from "react";
import { ChevronDown, Copy, FileText, Info, Loader2 } from "lucide-react";
import { toast } from "sonner";

import {
  useDataStore,
  type DatasetFile,
  type DatasetFilesResponse,
  type DatasetItem,
  type ProvenanceResponse,
  type QaPreviewItem,
  type StorageInfoResponse,
} from "../../stores/dataStore";
import { ModalShell } from "./ModalShell";
import { tp, useT } from "../../i18n";
import { formatDateTime, formatNumber } from "../../i18n/locale";

function formatSize(bytes: number): string {
  if (!bytes) return "0 B";
  const k = 1024;
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${units[i]}`;
}

function formatDate(ts: number | string | undefined): string {
  if (ts === undefined || ts === null || ts === "") return "—";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return "—";
  return formatDateTime(d, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Storage information timestamp format.
function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return formatDateTime(d, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-100 bg-neutral-50 px-3.5 py-3">
      <div className="text-[9px] font-bold uppercase tracking-[.1em] text-neutral-400">{label}</div>
      <div className="mt-1.5 text-[22px] font-bold tabular-nums leading-none text-neutral-900">{value}</div>
    </div>
  );
}

interface Props {
  item: DatasetItem | null;
  open: boolean;
  onClose: () => void;
  /** Number of versions of this dataset (base_id); >1 shows the version chip. */
  totalVersions?: number;
  /** Version the user clicked in the list (header chip). */
  clickedVersion?: number;
}

export function DatasetDetailModal({
  item,
  open,
  onClose,
  totalVersions = 1,
  clickedVersion,
}: Props) {
  const t = useT();
  const fetchProvenance = useDataStore((s) => s.fetchProvenance);
  const fetchStorageInfo = useDataStore((s) => s.fetchStorageInfo);
  const fetchDatasetFiles = useDataStore((s) => s.fetchDatasetFiles);
  const previewDataset = useDataStore((s) => s.previewDataset);
  const [prov, setProv] = useState<ProvenanceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  const [qaItems, setQaItems] = useState<QaPreviewItem[]>([]);
  const [qaTotal, setQaTotal] = useState(0);
  const [qaLoading, setQaLoading] = useState(false);
  const [qaPage, setQaPage] = useState(0);
  // Storage information: fetched once when the modal opens.
  const [storage, setStorage] = useState<StorageInfoResponse | null>(null);
  // Source files read directly from the folder — fallback when provenance is absent.
  const [folderFiles, setFolderFiles] = useState<DatasetFilesResponse | null>(null);
  const [filesLoading, setFilesLoading] = useState(false);
  // Keep the header chip in sync with the version shown.
  const [currentVersion, setCurrentVersion] = useState<number | undefined>(clickedVersion);

  useEffect(() => {
    if (!open || !item) return;
    let cancelled = false;
    setProv(null);
    setLoading(true);
    setStorage(null);
    setFolderFiles(null);
    setFilesLoading(true);
    setCurrentVersion(clickedVersion);
    setFilesOpen(false);
    setQaItems([]);
    setQaTotal(0);
    setQaPage(0);
    setQaLoading(item.type !== "raw");
    fetchStorageInfo(item.id)
      .then((r) => {
        if (!cancelled) setStorage(r);
      })
      .catch(() => {
        if (!cancelled) setStorage(null);
      });
    fetchProvenance(item.id)
      .then((r) => {
        if (!cancelled) setProv(r);
      })
      .catch(() => {
        if (!cancelled) setProv({ dataset_id: item.id, available: false, reason: "error" });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    fetchDatasetFiles(item.id)
      .then((r) => {
        if (!cancelled) setFolderFiles(r);
      })
      .catch(() => {
        if (!cancelled) setFolderFiles(null);
      })
      .finally(() => {
        if (!cancelled) setFilesLoading(false);
      });
  // QA preview: not for raw datasets.
    if (item.type !== "raw") {
      previewDataset(item.id, 20)
        .then((r) => {
          if (!cancelled) {
            setQaItems(r.items);
            setQaTotal(r.total);
            setQaLoading(false);
          }
        })
        .catch(() => {
          if (!cancelled) setQaLoading(false);
        });
    }
    return () => {
      cancelled = true;
    };
  }, [open, item, clickedVersion, fetchProvenance, fetchStorageInfo, fetchDatasetFiles, previewDataset]);

  const handleCopyPath = async () => {
    const path = storage?.folder_path;
    if (!path) return;
    try {
      await navigator.clipboard.writeText(path);
      toast.success(t("Path copied"));
    } catch {
      toast.error(t("Copy failed"));
    }
  };

  if (!item) return null;

  const available = prov?.available === true && !!prov.provenance;
  const folderLabel = available ? prov!.provenance!.source_folder_label : null;
  const chunkCount = item.indexing?.chunk_count;

  const subtitleBase = item.type === "training" ? t("Datasets") : t("Datasets for generation");

  // Source of the file list: provenance when present, otherwise a direct folder read.
  // The metadata cell and the "Source file" section read the *same* source, so they agree.
  const displayFiles: DatasetFile[] = available
    ? prov!.provenance!.source_files
    : (folderFiles?.files ?? []);
  const hasFiles = displayFiles.length > 0;
  // Section loading: provenance unresolved, or absent with a folder read in flight.
  const filesSectionLoading = loading || (!available && filesLoading);

  // Multi-version datasets show a version chip in the header.
  const versionToShow = currentVersion ?? item.version;
  const showVersionChip = typeof versionToShow === "number" && totalVersions > 1;
  const titleNode = showVersionChip ? (
    <span className="flex items-center gap-2">
      <span className="truncate">{item.name}</span>
      <span className="shrink-0 rounded-full bg-neutral-900 px-2 py-0.5 text-[10px] font-bold text-white">
        v{versionToShow}
      </span>
    </span>
  ) : (
    item.name
  );

  return (
    <ModalShell
      open={open}
      onOpenChange={(o) => { if (!o) onClose(); }}
      title={titleNode}
      subtitle={`${subtitleBase}${available ? t(" · source traceable") : ""}`}
    >
      {/* Metadata grid — three key cells plus a secondary line */}
      <div className="px-6 pt-5 pb-4">
        <div className="grid gap-3 grid-cols-3">
          {item.type === "raw" ? (
            <>
              <Stat label={t("Files")} value={hasFiles ? displayFiles.length : "—"} />
              <Stat label={t("Size")} value={storage ? formatSize(storage.disk_size_bytes) : "—"} />
              <Stat label={t("Chunks")} value={typeof chunkCount === "number" ? formatNumber(chunkCount) : "—"} />
            </>
          ) : (
            <>
              <Stat label={t("QA count")} value={typeof item.qa_count === "number" ? formatNumber(item.qa_count) : "—"} />
              <Stat label={t("Size")} value={storage ? formatSize(storage.disk_size_bytes) : "—"} />
              <Stat label={t("Format")} value="SFT" />
            </>
          )}
        </div>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 border-t border-neutral-100 pt-2.5 text-[11px] text-neutral-500">
          <span>{t("Created")} <b className="font-medium text-neutral-700">{formatDate(item.created_at)}</b></span>
          {item.type !== "raw" && hasFiles && (
            <span>{t("Source")} <b className="font-medium text-neutral-700">{tp("{0}", displayFiles.length)}</b></span>
          )}
        </div>
      </div>

      {/* Source file section */}
      <div className="border-t border-neutral-100 p-6">
        <button
          type="button"
          onClick={() => setFilesOpen((v) => !v)}
          className="flex w-full items-center justify-between"
        >
          <span className="text-[13px] font-semibold text-neutral-700">{t("Source file")}</span>
          <span className="flex items-center gap-2 text-[11px] text-neutral-400">
            {hasFiles && <span>{tp("{0}", displayFiles.length)}</span>}
            <ChevronDown className={`h-4 w-4 transition-transform ${filesOpen ? "" : "-rotate-90"}`} />
          </span>
        </button>

        {filesOpen && (
          <div className="mt-3">
            {filesSectionLoading ? (
              <div className="flex items-center justify-center gap-2 py-6 text-[12px] text-neutral-400">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> {t("Loading…")}
              </div>
            ) : hasFiles ? (
              <>
                {folderLabel && (
                  <div className="mb-2 text-[12px] text-neutral-500">
                    {t("Source folder")} <span className="font-medium text-neutral-800">{folderLabel}</span>
                  </div>
                )}
                <div className="divide-y divide-neutral-50 overflow-hidden rounded-lg border border-neutral-100">
                  {displayFiles.map((f, i) => (
                    <div key={`${f.name}_${i}`} className="flex items-center gap-2.5 px-3 py-2 transition-colors hover:bg-neutral-50">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-neutral-400" />
                      <span className="flex-1 truncate text-[13px] font-medium text-neutral-700">{f.name}</span>
                      <span className="shrink-0 text-[11px] tabular-nums text-neutral-400">{formatSize(f.size)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex items-start gap-2.5 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3">
                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-neutral-400" />
                <div className="text-[12px] leading-relaxed text-neutral-600">
                  {t("This dataset has no source files, or no provenance was recorded.")}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-neutral-100 p-6">
        <div className="mb-3 text-[13px] font-semibold text-neutral-700">{t("Storage information")}</div>
        <dl className="space-y-2.5">
          {/* Path — absolute path plus a copy button */}
          <div className="flex items-center gap-2">
            <dt className="w-24 shrink-0 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              {t("Path")}
            </dt>
            <dd className="flex min-w-0 flex-1 items-center gap-1.5">
              <span className="truncate font-mono text-[12px] text-neutral-700">
                {storage?.folder_path || "—"}
              </span>
              {storage?.folder_path && (
                <button
                  type="button"
                  onClick={handleCopyPath}
                  title={t("Copy path")}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-700"
                >
                  <Copy className="h-4 w-4" />
                  <span className="sr-only">{t("Copy path")}</span>
                </button>
              )}
            </dd>
          </div>
          {/* Disk size */}
          <div className="flex items-center gap-2">
            <dt className="w-24 shrink-0 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              {t("Disk size")}
            </dt>
            <dd className="flex-1 text-[14px] font-normal text-neutral-700">
              {storage ? formatSize(storage.disk_size_bytes) : "—"}
            </dd>
          </div>
          {/* File count */}
          <div className="flex items-center gap-2">
            <dt className="w-24 shrink-0 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              {t("File count")}
            </dt>
            <dd className="flex-1 text-[14px] font-normal text-neutral-700">
              {!storage ? "—" : storage.file_count === 0 ? t("No files") : tp("{0}", storage.file_count)}
            </dd>
          </div>
          {/* Last modified */}
          <div className="flex items-center gap-2">
            <dt className="w-24 shrink-0 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              {t("Last modified")}
            </dt>
            <dd className="flex-1 text-[14px] font-normal text-neutral-700">
              {formatTimestamp(storage?.last_modified)}
            </dd>
          </div>
        </dl>
      </div>
      {/* QA preview — training datasets only, paginated. */}
      {item.type !== "raw" && (
        <div className="border-t border-neutral-100 p-6">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-[13px] font-semibold text-neutral-700">{t("QA preview")}</div>
            {!qaLoading && qaTotal > 0 && (
              <span className="text-[11px] tabular-nums text-neutral-400">
                {tp("Total {0}", formatNumber(qaTotal))}
              </span>
            )}
          </div>
          {qaLoading ? (
            <div className="flex items-center justify-center gap-2 py-6 text-[12px] text-neutral-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> {t("Loading…")}
            </div>
          ) : qaItems.length === 0 ? (
            <div className="rounded-lg border border-neutral-100 bg-neutral-50 px-4 py-3 text-[12px] text-neutral-500">
              {t("No QA items to show.")}
            </div>
          ) : (
            <>
              {(() => {
                const cur = qaItems[qaPage];
                if (!cur) return null;
                const q = String((cur.question as string | undefined) ?? (cur.instruction as string | undefined) ?? (cur.input as string | undefined) ?? "");
                const a = String((cur.answer as string | undefined) ?? (cur.output as string | undefined) ?? "");
                return (
                  <div className="rounded-lg border border-neutral-100 px-4 py-3.5">
                    <p className="mb-1 text-[9px] font-bold uppercase tracking-[.1em] text-neutral-400">Q</p>
                    <p className="mb-3 text-[13px] leading-relaxed text-neutral-800">{q}</p>
                    <p className="mb-1 text-[9px] font-bold uppercase tracking-[.1em] text-neutral-400">A</p>
                    <p className="text-[12.5px] leading-relaxed text-neutral-600">{a}</p>
                  </div>
                );
              })()}
              {qaItems.length > 1 && (
                <div className="mt-3 flex items-center justify-between">
                  <button
                    type="button"
                    disabled={qaPage === 0}
                    onClick={() => setQaPage((p) => Math.max(0, p - 1))}
                    className="text-[11px] text-neutral-500 transition-colors disabled:text-neutral-300 hover:text-neutral-700"
                  >{t("← Previous")}</button>
                  <span className="text-[11px] tabular-nums text-neutral-400">
                    <b className="font-semibold text-neutral-700">{qaPage + 1}</b> / {qaItems.length}
                    {qaTotal > qaItems.length && (
                      <span className="ml-1">{tp("({0} total)", formatNumber(qaTotal))}</span>
                    )}
                  </span>
                  <button
                    type="button"
                    disabled={qaPage === qaItems.length - 1}
                    onClick={() => setQaPage((p) => Math.min(qaItems.length - 1, p + 1))}
                    className="text-[11px] text-neutral-500 transition-colors disabled:text-neutral-300 hover:text-neutral-700"
                  >{t("Next →")}</button>
                </div>
              )}
            </>
          )}
        </div>
      )}

    </ModalShell>
  );
}
