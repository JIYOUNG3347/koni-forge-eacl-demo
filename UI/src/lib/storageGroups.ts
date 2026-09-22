/**
 * Group the categories from GET /api/system/storage into three display groups.
 *
 * The backend reports nine directories individually (raw_corpus, models, logs, ...).
 * What a user wants to know is "what is using the disk", so they are grouped by purpose.
 * Pure functions — testable without React.
 */

import { formatNumber } from "../i18n/locale";

export interface StorageCategory {
  size_mb?: number;
}

export interface StorageGroup {
  key: string;
  label: string;
  gb: number;
  /** Share of disk usage, in percent. */
  pct: number;
}

export interface StorageBreakdown {
  groups: StorageGroup[];
  /** Disk usage not accounted for by KONI storage (the OS, other applications). */
  otherGb: number;
  /** Free space. */
  freeGb: number;
}

const GROUP_DEFS: { key: string; label: string; dirs: string[] }[] = [
  { key: "dataset", label: "Datasets", dirs: ["raw_corpus", "corpus", "chroma"] },
  { key: "model", label: "Checkpoint", dirs: ["models", "checkpoints"] },
  { key: "output", label: "Logs · artifacts", dirs: ["outputs", "logs"] },
];

function gb(mb: number): number {
  return mb / 1024;
}

/**
 * @param categories the categories from /api/system/storage (a missing key counts as 0)
 * @param diskUsedGb disk.used_gb — anything above the group total becomes 'Other'
 * @param diskFreeGb disk.free_gb
 */
export function groupStorage(
  categories: Record<string, StorageCategory> | null | undefined,
  diskUsedGb: number,
  diskFreeGb: number,
): StorageBreakdown {
  const cats = categories ?? {};
  const used = Math.max(0, diskUsedGb);

  const raw = GROUP_DEFS.map((def) => ({
    key: def.key,
    label: def.label,
    gb: def.dirs.reduce((sum, d) => sum + gb(cats[d]?.size_mb ?? 0), 0),
  }));

  const known = raw.reduce((sum, g) => sum + g.gb, 0);
  // Percentages are taken against disk usage. If the storage total is larger
  // (measurement skew), use the storage total as the denominator so nothing exceeds 100%.
  const denom = Math.max(used, known) || 1;

  return {
    groups: raw.map((g) => ({ ...g, pct: (g.gb / denom) * 100 })),
    otherGb: Math.max(0, used - known),
    freeGb: Math.max(0, diskFreeGb),
  };
}

/** Display form: no decimals, except one decimal below 1GB. */
export function formatGb(value: number): string {
  if (value >= 100) return formatNumber(Math.round(value));
  if (value >= 10) return value.toFixed(0);
  if (value >= 1) return value.toFixed(1);
  return value.toFixed(2);
}
