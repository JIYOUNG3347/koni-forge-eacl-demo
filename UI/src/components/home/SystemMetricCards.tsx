/**
 * System metric cards on the guided home page.
 *
 * GPU: three aggregate figures (mean load, peak temperature, free VRAM) plus one row per device.
 * Storage: used and free, a stacked bar by purpose, and an item list.
 */
import { useEffect, useState } from "react";

import { ChevronDown, HardDrive, Loader2, Server } from "lucide-react";

import { useGpuResources } from "../../hooks/useGpuResources";
import { apiFetch } from "../../lib/apiFetch";
import { formatGb, groupStorage, StorageCategory } from "../../lib/storageGroups";
import { tp, useT } from "../../i18n";

// h-full: GPU and storage share a row, so the cards match its height despite
// different content lengths.
const CARD = "h-full bg-white border border-neutral-200/80 rounded-[20px] shadow-[0_1px_3px_rgba(0,0,0,0.04)]";
const TRACK = "#F0F0EE";

/* ── Shared pieces ─────────────────────────────────────────── */

function CardHead({ icon: Icon, label }: { icon: typeof Server; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-2.5">
      <div className="w-[30px] h-[30px] rounded-[10px] bg-neutral-100 flex items-center justify-center">
        <Icon className="w-[15px] h-[15px] text-neutral-800" />
      </div>
      <span className="text-[12px] font-extrabold tracking-[-0.02em] text-neutral-900">{label}</span>
    </div>
  );
}

function SummaryRow({ children, open, onToggle }: { children: React.ReactNode; open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center justify-between gap-2 text-[11px] text-neutral-500 mt-[7px]"
    >
      <span className="truncate">{children}</span>
      <ChevronDown className={`w-3.5 h-3.5 flex-shrink-0 text-neutral-400 transition-transform ${open ? "" : "-rotate-90"}`} />
    </button>
  );
}

/** A large number, a caption and a gauge track. */
function Metric({
  value, unit, caption, fillPct, muted, dangerZonePct, ticks,
}: {
  value: string;
  unit?: string;
  caption: string;
  fillPct?: number;
  muted?: boolean;
  /** Width of the danger zone at the right of the track, in percent. */
  dangerZonePct?: number;
  ticks?: number[];
}) {
  return (
    <div className="flex-1 min-w-0">
      <p className="text-[20px] font-extrabold leading-none tracking-[-0.035em] text-neutral-900 tabular-nums">
        {value}
        {unit && <span className="text-[10.5px] font-semibold text-neutral-400 tracking-normal ml-0.5">{unit}</span>}
      </p>
      <p className="mt-[5px] text-[8.5px] font-bold uppercase tracking-[0.1em] text-neutral-400">{caption}</p>
      {fillPct != null && (
        <span className="relative mt-[7px] block h-[3px] rounded-sm overflow-hidden" style={{ background: TRACK }}>
          {dangerZonePct != null && (
            <i className="absolute right-0 top-0 h-full" style={{ width: `${dangerZonePct}%`, background: "#D2CEC4" }} />
          )}
          <i
            className={`absolute left-0 top-0 h-full rounded-sm ${muted ? "bg-neutral-300" : "bg-neutral-800"}`}
            style={{ width: `${Math.min(100, Math.max(0, fillPct))}%` }}
          />
          {(ticks ?? []).map((t) => (
            <i key={t} className="absolute top-0 w-[1.5px] h-[3px] bg-white" style={{ left: `${t}%` }} />
          ))}
        </span>
      )}
    </div>
  );
}

/* ── GPU ───────────────────────────────────────────────────── */

export function GpuMetricCard() {
  const t = useT();
  const { gpus, count, totalUsedMb, totalVramMb, avgUtil, available, loading } = useGpuResources();
  const [open, setOpen] = useState(false);

  const measured = gpus.filter((g) => g.available);
  const maxTemp = measured.reduce((m, g) => Math.max(m, g.temp), 0);
  const freeGb = (totalVramMb - totalUsedMb) / 1024;
  const totalGb = totalVramMb / 1024;

  return (
    <div className={CARD}>
      <div className="px-6 pt-5 pb-[18px]">
        <CardHead icon={Server} label="GPU" />

        {loading && count === 0 ? (
          <div className="flex items-center gap-2 text-[11px] text-neutral-400 mt-3">
            <Loader2 className="w-4 h-4 animate-spin" /> {t("Loading GPU resources…")}
          </div>
        ) : count === 0 || !available ? (
          <p className="text-[11px] text-neutral-400 mt-3">{t("No GPU detected")}</p>
        ) : (
          <>
            <SummaryRow open={open} onToggle={() => setOpen((v) => !v)}>
              {tp("{0} GPUs · {1}", count, gpus[0]?.name)}
            </SummaryRow>

            <div className="flex gap-4 mt-3.5">
              <Metric value={String(avgUtil)} unit="%" caption={t("Mean load")} fillPct={avgUtil} ticks={[25, 50, 75]} />
              <Metric
                value={String(maxTemp)} unit="°C" caption={t("Peak temperature")}
                fillPct={maxTemp} dangerZonePct={16.7} ticks={[16.7, 50, 83.3]}
              />
              <Metric
                value={formatGb(freeGb)} unit={`/${formatGb(totalGb)} GB`} caption={t("Free VRAM")}
                fillPct={totalGb ? (freeGb / totalGb) * 100 : 0} ticks={[25, 50, 75]}
              />
            </div>

            {open && (
              <div className="mt-3.5 bg-neutral-50 rounded-[14px] px-3.5 py-0.5">
                {gpus.map((g) => {
                  const busy = g.consumers.length > 0;
                  return (
                    <div key={g.index} className="flex items-center gap-2.5 py-[9px] border-t border-neutral-200/60 first:border-t-0">
                      <span className={`flex-shrink-0 w-[5px] h-[5px] rounded-full ${busy ? "bg-blue-600" : "bg-neutral-300"}`} />
                      <span className="flex-1 min-w-0 text-[10.5px] font-semibold text-neutral-600 truncate">
                        GPU{g.index} · {g.name}
                      </span>
                      {g.available ? (
                        <>
                          <span className="flex-shrink-0 relative w-[52px] h-[3px] rounded-sm bg-neutral-200 overflow-hidden">
                            <i
                              className={`absolute left-0 top-0 h-full rounded-sm ${busy ? "bg-neutral-700" : "bg-neutral-300"}`}
                              style={{ width: `${Math.min(100, g.util)}%` }}
                            />
                          </span>
                          <span className={`flex-shrink-0 w-[34px] text-right text-[10.5px] font-bold tabular-nums ${busy ? "text-neutral-700" : "text-neutral-400"}`}>
                            {g.util}%
                          </span>
                          <span className={`flex-shrink-0 w-[38px] text-right text-[10.5px] font-bold tabular-nums ${busy ? "text-neutral-700" : "text-neutral-400"}`}>
                            {g.temp}°C
                          </span>
                          <span className="flex-shrink-0 w-[52px] text-right text-[10.5px] font-medium text-neutral-400 tabular-nums">
                            {Math.round(g.usedMb / 1024)}/{Math.round(g.totalMb / 1024)}
                          </span>
                        </>
                      ) : (
                        <span className="text-[10.5px] text-neutral-400">{t("Metrics unavailable")}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ── Storage ───────────────────────────────────────────────── */

interface StorageResponse {
  total_size_mb: number;
  categories?: Record<string, StorageCategory>;
  disk?: { total_gb: number; used_gb: number; free_gb: number; usage_pct: number };
}

const SEG_COLOR: Record<string, string> = {
  dataset: "#2D2D2D",
  model: "#555555",
  output: "#A0A0A0",
  other: "#D4D4D4",
};

export function StorageMetricCard() {
  const t = useT();
  const [storage, setStorage] = useState<StorageResponse | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    apiFetch("/api/system/storage")
      .then((r) => r.json())
      .then(setStorage)
      .catch(() => {});
  }, []);

  const disk = storage?.disk;
  if (!disk) {
    return (
      <div className={CARD}>
        <div className="px-6 pt-5 pb-[18px]">
          <CardHead icon={HardDrive} label={t("Storage")} />
          <div className="flex items-center h-16">
            <Loader2 className="w-4 h-4 animate-spin text-neutral-300" />
          </div>
        </div>
      </div>
    );
  }

  const { groups, otherGb, freeGb } = groupStorage(storage?.categories, disk.used_gb, disk.free_gb);
  const rows = [...groups, { key: "other", label: "Other", gb: otherGb, pct: disk.used_gb ? (otherGb / disk.used_gb) * 100 : 0 }]
    .filter((r) => r.gb > 0.05);

  return (
    <div className={CARD}>
      <div className="px-6 pt-5 pb-[18px]">
        <CardHead icon={HardDrive} label={t("Storage")} />

        <SummaryRow open={open} onToggle={() => setOpen((v) => !v)}>
          {t("/storage · total")} <b className="font-bold text-neutral-700">{formatGb(disk.total_gb)} GB</b>
        </SummaryRow>

        <div className="flex gap-4 mt-3.5">
          <Metric value={formatGb(disk.used_gb)} unit={`/${formatGb(disk.total_gb)} GB`} caption={tp("in use · {0}%", Math.round(disk.usage_pct))} />
          <Metric value={formatGb(freeGb)} unit="GB" caption={t("Free space")} />
        </div>

        {/* Stacked bar by purpose */}
        <div className="flex gap-[1.5px] h-1.5 mt-[11px]">
          {rows.map((r, i) => (
            <i
              key={r.key}
              className={`block h-full ${i === 0 ? "rounded-l-[3px]" : ""}`}
              style={{ flex: Math.max(r.gb, 0.01), background: SEG_COLOR[r.key] ?? "#D4D4D4" }}
            />
          ))}
          <i className="block h-full rounded-r-[3px]" style={{ flex: Math.max(freeGb, 0.01), background: "#EDEDEB" }} />
        </div>

        {open && rows.length > 0 && (
          <div className="mt-3.5 bg-neutral-50 rounded-[14px] px-3.5 py-0.5">
            {rows.map((r) => (
              <div key={r.key} className="flex items-center gap-2.5 py-[7px] border-t border-neutral-200/60 first:border-t-0">
                <span className="flex-shrink-0 w-[7px] h-[7px] rounded-sm" style={{ background: SEG_COLOR[r.key] ?? "#D4D4D4" }} />
                <span className="flex-1 min-w-0 text-[10.5px] font-semibold text-neutral-600">{t(r.label)}</span>
                <span className="flex-shrink-0 w-[58px] text-right text-[10.5px] font-bold text-neutral-700 tabular-nums">
                  {formatGb(r.gb)} GB
                </span>
                <span className="flex-shrink-0 w-8 text-right text-[10.5px] font-medium text-neutral-400 tabular-nums">
                  {Math.round(r.pct)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
