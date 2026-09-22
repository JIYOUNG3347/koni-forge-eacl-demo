/**
 * GuidedConfirmParamsModal — final parameter confirmation before the
 * orchestrator actually fires the next chain step.
 *
 * Flow:
 *   chain card [Proceed] → this modal pops up with the structured params →
 *   user edits inline if desired → [Proceed with these parameters] calls
 *   advanceChain("approve", overrides). Cancelling keeps the card.
 */

import React, { useEffect, useMemo, useState } from "react";
import { X, ArrowRight, Loader2 } from "lucide-react";
import { Button } from "../ui/button";
import { apiFetch } from "../../lib/apiFetch";
import { t, tp, useT } from "../../i18n";

interface GuidedConfirmParamsModalProps {
  stage: string;          // "tuning" | ...
  params: Record<string, unknown>;
  announce?: string;
  busy?: boolean;
  onConfirm: (overrides?: Record<string, unknown>) => void;
  onCancel: () => void;
}

export function GuidedConfirmParamsModal({
  stage, params, announce, busy, onConfirm, onCancel,
}: GuidedConfirmParamsModalProps) {
  const t = useT();
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onCancel]);

  const title = stageTitle(stage);

  if (stage === "tuning") {
    return (
      <TuningEditor
        params={params}
        announce={announce}
        busy={busy}
        title={title}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    );
  }

  // Non-corpus stages: read-only display.
  const rows = buildRows(stage, params);
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onCancel}>
      <div
        className="w-[560px] max-w-[94vw] max-h-[86vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-100">
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-wide text-blue-600">CONFIRM PARAMETERS</p>
            <h3 className="text-[14px] font-bold text-neutral-900 truncate">{title}</h3>
          </div>
          <button onClick={onCancel} className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-800 hover:bg-neutral-100">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
          {announce && (
            <p className="text-[12px] text-neutral-600 leading-relaxed">{announce}</p>
          )}
          <div className="border border-neutral-100 rounded-lg overflow-hidden">
            {rows.length === 0 ? (
              <p className="px-3 py-3 text-[12px] text-neutral-400 text-center">{t("No parameters to show.")}</p>
            ) : (
              rows.map(({ label, value, mono, highlight }) => (
                <div key={label} className="flex items-start justify-between gap-3 px-3 py-2 border-b last:border-0 border-neutral-100 text-[12px]">
                  <span className="text-neutral-500 flex-shrink-0">{t(label)}</span>
                  <span className={`text-right ${mono ? "font-mono text-[11px]" : ""} ${highlight ? "font-bold text-neutral-900" : "text-neutral-700"} break-all`}>
                    {value}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-neutral-100 bg-neutral-50/50">
          <Button variant="outline" size="sm" className="h-9" onClick={onCancel} disabled={busy}>
            {t("Cancel")}
          </Button>
          <Button size="sm" className="h-9 gap-1.5 bg-blue-600 hover:bg-blue-700 text-white" onClick={() => onConfirm()} disabled={busy}>
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowRight className="w-3.5 h-3.5" />}
            {t("Proceed with these parameters")}
          </Button>
        </div>
      </div>
    </div>
  );
}

/* ── Corpus generation editor ───────────────────────────── */

/* ── Training (tuning) editor ─────────────────────────── */

function TuningEditor({
  params, announce, busy, title, onConfirm, onCancel,
}: {
  params: Record<string, unknown>;
  announce?: string;
  busy?: boolean;
  title: string;
  onConfirm: (overrides?: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const t = useT();
  // Recommended params from the dispatcher (stage="tuning")
  const initialJobName = String(params.job_name ?? "");
  const initialBaseModel = String(params.base_model ?? "");
  const initialDataset = String(params.dataset ?? params.dataset_name ?? "");
  const initialMethod = (String(params.method ?? "lora").toLowerCase() === "sft" ? "sft" : "lora") as "sft" | "lora";
  const initialEpochs = Number(params.epochs ?? 3) || 3;
  const initialBatch = Number(params.batch_size ?? 4) || 4;
  const initialLr = Number(params.learning_rate ?? 2e-4);
  const initialMaxSeq = Number(params.max_seq_length ?? 512) || 512;
  const initialLoraR = Number(params.lora_r ?? 8) || 8;
  const initialLoraAlpha = Number(params.lora_alpha ?? 16) || 16;
  const initialLoraDropout = Number(params.lora_dropout ?? 0.1);
  const datasetCount = params.dataset_count;
  const gpuInfo = String(params.gpu_info ?? "");

  const [jobName, setJobName] = useState<string>(initialJobName);
  const [baseModel, setBaseModel] = useState<string>(initialBaseModel);
  const [dataset, setDataset] = useState<string>(initialDataset);
  const [method, setMethod] = useState<"sft" | "lora">(initialMethod);
  const [epochs, setEpochs] = useState<number>(initialEpochs);
  const [batchSize, setBatchSize] = useState<number>(initialBatch);
  const [learningRate, setLearningRate] = useState<number>(Number.isFinite(initialLr) ? initialLr : 2e-4);
  const [maxSeq, setMaxSeq] = useState<number>(initialMaxSeq);
  const [loraR, setLoraR] = useState<number>(initialLoraR);
  const [loraAlpha, setLoraAlpha] = useState<number>(initialLoraAlpha);
  const [loraDropout, setLoraDropout] = useState<number>(Number.isFinite(initialLoraDropout) ? initialLoraDropout : 0.1);

  // Live model list from /api/models/base
  const [baseModels, setBaseModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState<boolean>(true);
  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/models/base")
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        const names = ((d?.models ?? []) as any[])
          .map((m) => (typeof m === "string" ? m : m?.name))
          .filter((n): n is string => !!n);
        setBaseModels(names);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setModelsLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const overrides = useMemo(() => {
    const out: Record<string, unknown> = {
      job_name: jobName,
      base_model: baseModel,
      dataset: dataset,
      dataset_name: dataset,  // backend alias
      method,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      max_seq_length: maxSeq,
    };
    if (method === "lora") {
      out.lora_r = loraR;
      out.lora_alpha = loraAlpha;
      out.lora_dropout = loraDropout;
    }
    return out;
  }, [jobName, baseModel, dataset, method, epochs, batchSize, learningRate, maxSeq,
      loraR, loraAlpha, loraDropout]);

  const disabled = !baseModel || !dataset || !jobName || !Number.isFinite(learningRate) || epochs <= 0 || batchSize <= 0;
  const submit = () => onConfirm(overrides);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onCancel}>
      <div
        className="w-[620px] max-w-[94vw] max-h-[86vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-100">
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-wide text-blue-600">CONFIRM PARAMETERS</p>
            <h3 className="text-[14px] font-bold text-neutral-900 truncate">{title}</h3>
          </div>
          <button onClick={onCancel} className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-800 hover:bg-neutral-100">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
          {announce && (
            <p className="text-[12px] text-neutral-600 leading-relaxed">{announce}</p>
          )}

          {/* Read-only context */}
          {(datasetCount != null || gpuInfo) && (
            <div className="border border-neutral-100 rounded-lg overflow-hidden">
              {datasetCount != null && (
                <RoRow label={t("Dataset samples")} value={tp("{0}", datasetCount)} />
              )}
              {gpuInfo && (
                <RoRow label="GPU" value={gpuInfo} last />
              )}
            </div>
          )}

          {/* Method tab */}
          <div className="border border-neutral-100 rounded-lg divide-y divide-neutral-100">
            <EditRow label={t("Method")}>
              <div className="flex gap-1.5">
                {(["sft", "lora"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMethod(m)}
                    disabled={busy}
                    className={`h-7 px-3 rounded border text-[11px] font-semibold transition-colors ${
                      method === m
                        ? "bg-neutral-900 text-white border-neutral-900"
                        : "bg-white text-neutral-500 border-neutral-200 hover:border-neutral-400"
                    }`}
                  >
                    {m === "sft" ? "FFT" : "LoRA"}
                  </button>
                ))}
              </div>
            </EditRow>
            <EditRow label={t("Job name")}>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                disabled={busy}
                placeholder={t("e.g. my-finetuning-job")}
                className="h-7 w-64 rounded border border-neutral-200 bg-white px-2 font-mono text-[11px] focus:outline-none focus:border-blue-400"
              />
            </EditRow>
            <EditRow label={t("Base model")}>
              {modelsLoading ? (
                <span className="text-[11px] text-neutral-400">{t("Loading models...")}</span>
              ) : baseModels.length > 0 ? (
                <select
                  value={baseModels.includes(baseModel) ? baseModel : ""}
                  onChange={(e) => setBaseModel(e.target.value)}
                  disabled={busy}
                  className="h-7 w-64 rounded border border-neutral-200 bg-white px-2 font-mono text-[11px] focus:outline-none focus:border-blue-400"
                >
                  {!baseModels.includes(baseModel) && <option value="" disabled>{t("Select a model...")}</option>}
                  {baseModels.map((name) => (<option key={name} value={name}>{name}</option>))}
                </select>
              ) : (
                <input
                  type="text"
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  disabled={busy}
                  className="h-7 w-64 rounded border border-neutral-200 bg-white px-2 font-mono text-[11px] focus:outline-none focus:border-blue-400"
                />
              )}
            </EditRow>
            <EditRow label={t("Dataset")}>
              <input
                type="text"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={busy}
                className="h-7 w-64 rounded border border-neutral-200 bg-white px-2 font-mono text-[11px] focus:outline-none focus:border-blue-400"
              />
            </EditRow>
          </div>

          {/* Hyperparameters */}
          <div className="border border-neutral-100 rounded-lg divide-y divide-neutral-100">
            <EditRow label={t("Epochs")}>
              <input type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} disabled={busy}
                className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400" />
            </EditRow>
            <EditRow label={t("Batch size")}>
              <input type="number" min={1} max={64} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} disabled={busy}
                className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400" />
            </EditRow>
            <EditRow label={t("Learning rate")}>
              <input type="number" step={0.00001} min={0.000001} max={0.01} value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} disabled={busy}
                className="h-7 w-32 rounded border border-neutral-200 bg-white px-2 text-right font-mono text-[11px] focus:outline-none focus:border-blue-400" />
            </EditRow>
            <EditRow label={t("Max sequence length")}>
              <input type="number" min={64} step={64} value={maxSeq} onChange={(e) => setMaxSeq(Number(e.target.value))} disabled={busy}
                className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400" />
            </EditRow>
          </div>

          {/* LoRA (method=lora only) */}
          {method === "lora" && (
            <div className="border border-neutral-100 rounded-lg divide-y divide-neutral-100">
              <EditRow label="LoRA Rank (r)">
                <select value={String(loraR)} onChange={(e) => setLoraR(Number(e.target.value))} disabled={busy}
                  className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400">
                  {[4, 8, 16, 32, 64].map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </EditRow>
              <EditRow label="LoRA Alpha">
                <input type="number" min={1} value={loraAlpha} onChange={(e) => setLoraAlpha(Number(e.target.value))} disabled={busy}
                  className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400" />
              </EditRow>
              <EditRow label="LoRA Dropout">
                <input type="number" step={0.05} min={0} max={1} value={loraDropout} onChange={(e) => setLoraDropout(Number(e.target.value))} disabled={busy}
                  className="h-7 w-24 rounded border border-neutral-200 bg-white px-2 text-right text-[12px] focus:outline-none focus:border-blue-400" />
              </EditRow>
            </div>
          )}

          <p className="text-[11px] text-neutral-400 leading-relaxed">
            {t("Adjust the values, then")} <b>{t("Proceed with these parameters")}</b>{t("to start training with the edited values.")}
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-neutral-100 bg-neutral-50/50">
          <Button variant="outline" size="sm" className="h-9" onClick={onCancel} disabled={busy}>
            {t("Cancel")}
          </Button>
          <Button size="sm" className="h-9 gap-1.5 bg-blue-600 hover:bg-blue-700 text-white" onClick={submit} disabled={busy || disabled}>
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowRight className="w-3.5 h-3.5" />}
            {t("Proceed with these parameters")}
          </Button>
        </div>
      </div>
    </div>
  );
}

function RoRow({ label, value, mono, last }: { label: string; value: string; mono?: boolean; last?: boolean }) {
  return (
    <div className={`flex items-start justify-between gap-3 px-3 py-2 text-[12px] ${last ? "" : "border-b border-neutral-100"}`}>
      <span className="text-neutral-500 flex-shrink-0">{t(label)}</span>
      <span className={`text-right break-all ${mono ? "font-mono text-[11px] text-neutral-900" : "text-neutral-700"}`}>{value}</span>
    </div>
  );
}

function EditRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2 text-[12px]">
      <span className="text-neutral-500 flex-shrink-0">{t(label)}</span>
      {children}
    </div>
  );
}

/* ── Helpers ───────────────────────────────────────────── */

function stageTitle(stage: string): string {
  switch (stage) {
    case "tuning":     return t("Training parameters");
    default:           return t("Stage parameters");
  }
}

interface Row { label: string; value: string; mono?: boolean; highlight?: boolean; }

function buildRows(stage: string, p: Record<string, unknown>): Row[] {
  const r = (label: string, value: unknown, opts: Partial<Row> = {}): Row | null => {
    if (value === undefined || value === null || value === "") return null;
    return { label, value: Array.isArray(value) ? value.join(", ") : String(value), mono: opts.mono, highlight: opts.highlight };
  };
  const rows: (Row | null)[] = [];

  if (stage === "tuning") {
    rows.push(r(t("Base model"),     p.base_model, { mono: true, highlight: true }));
    rows.push(r(t("Datasets"),        p.dataset, { mono: true }));
    rows.push(r(t("Number of dataset samples"), p.dataset_count != null ? tp("{0}", p.dataset_count) : null));
    rows.push(r(t("Job name"),       p.job_name, { mono: true }));
    rows.push(r(t("Method"),       typeof p.method === "string" ? p.method.toUpperCase() : p.method, { highlight: true }));
    rows.push(r(t("Epochs"),          p.epochs));
    rows.push(r(t("Batch size"),       p.batch_size));
    rows.push(r(t("Learning rate"),          p.learning_rate, { mono: true }));
    rows.push(r("GPU",             p.gpu_info));
  } else {
    for (const [k, v] of Object.entries(p)) {
      rows.push(r(k, v));
    }
  }
  return rows.filter((x): x is Row => x !== null);
}
