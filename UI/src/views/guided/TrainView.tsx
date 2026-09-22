/**
 * TrainView — Full-FT / LoRA with GUI+CLI modes, live polling and result charts.
 *
 * Backend: POST /api/train/start, GET /api/train/status/{id},
 * GET /api/train/checkpoints, DELETE /api/train/jobs/{id},
 * GET /api/train/results/{id}, GET /api/eval/report/{id}.
 */

import React, { useEffect, useState, useRef, useMemo, useCallback } from "react";
import {
  Brain, Zap, Play, Loader2, Trash2, StopCircle, Settings, BarChart2, } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { toast } from "sonner";
import { apiFetch } from "../../lib/apiFetch";
import { useDataStore, groupByBaseId } from "../../stores/dataStore";
import { useGpuResources } from "../../hooks/useGpuResources";
import { usePolling } from "../../hooks/usePolling";
import { useTrainActive, getTrainActiveSnapshot } from "../../hooks/useTrainActive";
import { HelpLabel } from "../../components/HelpLabel";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue, SelectGroup, SelectLabel,
} from "../../components/ui/select";
import { Slider } from "../../components/ui/slider";
import { Textarea } from "../../components/ui/textarea";
import { ScrollArea } from "../../components/ui/scroll-area";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from "../../components/ui/dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../../components/ui/tabs";
import {
  Card, CardContent, CardHeader, CardTitle, CardDescription,
} from "../../components/ui/card";
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "../../components/ui/chart";
import { tp, useT } from "../../i18n";
import { formatNumber } from "../../i18n/locale";

interface TrainJob {
  id: string;
  status: string;
  progress: number;
  message?: string;
  method: string;
  model?: string;
  dataset?: string;
  created_at: number;
  result?: Record<string, unknown>;
  params?: Record<string, unknown>;
  jobStatus?: string | null;  // train:state.status from backend ("stopped" | "failed" | null)
  // Child entries (checkpoints) under a job folder. When present, the history
  // row is expandable.
  children?: Array<{
    name: string;        // "round_1", "round_2", ...
    size_mb: number;
    path: string;
    location: string;    // "checkpoint" | "completed"
    is_lora: boolean;
  }>;
}

interface BaseModel { name: string; size_gb: number; }

interface CompletedModel {
  job_id: string;
  name: string;
  path: string;
  method: string | null;
  size_mb: number;
  is_lora: boolean;
  location: string;
}

/* ── Method label mapping ─────────────────────────────────────── */
// Backend uses `method="sft"` internally (Supervised Fine-Tuning) for the
// "Full Fine-tuning" path. We display it as "FFT" to the user since that's
// how the tab is labeled. LoRA stays LoRA.
function methodLabel(method?: string): string {
  const m = (method ?? "").toLowerCase();
  if (m === "lora") return "LoRA";
  if (m === "sft" || m === "fft" || m === "full" || m === "") return "FFT";
  return method!.toUpperCase();
}

/** Smart size formatter — MB ≥ 1024 shown as GB, else MB. */
function formatSize(sizeMb?: number | string | null): string {
  const n = Number(sizeMb);
  if (!isFinite(n) || n <= 0) return "-";
  if (n >= 1024) return `${(n / 1024).toFixed(2)} GB`;
  if (n >= 1) return `${n.toFixed(1)} MB`;
  return `${(n * 1024).toFixed(0)} KB`;
}

/* ── Param tooltips ───────────────────────────────────────────── */
const TIPS = {
  jobName: "Folder name for the training artifacts. Left empty, a timestamp-based name is assigned. Only letters, digits and _-. are allowed.",
  baseModel: "The pre-trained model fine-tuning starts from. Choose one of the HuggingFace models already downloaded to storage (/storage/models).",
  dataset: "The QA dataset to train on (see the training datasets section).",
  epochs: "How many times to train over the whole dataset. Too many risks overfitting. Usually 1-5.",
  batchSize: "How many samples go into the model at once. Larger is faster but GPU memory grows linearly.",
  maxSeq: "Maximum token length of one sample. Too short truncates, too long blows up memory. Pick 512-4096 based on your data.",
  lr: "How much to update the parameters at each step. Too high diverges, too low converges slowly. 1e-4-2e-4 is typical for LoRA, 1e-5-5e-5 for full fine-tuning.",
  loraR: "Rank (dimension) of the LoRA adapter. Higher means more capacity but more memory and parameters. 8-32 recommended.",
  loraAlpha: "LoRA scaling factor. By convention it is often set to twice r.",
  loraDropout: "Drops some adapter connections during training to curb overfitting. 0.05-0.1 is typical.",
  cli: "For writing or editing the SFTTrainer CLI command directly instead of using the form. It is parsed as --flag value pairs and used when training starts.",
};

function StatusBadge({ status }: { status: string }) {
  const t = useT();
  const map: Record<string, string> = {
    SUCCESS: "bg-neutral-900 text-white",
    completed: "bg-neutral-900 text-white",
    STARTED: "border border-neutral-300 text-neutral-600",
    running: "border border-neutral-300 text-neutral-600",
    queued: "border border-neutral-200 text-neutral-400",
    PENDING: "border border-neutral-200 text-neutral-400",
    FAILURE: "bg-red-50 text-red-600 border border-red-200",
    failed: "bg-red-50 text-red-600 border border-red-200",
    stopped: "bg-neutral-100 text-neutral-500",
    REVOKED: "bg-neutral-100 text-neutral-500",
  };
  const label: Record<string, string> = {
    SUCCESS: t("Completed"), completed: t("Completed"),
    STARTED: t("Training"), running: t("Training"),
    queued: t("Queued"), PENDING: t("Queued"),
    FAILURE: t("Failed"), failed: t("Failed"),
    stopped: t("Stop"), REVOKED: t("Stop"),
  };
  const cls = map[status] || "border border-neutral-200 text-neutral-400";
  const display = label[status] || status;
  const pulse = status === "STARTED" || status === "running";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold ${cls} max-w-[220px] truncate`}>
      {pulse && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5 flex-shrink-0 animate-pulse" />}
      {display}
    </span>
  );
}

/* ── CLI parsing ─────────────────────────────────────────────── */
function parseCli(cli: string): Record<string, string> {
  // Very small --flag value parser; values can be quoted.
  const out: Record<string, string> = {};
  const tokens = cli.trim().match(/(?:"[^"]*"|'[^']*'|\S)+/g) ?? [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.startsWith("--")) {
      const key = t.slice(2);
      const next = tokens[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        out[key] = next.replace(/^["']|["']$/g, "");
        i++;
      } else {
        out[key] = "true";
      }
    }
  }
  return out;
}

export default function TrainView() {
  const t = useT();
  const trainingDatasets = useDataStore((s) => s.trainingDatasets);
  const fetchDatasets = useDataStore((s) => s.fetchDatasets);
  const gpuRes = useGpuResources();
  const gpuCount = gpuRes.count;
  // MLflow tracking opt-in, which gates the "View MLflow history" button.

  const [baseModels, setBaseModels] = useState<BaseModel[]>([]);
  const [completedModels, setCompletedModels] = useState<CompletedModel[]>([]);
  const [activeTab, setActiveTab] = useState<"sft" | "lora">("sft");
  const [inputMode, setInputMode] = useState<"gui" | "cli">("gui");

  // Form state
  const [jobName, setJobName] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedBaseId, setSelectedBaseId] = useState("");
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [epochs, setEpochs] = useState("3");
  const [batchSize, setBatchSize] = useState("2");
  const [maxSeqLen, setMaxSeqLen] = useState("512");
  const [learningRate, setLearningRate] = useState([0.0002]);
  const [loraR, setLoraR] = useState("8");
  const [loraAlpha, setLoraAlpha] = useState("16");
  const [loraDropout, setLoraDropout] = useState("0.1");

  const trainMethod: "sft" | "lora" = activeTab;

  // Group dataset versions by base_id for the version selector.
  const corpusGroups = useMemo(() => groupByBaseId(trainingDatasets), [trainingDatasets]);

  // Selected base_id group and the versions available in it.
  const selectedGroup = corpusGroups.find((g) => g.base_id === selectedBaseId) ?? null;
  const availableVersions = selectedGroup?.versions ?? [];
  const isMultiVersion = availableVersions.length > 1;

  // With no version selected, apply the latest (highest version) automatically.
  const effectiveVersion = selectedVersion ?? (availableVersions[0]?.version ?? null);
  const effectiveItem = availableVersions.find((v) => v.version === effectiveVersion) ?? null;

  // Folder name sent in the training payload: v1 is the bare base_id, v2+ adds _v{N}.
  function versionedFolderName(baseId: string, ver: number): string {
    return ver <= 1 ? baseId : `${baseId}_v${ver}`;
  }
  const resolvedDatasetName = selectedBaseId && effectiveVersion
    ? versionedFolderName(selectedBaseId, effectiveVersion)
    : "";

  // CLI text (auto-generated from GUI state, user can override in CLI mode)
  const [cliText, setCliText] = useState("");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [trainJobs, setTrainJobs] = useState<TrainJob[]>([]);
  const stoppedJobIds = useRef<Set<string>>(new Set());
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Modals
  const [resultOpen, setResultOpen] = useState(false);
  const [resultJob, setResultJob] = useState<TrainJob | null>(null);
  const [resultDetail, setResultDetail] = useState<any>(null);
  const [resultLoading, setResultLoading] = useState(false);
  const [paramOpen, setParamOpen] = useState(false);
  const [paramKV, setParamKV] = useState<Record<string, unknown>>({});
  const [paramTitle, setParamTitle] = useState("");

  // Multi-round expand/collapse for the "Completed runs" table
  const [expandedJobs, setExpandedJobs] = useState<Record<string, boolean>>({});

  /* ── Load lists + GPU info ───────────────────────────── */
  const refreshCheckpoints = useCallback(async () => {
    try {
      const snap = getTrainActiveSnapshot();
      const activeJobsFromServer: TrainJob[] = snap.active.map((j) => ({
        id: j.id,
        status: j.status,
        progress: j.progress,
        message: j.message,
        method: j.method,
        model: j.model,
        dataset: j.dataset,
        created_at: j.created_at,
      }));
      const activeIds = new Set<string>(snap.active.map((j) => j.id));

      const r = await apiFetch("/api/train/checkpoints");
      if (!r.ok) return;
      const d = await r.json();
      const cps = d.checkpoints ?? [];
      const jobMap: Record<string, TrainJob> = {};
      for (const cp of cps) {
        // Skip checkpoints whose parent job is still running — otherwise interim
        // dumps get promoted to "SUCCESS" while training is mid-flight.
        if (activeIds.has(cp.job_id)) continue;

        const existing = jobMap[cp.job_id];
        const childRow = {
          name: cp.name,                // e.g. "round_1" / "final" / "checkpoint-75"
          size_mb: Number(cp.size_mb ?? 0),
          path: cp.path,
          location: cp.location,
          is_lora: Boolean(cp.is_lora),
        };
        if (!existing) {
          jobMap[cp.job_id] = {
            id: cp.job_id,
            status: "SUCCESS",
            progress: 100,
            method: cp.method || (cp.is_lora ? "lora" : "sft"),
            dataset: cp.dataset_name ?? "",
            created_at: new Date(cp.created_at).getTime(),
            result: { size_mb: cp.size_mb, path: cp.path, location: cp.location },
            params: {
              method: cp.method || (cp.is_lora ? "lora" : "sft"),
              dataset_name: cp.dataset_name ?? "",
            },
            jobStatus: cp.job_status ?? null,
            children: [childRow],
          };
        } else {
          const prevSize = Number((existing.result as any)?.size_mb ?? 0);
          existing.result = {
            ...(existing.result ?? {}),
            size_mb: prevSize + Number(cp.size_mb ?? 0),
          };
          existing.children = [...(existing.children ?? []), childRow];
        }
      }
      // Sort children by name so round_1 → round_2 → checkpoint-N appear in order.
      for (const j of Object.values(jobMap)) {
        if (j.children && j.children.length > 0) {
          j.children.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
        }
      }
      setTrainJobs((prev) => {
        // Merge priority:
        //   1. Local in-flight rows from prev (progress/message updated by poll).
        //   2. Server-side active rows — fills gap after page reload.
        //   3. Local stopped rows — preserved so user can manually delete.
        //   4. Completed rows from /api/train/checkpoints.
        const localActive = prev.filter((j) => ["queued", "STARTED", "running", "PENDING"].includes(j.status));
        const localStopped = prev.filter((j) => ["stopped", "REVOKED"].includes(j.status));
        const localActiveIds = new Set(localActive.map((j) => j.id));
        const localStoppedIds = new Set(localStopped.map((j) => j.id));
        const serverActiveNotInLocal = activeJobsFromServer.filter((j) => !localActiveIds.has(j.id));
        const allActiveIds = new Set([...localActiveIds, ...serverActiveNotInLocal.map((j) => j.id)]);
        // Exclude checkpoint-scan results for stopped jobs (stoppedJobIds ref covers
        // newly-stopped jobs in this session; localStoppedIds covers page-reload case).
        // Also exclude jobs whose train:state.status is "stopped" (server-side flag,
        // survives page reload — fixes the "stopped → refresh → misreported as done" bug).
        const stopped = stoppedJobIds.current;
        const diskStopped = Object.values(jobMap).filter(
          (j) => !allActiveIds.has(j.id) && j.jobStatus === "stopped"
              && !localStoppedIds.has(j.id) && !stopped.has(j.id)
        );
        const completed = Object.values(jobMap).filter(
          (j) => !allActiveIds.has(j.id) && !stopped.has(j.id) && !localStoppedIds.has(j.id)
              && j.jobStatus !== "stopped"
        );
        return [...localActive, ...serverActiveNotInLocal, ...localStopped, ...diskStopped, ...completed];
      });
    } catch { /* keep state */ }
  }, []);

  useEffect(() => {
    fetchDatasets();
    apiFetch("/api/models/base")
      .then((r) => r.json())
      .then((d) => setBaseModels(d.models ?? []))
      .catch(() => {});
    apiFetch("/api/train/checkpoints")
      .then((r) => (r.ok ? r.json() : { checkpoints: [] }))
      .then((d) => {
        const cps: CompletedModel[] = (d.checkpoints ?? [])
          .filter(
            (c: { location?: string; job_status?: string | null; is_lora?: boolean }) =>
              (c.location === "completed" || c.location === "checkpoint") &&
              c.job_status !== "stopped" &&
              !c.is_lora,
          )
          .map((c: Record<string, unknown>) => ({
            job_id: String(c.job_id ?? ""),
            name: String(c.name ?? ""),
            path: String(c.path ?? ""),
            method: (c.method as string | null) ?? null,
            size_mb: Number(c.size_mb ?? 0),
            is_lora: Boolean(c.is_lora),
            location: String(c.location ?? ""),
          }));
        setCompletedModels(cps);
      })
      .catch(() => {});
  }, [fetchDatasets]);

  // The checkpoint list is a disk scan, so poll every 30s (visibility aware), once immediately.
  usePolling(refreshCheckpoints, 30000);

  // Pick up a job an agent started elsewhere: when the active id set from the
  // 5s /active poller changes, refresh the checkpoint and progress rows at once.
  const trainActive = useTrainActive();
  const activeKey = trainActive.active.map((j) => j.id).sort().join(",");
  useEffect(() => {
    void refreshCheckpoints();
  }, [activeKey, refreshCheckpoints]);

  /* ── GUI → CLI live sync ──────────────────────────────── */
  useEffect(() => {
    if (inputMode !== "gui") return;
    const cmd = [
      "sft_trainer.py",
      `--mode ${trainMethod}`,
      `--model_name_or_path ${selectedModel || "<model>"}`,
      `--dataset_name ${resolvedDatasetName || "<dataset>"}`,
      `--num_train_epochs ${epochs}`,
      `--per_device_train_batch_size ${batchSize}`,
      `--learning_rate ${learningRate[0]}`,
      `--max_length ${maxSeqLen}`,
      ...(activeTab === "lora"
        ? [`--lora_r ${loraR}`, `--lora_alpha ${loraAlpha}`, `--lora_dropout ${loraDropout}`]
        : []),
      `--output_dir /storage/outputs/{user}/completed/${jobName || "<job_name>"}`,
    ].join(" ");
    setCliText(cmd);
  }, [
    inputMode, trainMethod, activeTab, selectedModel, resolvedDatasetName,
    epochs, batchSize, maxSeqLen, learningRate, loraR, loraAlpha, loraDropout,
    jobName,
  ]);

  /* ── Live polling for in-flight jobs ─────────────────── */
  useEffect(() => {
    const active = trainJobs.filter((j) => j.status === "queued" || j.status === "STARTED" || j.status === "running" || j.status === "PENDING");
    if (active.length === 0) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      if (document.hidden) return;
      const latest = await Promise.all(
        trainJobs.map(async (j) => {
          if (!["queued", "STARTED", "running", "PENDING"].includes(j.status)) return j;
          try {
            const r = await apiFetch(`/api/train/status/${encodeURIComponent(j.id)}`);
            if (!r.ok) return j;
            const d = await r.json();
            return {
              ...j,
              status: d.status ?? j.status,
              progress: typeof d.progress === "number" ? d.progress : j.progress,
              message: d.message ?? j.message,
            };
          } catch { return j; }
        }),
      );
      setTrainJobs(latest);
      // If a job transitioned to SUCCESS, refresh the checkpoint list so the
      // artifacts / delete button show up properly.
      const justFinished = latest.some(
        (j, i) => j.status !== trainJobs[i]?.status && (j.status === "completed" || j.status === "SUCCESS"),
      );
      if (justFinished) void refreshCheckpoints();
    }, 5000);
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [trainJobs, refreshCheckpoints]);

  /* ── Start training ──────────────────────────────────── */
  const handleStart = async () => {
    // Resolve params — either directly from GUI, or parsed from CLI textarea
    let payload: Record<string, unknown>;
    if (inputMode === "cli") {
      const parsed = parseCli(cliText);
      if (!parsed.model_name_or_path && !parsed.model_name) {
        alert(t("Specify --model_name_or_path or --model_name in the CLI command."));
        return;
      }
      if (!parsed.dataset_name && !parsed.dataset_path) {
        alert(t("Specify --dataset_name or --dataset_path in the CLI command."));
        return;
      }
      const modelPath = parsed.model_name_or_path ?? parsed.model_name!;
      const datasetPath = parsed.dataset_name ?? parsed.dataset_path!;
      // extract folder name from paths
      const modelLeaf = modelPath.replace(/\/+$/, "").split("/").pop() || modelPath;
      const datasetLeaf = datasetPath.replace(/\/+$/, "").split("/").pop() || datasetPath;
      const modelIsAbsPath = modelPath.startsWith("/");
      payload = {
        model_name: modelLeaf,
        model_path: modelIsAbsPath ? modelPath : undefined,
        dataset_name: datasetLeaf,
        method: parsed.mode || trainMethod,
        epochs: Number(parsed.num_train_epochs ?? 3),
        batch_size: Number(parsed.per_device_train_batch_size ?? 2),
        learning_rate: Number(parsed.learning_rate ?? 2e-4),
        max_seq_length: Number(parsed.max_length ?? 512),
        lora_r: Number(parsed.lora_r ?? 8),
        lora_alpha: Number(parsed.lora_alpha ?? 16),
        lora_dropout: Number(parsed.lora_dropout ?? 0.1),
        job_name: jobName || undefined,
      };
    } else {
      if (!selectedModel || !resolvedDatasetName) {
        alert(t("Select both a base model and a dataset."));
        return;
      }
      const selectedTrained = completedModels.find((c) => c.path === selectedModel);
      payload = {
        model_name: selectedTrained ? selectedTrained.job_id : selectedModel,
        model_path: selectedTrained ? selectedTrained.path : undefined,
        dataset_name: resolvedDatasetName,
        method: trainMethod,
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        learning_rate: learningRate[0],
        max_seq_length: parseInt(maxSeqLen),
        lora_r: activeTab === "lora" ? parseInt(loraR) : undefined,
        lora_alpha: activeTab === "lora" ? parseInt(loraAlpha) : undefined,
        lora_dropout: activeTab === "lora" ? parseFloat(loraDropout) : undefined,
        job_name: jobName || undefined,
      };
    }

    setIsSubmitting(true);
    try {
      const res = await apiFetch("/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        alert(tp("Could not start training: {0}", (d as { detail?: string }).detail ?? `HTTP ${res.status}`));
        return;
      }
      const d = await res.json();
      setTrainJobs((prev) => [
        {
          id: d.job_id,
          status: "queued",
          progress: 0,
          message: t("Waiting..."),
          method: String(payload.method ?? activeTab),
          model: String(payload.model_name ?? ""),
          dataset: String(payload.dataset_name ?? ""),
          created_at: Date.now(),
          params: payload,
        },
        ...prev,
      ]);
    } catch (e) {
      alert(tp("Could not start training: {0}", (e as Error).message));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStop = async (jobId: string) => {
    if (!confirm(t("Stop the training run in progress?"))) return;
    try {
      const r = await apiFetch(`/api/train/stop/${encodeURIComponent(jobId)}`, { method: "POST" });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        alert(tp("Could not stop: {0}", (d as { detail?: string }).detail ?? `HTTP ${r.status}`));
        return;
      }
      // Mark as stopped so refreshCheckpoints won't re-surface this job's
      // partial checkpoints as a new SUCCESS row.
      stoppedJobIds.current.add(jobId);
      setTrainJobs((p) => p.map((j) => j.id === jobId ? { ...j, status: "stopped" } : j));
    } catch (e) { alert(tp("Could not stop: {0}", (e as Error).message)); }
  };

  const handleDeleteJob = useCallback(async (jobId: string) => {
    if (!confirm(tp("This deletes the training artifacts for '{0}' from disk. Continue?", jobId))) return;
    try {
      const r = await apiFetch(`/api/train/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        toast.error(tp("Delete failed: {0}", (d as { detail?: string }).detail ?? `HTTP ${r.status}`));
        return;
      }
      const done = (await r.json().catch(() => ({}))) as { deleted?: string[] };
      const n = done.deleted?.length ?? 0;
      toast.success(n > 0 ? tp("'{0}' deleted — {1} folders cleaned up", jobId, n) : tp("Deleted the record for '{0}'", jobId));
      await refreshCheckpoints();
    } catch (e) { toast.error(tp("Delete failed: {0}", (e as Error).message)); }
  }, [refreshCheckpoints]);

  const openResult = async (job: TrainJob) => {
    setResultJob(job); setResultOpen(true); setResultLoading(true);
    setResultDetail(null);
    try {
      const r = await apiFetch(`/api/train/results/${encodeURIComponent(job.id)}`);
      if (r.ok) setResultDetail(await r.json());
    } catch { /* */ } finally { setResultLoading(false); }
  };

  const openParams = async (job: TrainJob) => {
    setParamTitle(job.id);
    setParamOpen(true);
    // Start with whatever we have in memory (newly-started jobs have
    // `params` from handleStart's payload).
    const initial = (job.params ?? {}) as Record<string, unknown>;
    setParamKV(initial);

    // Always refresh from backend so the modal reflects live state even for
    // jobs loaded from /checkpoints (page reload) or jobs still running.
    // Priority: training_params.json > training_results.json > /status snapshot
    // > whatever we had in memory. Merge keeps any caller-supplied overrides.
    try {
      const merged: Record<string, unknown> = { ...initial };

      // /status — authoritative for live hyperparameter snapshot + progress
      try {
        const s = await apiFetch(`/api/train/status/${encodeURIComponent(job.id)}`);
        if (s.ok) {
          const sd = await s.json();
          for (const k of [
            "method", "model_name", "dataset_name",
            "epochs", "batch_size", "learning_rate", "max_seq_length",
            "lora_r", "lora_alpha", "lora_dropout",
            "output_dir",
          ] as const) {
            const v = (sd as Record<string, unknown>)[k];
            if (v != null && v !== "") merged[k] = v;
          }
        }
      } catch { /* ignore */ }

      // /results — disk files, adds runtime/loss/tokens metrics if training done
      try {
        const r = await apiFetch(`/api/train/results/${encodeURIComponent(job.id)}`);
        if (r.ok) {
          const d = await r.json();
          Object.assign(merged, (d.training_params ?? {}) as Record<string, unknown>);
          Object.assign(merged, (d.training_results ?? {}) as Record<string, unknown>);
        }
      } catch { /* ignore */ }

      if (Object.keys(merged).length > 0) {
        setParamKV(merged);
        setTrainJobs((prev) => prev.map((j) => j.id === job.id ? { ...j, params: merged } : j));
      }
    } catch { /* ignore */ }
  };

  /* ── Derived ───────────────────────────────────────────── */
  const activeJobs = trainJobs.filter((j) => ["queued", "STARTED", "running", "PENDING"].includes(j.status));
  const stoppedJobs = trainJobs.filter((j) => ["stopped", "REVOKED"].includes(j.status));
  const completedJobs = trainJobs
    .filter((j) => ["SUCCESS", "completed"].includes(j.status))
    .sort((a, b) => b.created_at - a.created_at);

  const logHistory = useMemo(() => resultDetail?.log_history ?? [], [resultDetail]);
  const chartData = useMemo(
    () =>
      logHistory
        .filter((e: any) => e.loss != null)
        .map((e: any) => ({
          epoch: e.epoch, loss: e.loss, accuracy: e.mean_token_accuracy ?? 0, step: e.step,
        })),
    [logHistory],
  );
  const epochTicks = useMemo(() => {
    if (chartData.length === 0) return [];
    const max = Math.ceil(Math.max(...chartData.map((d: any) => d.epoch)));
    return Array.from({ length: max }, (_, i) => i + 1);
  }, [chartData]);

  const trainResult = useMemo(
    () => resultDetail?.training_results ?? resultDetail?.training_params ?? {},
    [resultDetail],
  );
  const fmtRuntime = (sec: number | undefined) => {
    const n = Number(sec);
    if (!n || isNaN(n)) return "-";
    const h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = Math.floor(n % 60);
    const parts: string[] = [];
    if (h > 0) parts.push(tp("{0}h", h));
    if (m > 0) parts.push(tp("{0}m", m));
    parts.push(tp("{0}s", s));
    return parts.join(" ");
  };
  const runtimeStr = useMemo(() => {
    return fmtRuntime(trainResult.train_runtime);
  }, [trainResult]);

  /* ── Render ────────────────────────────────────────────── */
  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <div>
        <p className="text-[10px] font-bold text-neutral-300 uppercase tracking-[0.2em] mb-2">Training Agent</p>
        <h1 className="text-[28px] font-black text-neutral-900 tracking-tight">{t("Training")}</h1>
        <p className="text-[13px] text-neutral-400 mt-1">{t("Tune the model's parameters to build a model optimised for your task")}</p>
      </div>

      {/* Settings card */}
      <div className="bg-white border border-neutral-100 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
        <div className="flex items-start justify-between gap-4 p-6 pb-0">
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2">
              {activeTab === "sft" ? <Brain className="w-5 h-5 text-muted-foreground" /> : <Zap className="w-5 h-5 text-muted-foreground" />}
              {activeTab === "sft" ? t("Full fine-tuning settings") : t("LoRA Settings")}
            </h3>
            <p className="text-xs text-muted-foreground/70 mt-1">
              {activeTab === "sft"
                ? t("Updates every model parameter. Large models need a lot of VRAM.")
                : t("Low-Rank Adaptation — trains only a small adapter, so it is efficient in memory and speed.")}
            </p>
          </div>
          <Tabs value={inputMode} onValueChange={(v) => setInputMode(v as "gui" | "cli")}>
            <TabsList className="bg-muted">
              <TabsTrigger value="gui" className="data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm">GUI</TabsTrigger>
              <TabsTrigger value="cli" className="data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm">CLI</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <div className="p-6 pt-4 space-y-4">
          {inputMode === "gui" ? (
            <>
              {/* Training method: Full fine-tuning or LoRA */}
              <div className="grid grid-cols-2 bg-neutral-50 rounded-lg p-0.5 gap-0.5">
                {[
                  { id: "sft" as const, label: "Full Fine-tuning" },
                  { id: "lora" as const, label: "LoRA" },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setActiveTab(m.id)}
                    className={`py-2 px-3 text-[12px] font-medium rounded-md transition-all ${
                      activeTab === m.id
                        ? "bg-white text-neutral-900 shadow-sm font-semibold"
                        : "text-neutral-400 hover:text-neutral-600"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              {/* Row 1: job name + base model */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.jobName)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Job name")}</Label></HelpLabel>
                  <Input value={jobName} onChange={(e) => setJobName(e.target.value)} placeholder={t("e.g. my-finetuning-job (generated automatically if left empty)")} className="rounded-lg" />
                </div>
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.baseModel)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Base model")}</Label></HelpLabel>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="rounded-lg"><SelectValue placeholder={t("Select a model")} /></SelectTrigger>
                    <SelectContent className="bg-white">
                      {baseModels.length > 0 ? (
                        <SelectGroup>
                          <SelectLabel className="text-[10px]">{t("Base model")}</SelectLabel>
                          {baseModels.map((m) => (
                            <SelectItem key={m.name} value={m.name}>{m.name} ({m.size_gb}GB)</SelectItem>
                          ))}
                        </SelectGroup>
                      ) : (completedModels.length === 0) ? (
                        <div className="px-3 py-2 text-[11px] text-neutral-400">{t("Download a model first from Settings → Models")}</div>
                      ) : null}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Row 2: dataset + epochs */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.dataset)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Dataset")}</Label></HelpLabel>
                  <Select
                    value={selectedBaseId}
                    onValueChange={(v) => {
                      setSelectedBaseId(v);
                      setSelectedVersion(null); // a new base_id resets the version to latest
                    }}
                  >
                    <SelectTrigger className="rounded-lg"><SelectValue placeholder={t("Select a dataset")} /></SelectTrigger>
                    <SelectContent className="bg-white">
                      {corpusGroups.length === 0 ? (
                        <div className="px-3 py-2 text-[11px] text-neutral-400">
                            {t("No SFT dataset. Run preprocessing and TWIST on the data page first.")}
                        </div>
                      ) : corpusGroups.map((g) => (
                        <SelectItem key={g.base_id} value={g.base_id}>{g.base_id}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {/* Version selector, shown only when there are several */}
                  {isMultiVersion && (
                    <Select
                      value={String(effectiveVersion ?? "")}
                      onValueChange={(v) => setSelectedVersion(Number(v))}
                    >
                      <SelectTrigger className="rounded-lg mt-1.5"><SelectValue placeholder={t("Select a version")} /></SelectTrigger>
                      <SelectContent className="bg-white">
                        {availableVersions.map((v) => (
                          <SelectItem key={v.id} value={String(v.version ?? 1)}>
                            v{v.version ?? 1}
                            {v.version === availableVersions[0]?.version ? t(" · latest") : ""}
                            {typeof v.qa_count === "number" && v.qa_count > 0
                              ? tp(" · {0} {1}", "SFT", v.qa_count)
                              : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                  {/* Secondary metadata for the active version */}
                  {effectiveItem && (
                    <p className="mt-1 text-[10.5px] text-neutral-500">
                      v{effectiveItem.version ?? 1}
                      {effectiveItem.version === availableVersions[0]?.version ? t(" · latest") : ""}
                      {typeof effectiveItem.qa_count === "number" && effectiveItem.qa_count > 0
                        ? tp(" · {0} {1}", "SFT", effectiveItem.qa_count) : ""}
                      {tp(" · folder: {0}", resolvedDatasetName)}
                    </p>
                  )}
                </div>
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.epochs)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Epochs")}</Label></HelpLabel>
                  <Input type="number" min={1} value={epochs} onChange={(e) => setEpochs(e.target.value)} className="rounded-lg" />
                </div>
              </div>

              {/* Row 3: batch + maxseq */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.batchSize)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Batch size")}</Label></HelpLabel>
                  <Select value={batchSize} onValueChange={setBatchSize}>
                    <SelectTrigger className="rounded-lg"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-white">
                      {[1, 2, 4, 8, 16, 32].map((s) => (<SelectItem key={s} value={String(s)}>{s}</SelectItem>))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <HelpLabel text={t(TIPS.maxSeq)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Max sequence length")}</Label></HelpLabel>
                  <Input type="number" min={64} step={64} value={maxSeqLen} onChange={(e) => setMaxSeqLen(e.target.value)} className="rounded-lg" />
                </div>
              </div>

              {/* LoRA extras — only on the LoRA sub-tab (DPO is full FT only) */}
              {activeTab === "lora" && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-1.5">
                    <HelpLabel text={t(TIPS.loraR)}><Label className="text-[12px] font-semibold text-neutral-600">LoRA Rank (r)</Label></HelpLabel>
                    <Select value={loraR} onValueChange={setLoraR}>
                      <SelectTrigger className="rounded-lg"><SelectValue /></SelectTrigger>
                      <SelectContent className="bg-white">
                        {[4, 8, 16, 32, 64].map((r) => (<SelectItem key={r} value={String(r)}>{r}</SelectItem>))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <HelpLabel text={t(TIPS.loraAlpha)}><Label className="text-[12px] font-semibold text-neutral-600">LoRA Alpha</Label></HelpLabel>
                    <Input type="number" value={loraAlpha} onChange={(e) => setLoraAlpha(e.target.value)} className="rounded-lg" />
                  </div>
                  <div className="space-y-1.5">
                    <HelpLabel text={t(TIPS.loraDropout)}><Label className="text-[12px] font-semibold text-neutral-600">LoRA Dropout</Label></HelpLabel>
                    <Input type="number" step={0.05} min={0} max={1} value={loraDropout} onChange={(e) => setLoraDropout(e.target.value)} className="rounded-lg" />
                  </div>
                </div>
              )}

              {/* LR slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <HelpLabel text={t(TIPS.lr)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Learning rate")}</Label></HelpLabel>
                  <span className="text-[12px] font-mono font-bold text-neutral-900 tabular-nums">
                    {learningRate[0].toExponential(1)}
                  </span>
                </div>
                <Slider value={learningRate} onValueChange={setLearningRate} max={0.001} min={0.00001} step={0.00001} />
              </div>

            </>
          ) : (
            /* CLI mode */
            <div className="space-y-2">
              <HelpLabel text={t(TIPS.cli)}><Label className="text-[12px] font-semibold text-neutral-600">{t("CLI command")}</Label></HelpLabel>
              <Textarea
                value={cliText}
                onChange={(e) => setCliText(e.target.value)}
                spellCheck={false}
                autoComplete="off"
                className="min-h-[160px] font-mono text-[11px] resize-none rounded-lg bg-neutral-50 border border-neutral-200"
              />
              <p className="text-[10px] text-neutral-400">
                <b>--flag value</b> {t("format. It is parsed into structured parameters on submit.")}
                <b> --model_name_or_path</b>{t("and")} <b> --dataset_name</b>{t("is required.")}
              </p>
              {/* Still show a job_name input in CLI mode — the backend treats it
                  separately from --output_dir so user can override/override-not. */}
              <div className="space-y-1.5 pt-2">
                <HelpLabel text={t(TIPS.jobName)}><Label className="text-[12px] font-semibold text-neutral-600">{t("Job name")}</Label></HelpLabel>
                <Input value={jobName} onChange={(e) => setJobName(e.target.value)} placeholder={t("e.g. my-finetuning-job (generated automatically if left empty)")} className="rounded-lg" />
              </div>
            </div>
          )}

          {/* GPU note — device inventory from the read model */}
          {gpuCount > 0 && (
            <div className="text-[10px] text-neutral-400 space-y-0.5">
              <div>
                {tp("Detected GPUs: {0}", gpuCount)}
                {gpuCount < 2 && t(" (distributed training needs at least 2 GPUs)")}
              </div>
              {gpuRes.gpus.map((g) => (
                <div key={g.index} className="tabular-nums">
                  · GPU{g.index} {g.name} — {(g.usedMb / 1024).toFixed(0)}/{(g.totalMb / 1024).toFixed(0)} GB · UTIL {g.util}%
                </div>
              ))}
            </div>
          )}

          {gpuRes.loading && (
            <div className="rounded-xl border border-dashed border-neutral-200 bg-white p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1.5">
                  <div className="h-3 w-[150px] rounded bg-neutral-100 animate-pulse" />
                  <div className="h-2.5 w-[230px] rounded bg-neutral-100 animate-pulse" />
                </div>
                <div className="h-6 w-10 rounded-full bg-neutral-100 animate-pulse flex-shrink-0" />
              </div>
            </div>
          )}

          <button
            onClick={handleStart}
            disabled={isSubmitting}
            className="w-full py-3 rounded-xl bg-neutral-900 text-white font-semibold text-[13px] transition-all hover:bg-neutral-800 shadow-sm hover:shadow-md disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isSubmitting ? (<><Loader2 className="w-4 h-4 animate-spin" /> {t("Starting...")}</>) : (<><Play className="w-4 h-4" /> {t("Start training")}</>)}
          </button>
        </div>
      </div>

      {/* Active jobs */}
      {activeJobs.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-[16px] font-bold text-neutral-900">{t("In progress")}</h3>
          {activeJobs.map((job) => (
            <div key={job.id} className="bg-white border border-neutral-100 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm font-medium">{job.id}</span>
                  <span className="text-[11px] font-semibold text-neutral-500 ml-2">{methodLabel(job.method)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge status={job.status} />
                  <button onClick={() => openParams(job)} className="p-1.5 rounded-md text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100" title={t("Parameter")}>
                    <Settings className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => handleStop(job.id)} className="p-1.5 rounded-md text-neutral-400 hover:text-red-600 hover:bg-red-50" title={t("Stop")}>
                    <StopCircle className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              {job.message && <div className="text-[10px] text-neutral-400">{t(job.message)}</div>}
              <div className="flex items-center gap-2">
                <div className="w-full h-1.5 rounded-full bg-neutral-100 overflow-hidden">
                  <div className="h-full rounded-full bg-neutral-900 transition-all duration-500" style={{ width: `${job.progress ?? 0}%` }} />
                </div>
                <span className="text-[10px] tabular-nums text-neutral-400">{job.progress ?? 0}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stopped jobs */}
      {stoppedJobs.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-[16px] font-bold text-neutral-900">{t("Stopped runs")}</h3>
          {stoppedJobs.map((job) => (
            <div key={job.id} className="bg-white border border-neutral-100 rounded-xl p-4 space-y-3 opacity-70">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm font-medium">{job.id}</span>
                  <span className="text-[11px] font-semibold text-neutral-500 ml-2">{methodLabel(job.method)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge status="stopped" />
                  <button onClick={() => openParams(job)} className="p-1.5 rounded-md text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100" title={t("Parameter")}>
                    <Settings className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => handleDeleteJob(job.id)} className="p-1.5 rounded-md text-neutral-400 hover:text-red-600 hover:bg-red-50" title={t("Delete")}>
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-full h-1.5 rounded-full bg-neutral-100 overflow-hidden">
                  <div className="h-full rounded-full bg-neutral-400 transition-all duration-500" style={{ width: `${job.progress ?? 0}%` }} />
                </div>
                <span className="text-[10px] tabular-nums text-neutral-400">{job.progress ?? 0}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Completed history */}
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold text-neutral-300 uppercase tracking-[0.2em] mb-1">History</p>
          <h3 className="text-[16px] font-bold text-neutral-900">{t("Completed runs")}</h3>
        </div>
      </div>
      <div className="bg-white border border-neutral-100 rounded-xl overflow-hidden">
        <div className="grid grid-cols-[2fr_1fr_1fr_auto] items-center gap-4 px-4 py-2.5 bg-neutral-50/80 border-b border-neutral-200 text-[10px] font-semibold text-neutral-500 uppercase tracking-wider">
          <span>{t("Actions")}</span><span>{t("Method")}</span><span>{t("Status")}</span><span className="text-right">{t("Actions")}</span>
        </div>
        <ScrollArea className="h-[280px]">
          {completedJobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[280px] text-neutral-300">
              <Brain className="h-6 w-6 opacity-30 mb-2" />
              <p className="text-[12px] font-medium">{t("No training history")}</p>
            </div>
          ) : completedJobs.map((job) => {
            const children = job.children ?? [];
            const hasMultipleChildren = children.length > 1;
            const isExpanded = !!expandedJobs[job.id];
            // Classify what KIND of sub-artifacts this parent holds. Helps
            // the summary text differentiate "2 rounds (independent models)" from
            // "3 checkpoints (snapshots taken during training)" — very different meanings.
            const roundCount = children.filter((c) => c.name.startsWith("round_")).length;
            const checkpointCount = children.filter((c) => c.name.startsWith("checkpoint-")).length;
            const summaryParts: string[] = [];
            if (roundCount > 0) summaryParts.push(tp("{0} rounds", roundCount));
            if (checkpointCount > 0) summaryParts.push(tp("{0} checkpoints", checkpointCount));
            const hasRoundsAndCheckpoints = roundCount > 0 && checkpointCount > 0;
            const tooltipHint = roundCount > 1
              ? t("Expand by round (each round is an independent run that differs only in seed)")
              : checkpointCount > 1
              ? t("Expand by checkpoint (snapshots saved during training)")
              : t("Expand");
            return (
            <React.Fragment key={job.id}>
            <div className="grid grid-cols-[2fr_1fr_1fr_auto] items-center gap-4 px-4 py-3 border-b border-neutral-50 last:border-0 hover:bg-neutral-50 transition-colors">
              <div className="min-w-0 flex items-center gap-1.5">
                {hasMultipleChildren && (
                  <button
                    onClick={() => setExpandedJobs((p) => ({ ...p, [job.id]: !p[job.id] }))}
                    title={isExpanded ? t("Collapse") : tooltipHint}
                    className="flex-shrink-0 w-4 h-4 rounded text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100 inline-flex items-center justify-center text-[10px]"
                  >
                    {isExpanded ? "▾" : "▸"}
                  </button>
                )}
                {!hasMultipleChildren && <span className="w-4 flex-shrink-0" />}
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate">{job.id}</div>
                  <div className="text-[9px] text-neutral-400 flex items-center gap-1 flex-wrap">
                    <span>{(job.result as any)?.location ?? ""}</span>
                    {(job.result as any)?.size_mb != null && <span>· {formatSize((job.result as any).size_mb)}</span>}
                    {summaryParts.length > 0 && <span>· {summaryParts.join(" + ")}</span>}
                    {hasRoundsAndCheckpoints && (
                      <span className="text-neutral-300">{t("(round = independent model, checkpoint = snapshot during training)")}</span>
                    )}
                  </div>
                </div>
              </div>
              <span className="text-xs text-neutral-500 flex items-center gap-1.5">
                {methodLabel(job.method)}
              </span>
              <StatusBadge status={job.status} />
              <div className="flex items-center gap-1 justify-end">
                <button onClick={() => openParams(job)} className="p-1.5 rounded-md text-neutral-300 hover:text-neutral-700 hover:bg-neutral-100" title={t("Parameter")}>
                  <Settings className="w-3.5 h-3.5" />
                </button>
                <button onClick={() => openResult(job)} className="p-1.5 rounded-md text-neutral-300 hover:text-neutral-700 hover:bg-neutral-100" title={t("Result")}>
                  <BarChart2 className="w-3.5 h-3.5" />
                </button>
                <button onClick={() => handleDeleteJob(job.id)} className="p-1.5 rounded-md text-neutral-300 hover:text-red-600 hover:bg-red-50" title={hasMultipleChildren ? t("Delete all rounds") : t("Delete")}>
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            {hasMultipleChildren && isExpanded && job.children!.map((child) => {
              // Classify the child by its folder name prefix:
              //   round_N       → one of N independent full trainings
              //                   (different seed, own model.safetensors).
              //   checkpoint-N  → interim snapshot saved during one run
              //                   (same seed, partial), not a distinct model.
              //   final         → final output of a single-run training.
              const isRound = child.name.startsWith("round_");
              const isCheckpoint = child.name.startsWith("checkpoint-");
              const isFinal = child.name === "final";
              const roundNum = isRound ? child.name.replace("round_", "") : "";
              const ckptStep = isCheckpoint ? child.name.replace("checkpoint-", "") : "";
              const childLabel =
                isRound ? `Round ${roundNum}` :
                isCheckpoint ? tp("Checkpoint (step {0})", ckptStep) :
                isFinal ? t("Final model") : child.name;
              const typeLabel =
                isRound ? tp("{0} · independent model (per seed)", child.is_lora ? "LoRA" : "FFT") :
                isCheckpoint ? tp("Mid-training snapshot ({0})", child.is_lora ? "LoRA" : "FFT") :
                isFinal ? tp("Final model ({0})", child.is_lora ? "LoRA" : "FFT") :
                child.is_lora ? t("LoRA adapter") : t("FFT model");
              const typeCls =
                isRound ? "text-neutral-700" :
                isCheckpoint ? "text-neutral-400" :
                "text-neutral-500";
              const confirmTarget = isRound
                ? tp("Round {0} (independent training result)", roundNum)
                : isCheckpoint ? tp("step {0} checkpoint", ckptStep)
                : child.name;
              return (
                <div
                  key={`${job.id}/${child.name}`}
                  className="grid grid-cols-[2fr_1fr_1fr_auto] items-center gap-4 px-4 py-2 border-b border-neutral-50 last:border-0 bg-neutral-50/40 hover:bg-neutral-50 transition-colors"
                >
                  <div className="min-w-0 flex items-center gap-1.5 pl-6">
                    <span className="text-neutral-300 text-[10px]">└</span>
                    <div className="min-w-0 flex-1">
                      <div className="text-[12px] font-medium text-neutral-700 truncate flex items-center gap-1.5">
                        {childLabel}
                        {isRound && (
                          <span className="inline-flex items-center px-1 py-0 rounded text-[8.5px] font-bold bg-indigo-50 text-indigo-600 border border-indigo-200">
                            Round
                          </span>
                        )}
                        {isCheckpoint && (
                          <span className="inline-flex items-center px-1 py-0 rounded text-[8.5px] font-bold bg-neutral-100 text-neutral-500 border border-neutral-200">
                            Checkpoint
                          </span>
                        )}
                      </div>
                      <div className="text-[9px] text-neutral-400 font-mono truncate">{child.path}</div>
                    </div>
                  </div>
                  <span className="text-[10px] text-neutral-400">{formatSize(child.size_mb)}</span>
                  <span className={`text-[10px] ${typeCls}`}>{typeLabel}</span>
                  <div className="flex items-center gap-1 justify-end">
                    <button
                      onClick={async () => {
                        if (!confirm(tp("Deletes only '{0} · {1}'. The remaining {2} are kept.", job.id, confirmTarget, isRound ? "round" : "Checkpoint"))) return;
                        try {
                          const r = await apiFetch(
                            `/api/train/jobs/${encodeURIComponent(job.id)}?path=${encodeURIComponent(child.path)}`,
                            { method: "DELETE" },
                          );
                          if (!r.ok) {
                            const d = await r.json().catch(() => ({}));
                            toast.error(tp("Delete failed: {0}", (d as { detail?: string }).detail ?? `HTTP ${r.status}`));
                            return;
                          }
                          toast.success(tp("'{0}' deleted", confirmTarget));
                          await refreshCheckpoints();
                        } catch (e) {
                          toast.error(tp("Delete failed: {0}", (e as Error).message));
                        }
                      }}
                      className="p-1.5 rounded-md text-neutral-300 hover:text-red-600 hover:bg-red-50"
                      title={isRound ? t("Delete only this round (keep the others)") : t("Delete only this checkpoint")}
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })}
            </React.Fragment>
            );
          })}
        </ScrollArea>
      </div>

      {/* Param modal — structured view mirroring the data-generation params modal */}
      <Dialog open={paramOpen} onOpenChange={setParamOpen}>
        <DialogContent className="max-w-[560px]">
          <DialogHeader>
            <div className="flex flex-col gap-1">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-neutral-400">TRAINING PARAMETERS</p>
              <DialogTitle className="text-[14px] font-bold text-neutral-900 truncate">{paramTitle}</DialogTitle>
            </div>
            <DialogDescription className="sr-only">{t("Training job status and effective parameters")}</DialogDescription>
          </DialogHeader>

          {Object.keys(paramKV).length === 0 ? (
            <p className="text-[12px] text-neutral-400 text-center py-6">{t("No parameter record (artifact from an earlier session)")}</p>
          ) : (
            (() => {
              const job = trainJobs.find((j) => j.id === paramTitle);
              const method = String((paramKV as any).method ?? job?.method ?? "").toLowerCase();
              const isLora = method === "lora";
              const lr = Number((paramKV as any).learning_rate ?? (paramKV as any).lr);
              const fmt = (v: unknown, fallback = "-"): string => {
                if (v === null || v === undefined || v === "") return fallback;
                if (typeof v === "object") return JSON.stringify(v);
                return String(v);
              };
              return (
                <div className="max-h-[520px] overflow-y-auto space-y-4 pr-1">
                  {/* ── Job status ── */}
                  {job && (
                    <ParamSection title={t("Job Status")}>
                      <ParamRow label={t("Status")} value={String(job.status)} highlight />
                      <ParamRow label={t("Job ID")} value={job.id} mono />
                      <ParamRow label={t("Progress")} value={`${job.progress ?? 0}%`} />
                      {job.message && <ParamRow label={t("Message")} value={t(job.message)} />}
                      {(job.result as any)?.path && (() => {
                        const p: string = (job.result as any).path;
                        const display = /\/round_\d+$/.test(p) ? p.replace(/\/round_\d+$/, "") : p;
                        return <ParamRow label={t("Output path")} value={display} mono />;
                      })()}
                      {(job.result as any)?.size_mb != null && (
                        <ParamRow label={t("Size")} value={`${formatNumber(Number((job.result as any).size_mb))} MB`} />
                      )}
                    </ParamSection>
                  )}

                  {/* ── Run parameters ── */}
                <ParamSection title={t("Effective Parameters")}>
                  <ParamRow label={t("Method")} value={methodLabel(method)} highlight />
                  {(paramKV as any).model_name && (
                    <ParamRow label={t("Base model")} value={fmt((paramKV as any).model_name)} mono />
                  )}
                  {(paramKV as any).dataset_name && (
                    <ParamRow label={t("Dataset")} value={fmt((paramKV as any).dataset_name)} mono />
                  )}
                  {(paramKV as any).epochs != null && (
                    <ParamRow label={t("Epochs")} value={fmt((paramKV as any).epochs)} />
                  )}
                  {(paramKV as any).max_seq_length != null && (
                    <ParamRow label={t("Max sequence length")} value={formatNumber(Number((paramKV as any).max_seq_length))} />
                  )}
                  {(paramKV as any).batch_size != null && (
                    <ParamRow label={t("Batch size")} value={fmt((paramKV as any).batch_size)} />
                  )}
                  {Number.isFinite(lr) && (
                    <ParamRow label={t("Learning rate")} value={lr.toExponential(1)} mono />
                  )}
                </ParamSection>

                  {/* ── LoRA settings (LoRA mode only) ── */}
                  {isLora && (
                    <ParamSection title={t("LoRA Settings")}>
                      {(paramKV as any).lora_r != null && (
                        <ParamRow label="Rank (r)" value={fmt((paramKV as any).lora_r)} />
                      )}
                      {(paramKV as any).lora_alpha != null && (
                        <ParamRow label="Alpha" value={fmt((paramKV as any).lora_alpha)} />
                      )}
                      {(paramKV as any).lora_dropout != null && (
                        <ParamRow label="Dropout" value={fmt((paramKV as any).lora_dropout)} />
                      )}
                    </ParamSection>
                  )}

                </div>
              );
            })()
          )}

          <DialogFooter>
            <Button className="bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg" onClick={() => setParamOpen(false)}>{t("Close")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Result modal */}
      <Dialog open={resultOpen} onOpenChange={setResultOpen}>
        <DialogContent className="sm:max-w-6xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{tp("Training result: {0}", resultJob?.id)}</DialogTitle>
            <DialogDescription>{t("Final metrics and charts for the training run.")}</DialogDescription>
          </DialogHeader>

          {resultLoading ? (
            <div className="flex items-center justify-center h-[200px] text-muted-foreground">
              <Loader2 className="w-6 h-6 animate-spin mr-3" /> {t("Loading result data...")}
            </div>
          ) : resultDetail ? (
            <div className="space-y-4">
              <Card className="border border-neutral-200/60 shadow-sm">
                <CardContent className="p-4">
                  <dl className="space-y-3">
                    <div className="flex justify-between items-center">
                      <dt className="text-muted-foreground text-sm">Train Runtime</dt>
                      <dd className="text-sm font-mono text-foreground">{runtimeStr}</dd>
                    </div>
                    <div className="flex justify-between items-center">
                      <dt className="text-muted-foreground text-sm">Train Loss</dt>
                      <dd className="text-sm font-mono text-foreground">{typeof trainResult.train_loss === "number" ? trainResult.train_loss.toFixed(4) : "-"}</dd>
                    </div>
                    <div className="flex justify-between items-center">
                      <dt className="text-muted-foreground text-sm">Global Steps</dt>
                      <dd className="text-sm font-mono text-foreground">{trainResult.global_step ?? "-"}</dd>
                    </div>
                    <div className="flex justify-between items-center">
                      <dt className="text-muted-foreground text-sm">Epochs</dt>
                      <dd className="text-sm font-mono text-foreground">{trainResult.epochs_completed ?? "-"}</dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              {chartData.length > 0 ? (
                <Tabs defaultValue="loss" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="loss">Loss</TabsTrigger>
                    <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
                  </TabsList>

                  {/* ── Loss ─────────────────────── */}
                  <TabsContent value="loss">
                    <Card>
                      <CardHeader>
                        <CardTitle>{t("Loss change")}</CardTitle>
                        <CardDescription>
                          {t("Loss per epoch.")}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer className="h-48 w-full" config={{ loss: { label: "Loss", color: "hsl(var(--chart-1))" } }}>
                          <LineChart accessibilityLayer data={chartData} margin={{ left: 12, right: 12 }}>
                            <CartesianGrid vertical={false} />
                            <XAxis dataKey="epoch" type="number" tickLine={false} axisLine={false} tickFormatter={(v) => `Epoch ${v}`} domain={["dataMin", "dataMax"]} ticks={epochTicks} />
                            <YAxis dataKey="loss" type="number" tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} domain={["auto", "auto"]} />
                            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                            <Line dataKey="loss" type="monotone" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ChartContainer>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  {/* ── Accuracy ─────────────────── */}
                  <TabsContent value="accuracy">
                    <Card>
                      <CardHeader><CardTitle>{t("Accuracy change")}</CardTitle><CardDescription>{t("Mean token accuracy per epoch.")}</CardDescription></CardHeader>
                      <CardContent>
                        <ChartContainer className="h-48 w-full" config={{ accuracy: { label: "Accuracy", color: "hsl(var(--chart-2))" } }}>
                          <LineChart accessibilityLayer data={chartData} margin={{ left: 12, right: 12 }}>
                            <CartesianGrid vertical={false} />
                            <XAxis dataKey="epoch" type="number" tickLine={false} axisLine={false} tickFormatter={(v) => `Epoch ${v}`} domain={["dataMin", "dataMax"]} ticks={epochTicks} />
                            <YAxis dataKey="accuracy" type="number" tickLine={false} axisLine={false} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} domain={[0, 1]} />
                            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                            <Line dataKey="accuracy" type="monotone" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ChartContainer>
                      </CardContent>
                    </Card>
                  </TabsContent>

                </Tabs>
              ) : (
                <div className="text-[11px] text-neutral-300 text-center py-4">{t("No training log data (short runs may not record logs)")}</div>
              )}

            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-8">{t("No result data")}</p>
          )}
          <DialogFooter className="mt-4">
            <Button className="bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg" onClick={() => setResultOpen(false)}>{t("Close")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/* ── Shared param modal components ─────────────────────────────
   Mirrors the data-generation params modal look (DATASET PREVIEW /
   GENERATION PARAMETERS style — "Job Status" + "Effective Parameters" sections). */

function ParamSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-[10px] font-bold uppercase tracking-wider text-neutral-400 mb-2">{title}</p>
      <div className="border border-neutral-100 rounded-lg overflow-hidden">{children}</div>
    </div>
  );
}

function ParamRow({ label, value, mono, highlight }: { label: string; value: string; mono?: boolean; highlight?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-3 px-3 py-2 border-b last:border-0 border-neutral-100 text-[12px]">
      <span className="text-neutral-500 flex-shrink-0">{label}</span>
      <span className={`text-right break-all ${mono ? "font-mono text-[11px]" : ""} ${highlight ? "font-bold text-neutral-900" : "text-neutral-700"}`}>
        {value}
      </span>
    </div>
  );
}
