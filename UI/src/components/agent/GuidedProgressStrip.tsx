/**
 * GuidedProgressStrip — compact horizontal strip of the pipeline stages.
 *
 * Rendered in Guided mode at the top of the Orchestrator panel. Uses
 * `agentStore.pipelineStatus` (completedStages / currentStage / busyAgent /
 * pendingChain / session.stages).
 *
 * Hovering a stage reveals its metadata (coverage, chunk count, etc.).
 */

import React from "react";
import { Database, Shield, Brain } from "lucide-react";
import type { PipelineStatus, PipelineSessionStage } from "../../stores/agentStore";
import { t, tp, useT } from "../../i18n";

type StageKey = "retrieval" | "kbd" | "tuning";

const STAGES: { key: StageKey; label: string; icon: React.ElementType }[] = [
  { key: "retrieval", label: "RAG",  icon: Database },
  { key: "kbd",       label: "KBD",  icon: Shield },
  { key: "tuning",    label: "Train", icon: Brain },
];

const AGENT_TO_STAGE: Record<string, StageKey> = {
  retrieval: "retrieval",
  boundary: "kbd",
  kbd: "kbd",
  tuning: "tuning",
};

function normalizeStageKey(raw: string | null | undefined): StageKey | null {
  if (!raw) return null;
  return AGENT_TO_STAGE[raw] ?? null;
}

type StageState = "done" | "skipped" | "running" | "pending" | "idle";

interface ResolvedStage {
  state: StageState;
  tooltip: string;
}

function formatTooltip(stage: StageKey, state: StageState, entry?: PipelineSessionStage, announce?: string): string {
  const base = t(STAGES.find((s) => s.key === stage)?.label ?? stage);
  if (state === "idle") return tp("{0} · idle", base);
  if (state === "pending") return announce ? `${base} · ${announce}` : tp("{0} · waiting", base);
  if (state === "running") return tp("{0} · running", base);
  if (state === "skipped") return tp("{0} · skipped", base);

  // done — metadata per stage
  if (!entry) return tp("{0} · done", base);
  switch (stage) {
    case "retrieval": {
      const n = entry.indexed_count as number | undefined;
      return n ? tp("{0} · {1} chunks", base, n) : tp("{0} · done", base);
    }
    case "kbd": {
      const cov = entry.coverage_pct as number | undefined;
      const rec = entry.recommendation as string | undefined;
      const parts = [];
      if (cov !== undefined) parts.push(tp("coverage {0}%", cov));
      if (rec) parts.push(rec);
      return `${base} · ${parts.join(", ") || "Completed"}`;
    }
    case "tuning": {
      const m = entry.model_name as string | undefined;
      return m ? `${base} · ${m}` : tp("{0} · done", base);
    }
  }
}

/** State precedence for merging multiple backend stages into one display row.
 *  Higher wins: a running sub-step visually dominates a done one. */
const STATE_RANK: Record<StageState, number> = {
  idle: 0, skipped: 1, done: 2, pending: 3, running: 4,
};

function resolve(status: PipelineStatus): Record<StageKey, ResolvedStage & { entry?: PipelineSessionStage }> {
  const out = {} as Record<StageKey, ResolvedStage & { entry?: PipelineSessionStage }>;
  for (const { key } of STAGES) {
    out[key] = { state: "idle", tooltip: formatTooltip(key, "idle") };
  }

  const upgrade = (key: StageKey, candidate: ResolvedStage & { entry?: PipelineSessionStage }) => {
    const prev = out[key];
    if (STATE_RANK[candidate.state] >= STATE_RANK[prev.state]) {
      out[key] = candidate;
    }
  };

  for (const entry of status.session.stages ?? []) {
    const key = normalizeStageKey(entry.stage);
    if (!key) continue;
    const state: StageState = entry.skipped ? "skipped" : "done";
    upgrade(key, { state, entry, tooltip: formatTooltip(key, state, entry) });
  }

  for (const name of status.completedStages ?? []) {
    const key = normalizeStageKey(name);
    if (!key) continue;
    upgrade(key, { state: "done", tooltip: formatTooltip(key, "done") });
  }

  const busyKey = normalizeStageKey(status.busyAgent) ?? normalizeStageKey(status.currentStage);
  if (status.isBusy && busyKey && out[busyKey].state !== "done" && out[busyKey].state !== "skipped") {
    upgrade(busyKey, { state: "running", tooltip: formatTooltip(busyKey, "running") });
  }

  const pendingKey = normalizeStageKey(status.pendingChain?.nextAgent);
  if (pendingKey) {
    upgrade(pendingKey, {
      state: "pending",
      tooltip: formatTooltip(pendingKey, "pending", undefined, status.pendingChain?.announce),
    });
  }

  return out;
}

export function GuidedProgressStrip({ status }: { status: PipelineStatus }) {
  const t = useT();
  const resolved = resolve(status);

  const hasActivity =
    (status.session.stages?.length ?? 0) > 0 ||
    status.completedStages.length > 0 ||
    !!status.pendingChain ||
    status.isBusy;

  return (
    <div className="border-b border-neutral-100 bg-neutral-50/50 px-3 py-2">
      <div className="flex items-center gap-0 justify-between">
        {STAGES.map(({ key, label, icon: Icon }, idx) => {
          const info = resolved[key];
          const isLast = idx === STAGES.length - 1;
          const nextInfo = !isLast ? resolved[STAGES[idx + 1].key] : null;

          const bubble =
            info.state === "done"    ? "bg-neutral-900 text-white border-neutral-900"
            : info.state === "skipped" ? "bg-neutral-200 text-neutral-400 border-neutral-200"
            : info.state === "running" ? "bg-neutral-900 text-white border-neutral-900 animate-pulse"
            : info.state === "pending" ? "bg-white text-neutral-900 border-neutral-900 ring-2 ring-neutral-200"
            : "bg-white text-neutral-300 border-neutral-200";

          const labelTone =
            info.state === "done"    ? "text-neutral-800 font-semibold"
            : info.state === "running" ? "text-neutral-900 font-bold"
            : info.state === "pending" ? "text-neutral-900 font-bold"
            : info.state === "skipped" ? "text-neutral-300 line-through"
            : "text-neutral-400";

          // Connector: darker if both done; lighter otherwise
          const connector =
            info.state === "done" && nextInfo && (nextInfo.state === "done" || nextInfo.state === "skipped")
              ? "bg-neutral-800"
              : info.state === "done"
              ? "bg-neutral-400"
              : info.state === "running" || info.state === "pending"
              ? "bg-neutral-200"
              : "bg-neutral-100";

          return (
            <React.Fragment key={key}>
              <div className="flex flex-col items-center gap-0.5 flex-shrink-0" title={info.tooltip}>
                <div className={`w-6 h-6 rounded-full border flex items-center justify-center ${bubble}`}>
                  <Icon className="w-3 h-3" />
                </div>
                <span className={`text-[8.5px] mt-0.5 ${labelTone}`}>{t(label)}</span>
              </div>
              {!isLast && <div className={`flex-1 h-0.5 mb-3 mx-1 ${connector}`} />}
            </React.Fragment>
          );
        })}
      </div>
      {!hasActivity && (
        <p className="text-[9.5px] text-neutral-400 mt-1 text-center">
          {t("Upload data, or start the orchestrator from the chat")}
        </p>
      )}
      {hasActivity && status.pendingChain && (
        <p className="text-[10px] text-neutral-700 mt-1 text-center font-semibold truncate">
          {tp("Next: {0}", status.pendingChain.announce)}
        </p>
      )}
      {hasActivity && !status.pendingChain && status.isBusy && (
        <p className="text-[10px] text-neutral-700 mt-1 text-center font-medium">
          {status.busyAgent ? tp("{0} working...", status.busyAgent) : t("In progress...")}
        </p>
      )}
    </div>
  );
}
