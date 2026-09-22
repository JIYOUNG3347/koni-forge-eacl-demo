/**
 * GuidedChainCard — Pending-chain approval card for Guided mode.
 *
 * Shown when the dispatcher has a pending_chain (next pipeline stage waiting
 * for user confirmation). Offers approve / skip / pause buttons that drive
 * /api/agent/chain/advance and /api/agent/chain/{pause,resume}.
 */

import React from "react";
import { ArrowRight, SkipForward, Pause, Play, Loader2 } from "lucide-react";
import type { PendingChain } from "../../stores/agentStore";
import { tp, useT } from "../../i18n";

const AGENT_LABEL: Record<string, string> = {
  retrieval: "RAG Indexing",
  boundary: "KBD Analysis",
  tuning: "Training",
};

export function agentLabel(name: string): string {
  return AGENT_LABEL[name] ?? name;
}

interface GuidedChainCardProps {
  pending: PendingChain;
  chainPaused: boolean;
  isBusy: boolean;
  onApprove: () => void;
  onSkip: () => void;
  onPauseToggle: () => void;
}

export function GuidedChainCard({
  pending,
  chainPaused,
  isBusy,
  onApprove,
  onSkip,
  onPauseToggle,
}: GuidedChainCardProps) {
  const t = useT();
  const nextLabel = t(pending.nextAgent ? agentLabel(pending.nextAgent) : "Next stage");

  return (
    <div className="mx-3 my-2 rounded-xl border border-blue-200 bg-blue-50/60 p-3 shadow-sm">
      <div className="flex items-center gap-1.5 mb-1.5">
        <ArrowRight className="w-3.5 h-3.5 text-blue-600" />
        <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wide">
          {tp("Next step · {0}", nextLabel)}
        </span>
      </div>
      <p className="text-[12px] leading-relaxed text-neutral-700 whitespace-pre-wrap mb-2.5">
        {pending.announce}
      </p>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={onApprove}
          disabled={isBusy}
          className="flex-1 inline-flex items-center justify-center gap-1 h-7 rounded-lg bg-blue-600 text-white text-[11px] font-semibold hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {isBusy ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <ArrowRight className="w-3 h-3" />
          )}
          {t("Proceed")}
        </button>
        <button
          type="button"
          onClick={onSkip}
          disabled={isBusy}
          className="flex-1 inline-flex items-center justify-center gap-1 h-7 rounded-lg border border-neutral-300 bg-white text-[11px] font-semibold text-neutral-600 hover:bg-neutral-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <SkipForward className="w-3 h-3" />{t("Skip")}</button>
        <button
          type="button"
          onClick={onPauseToggle}
          disabled={isBusy}
          title={chainPaused ? t("Resume chaining") : t("Pause chaining")}
          className="inline-flex items-center justify-center w-7 h-7 rounded-lg border border-neutral-300 bg-white text-neutral-500 hover:bg-neutral-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {chainPaused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
        </button>
      </div>
    </div>
  );
}
