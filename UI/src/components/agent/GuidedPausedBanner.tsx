
import { Pause, Play } from "lucide-react";
import type { PendingChain } from "../../stores/agentStore";
import { agentLabel } from "./GuidedChainCard";
import { tp, useT } from "../../i18n";

interface GuidedPausedBannerProps {
  pending: PendingChain;
  isBusy: boolean;
  onResume: () => void;
}

export function GuidedPausedBanner({ pending, isBusy, onResume }: GuidedPausedBannerProps) {
  const t = useT();
  const nextLabel = pending.nextAgent ? agentLabel(pending.nextAgent) : "Next stage";

  return (
    <div className="mx-3 my-2 rounded-lg border border-neutral-200 border-l-2 border-l-neutral-400 bg-gradient-to-b from-neutral-50 to-neutral-100 px-3 py-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 mb-1">
            <Pause className="w-3.5 h-3.5 text-neutral-500" />
            <span className="text-[11px] font-bold text-neutral-600 uppercase tracking-wide">
              {t("Paused")}
            </span>
          </div>
          <p className="text-[12px] leading-relaxed text-neutral-600">
            {t("Describe the next step in plain language, or press Resume to continue.")}
          </p>
          <p className="text-[11px] italic text-neutral-400 mt-1">
            {tp("e.g. “run {0}”", nextLabel)}
          </p>
        </div>
        <button
          type="button"
          onClick={onResume}
          disabled={isBusy}
          title="Resume chaining"
          className="flex-shrink-0 inline-flex items-center gap-1 h-[26px] px-2.5 rounded-lg border border-neutral-300 bg-white text-[11px] font-semibold text-neutral-700 hover:bg-neutral-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Play className="w-3 h-3" />
          {t("Resume")}
        </button>
      </div>
    </div>
  );
}
