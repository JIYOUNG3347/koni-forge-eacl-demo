
import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Send, Bot, User, Loader2,
  Eraser, Wrench, StopCircle,
  } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAgentStore, type ChatMessage } from "../stores/agentStore";
import { useSystemStore } from "../stores/systemStore";
import { GuidedChainCard } from "./agent/GuidedChainCard";
import { GuidedPausedBanner } from "./agent/GuidedPausedBanner";
import { GuidedProgressStrip } from "./agent/GuidedProgressStrip";
import { GuidedConfirmParamsModal } from "./agent/GuidedConfirmParamsModal";
import { OptionPicker, parseAgentOptions, type AgentOption } from "./agent/OptionPicker";
import Markdown from "react-markdown";
import { tp, useT } from "../i18n";
import { formatTime as fmtTime } from "../i18n/locale";

/* ── Constants ── */
const AGENT_TO_ROUTE: Record<string, string> = {
  retrieval: "/data",
  boundary: "/data",
  tuning: "/train",
};

/* Pipeline-runner event labels (Auto mode) */
function formatTime(ts: number) {
  return fmtTime(ts, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}


export function AgentSidebar() {
  const t = useT();
  /* ── Store wiring ── */
  const messages = useAgentStore((s) => s.messages);
  const fetchAgentMessages = useAgentStore((s) => s.fetchAgentMessages);
  const resetSession = useAgentStore((s) => s.resetSession);
  const dispatchMessage = useAgentStore((s) => s.dispatchMessage);
  const advanceChain = useAgentStore((s) => s.advanceChain);
  const confirmExecution = useAgentStore((s) => s.confirmExecution);
  const pauseChain = useAgentStore((s) => s.pauseChain);
  const resumeChain = useAgentStore((s) => s.resumeChain);
  const fetchPipelineStatus = useAgentStore((s) => s.fetchPipelineStatus);
  const pipelineStatus = useAgentStore((s) => s.pipelineStatus);
  const navigateTarget = useAgentStore((s) => s.navigateTarget);
  const clearNavigateTarget = useAgentStore((s) => s.clearNavigateTarget);
  const storeStreaming = useAgentStore((s) => s.isStreaming);
  const storeAbort = useAgentStore((s) => s.abort);
  const agentAutonomy = useSystemStore((s) => s.systemSettings.agentAutonomy);
  const isGuided = agentAutonomy === "guided";

  const navigate = useNavigate();

  /* ── Local state ── */
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const streaming = storeStreaming;

  /* ── Mount: load agent messages from server ── */
  useEffect(() => {
    void fetchAgentMessages();
  }, [fetchAgentMessages]);

  /* ── Guided: poll dispatcher pipeline status as a safety net ── */
  useEffect(() => {
    if (!isGuided) return;
    fetchPipelineStatus();
    const id = window.setInterval(() => {
      if (!useAgentStore.getState().isStreaming) fetchPipelineStatus();
    }, 8000);
    return () => window.clearInterval(id);
  }, [isGuided, fetchPipelineStatus]);

  /* ── Guided: route on dispatcher [navigate:X] markers ── */
  useEffect(() => {
    if (!isGuided || !navigateTarget) return;
    const route = AGENT_TO_ROUTE[navigateTarget];
    if (route) navigate(route);
    clearNavigateTarget();
  }, [isGuided, navigateTarget, navigate, clearNavigateTarget]);

  /* ── Auto-scroll ── */
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages.length, streaming, pipelineStatus.pendingChain]);

  // Confirmation modal state. Opening the modal snapshots pendingChain so it is
  // decoupled from store updates: the store's pendingChain is re-derived on every
  // stream or poll (normalizePipelineStatus) and can blink to null, which would
  // unmount and remount the modal, cancelling its model-list fetch every time and
  // leaving it stuck on "Loading". The snapshot breaks that remount loop.
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmSnapshot, setConfirmSnapshot] = useState<{
    params: Record<string, unknown>;
    stage: string;
    announce?: string;
  } | null>(null);

  const openConfirmModal = useCallback((): boolean => {
    const pc = pipelineStatus.pendingChain;
    const p = pc?.params;
    if (!pc || !p || typeof p !== "object" || Object.keys(p).length === 0) return false;
    setConfirmSnapshot({
      params: p as Record<string, unknown>,
      stage: ((p as Record<string, unknown>).stage as string) || pc.nextAgent,
      announce: pc.announce,
    });
    setConfirmOpen(true);
    return true;
  }, [pipelineStatus.pendingChain]);

  const closeConfirmModal = useCallback(() => {
    setConfirmOpen(false);
    setConfirmSnapshot(null);
  }, []);

  /* ── Send handlers (mode-aware) ── */
  const handleSendGuided = useCallback(async () => {
    if (!input.trim() || streaming) return;
    const text = input.trim();
    // Text confirmation (e.g. "Proceeding") on a gate that carries
    // recommended_params must NOT bypass the edit modal — otherwise the
    // specialist fires with baked defaults and the user never gets to tweak
    // base_model/epochs etc. Intercept here and open the same modal the
    // "Proceed" button opens, then clear the input.
    const confirmRe = /^(yes|y|ok|okay|go|sure|proceed(ing)?|start|continue)[.!?~\s]*$/i;
    const params = pipelineStatus.pendingChain?.params;
    if (
      confirmRe.test(text) &&
      params && typeof params === "object" && Object.keys(params).length > 0
    ) {
      setInput("");
      openConfirmModal();
      return;
    }
    setInput("");
    await dispatchMessage(text);
  }, [input, streaming, dispatchMessage, pipelineStatus.pendingChain, openConfirmModal]);

  const handleSendAuto = useCallback(async () => {
    if (!input.trim() || streaming) return;
    const text = input.trim();
    setInput("");
    await dispatchMessage(text);
  }, [input, streaming, dispatchMessage]);

  const handleSend = isGuided ? handleSendGuided : handleSendAuto;

  const handleApprove = useCallback(() => {
    if (openConfirmModal()) return;
    void advanceChain("approve");
  }, [advanceChain, openConfirmModal]);

  const handleConfirmParams = useCallback(async (overrides?: Record<string, unknown>) => {
    closeConfirmModal();
    await advanceChain("approve", overrides);
  }, [advanceChain, closeConfirmModal]);

  const handleConfirmExecution = useCallback(
    async (overrides?: Record<string, unknown>) => {
      await confirmExecution("approve", overrides);
    },
    [confirmExecution],
  );
  const handleCancelExecution = useCallback(() => {
    void confirmExecution("cancel");
  }, [confirmExecution]);

  const handleSkip = useCallback(() => { void advanceChain("skip"); }, [advanceChain]);
  const handlePauseToggle = useCallback(() => {
    if (pipelineStatus.chainPaused) void resumeChain(); else void pauseChain();
  }, [pipelineStatus.chainPaused, pauseChain, resumeChain]);

  /* Identify the "live" option picker: the LAST assistant message that has
     Option blocks, with no subsequent user message after it. Older option
     panels stay visible (as history) but are disabled. */
  const activePickerMsgId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "user") return null;
      if (m.role === "assistant" && parseAgentOptions(m.content).length > 0) return m.id;
    }
    return null;
  }, [messages]);

  const handleOptionPick = useCallback((opt: AgentOption) => {
    if (streaming) return;
    void dispatchMessage(`Option ${opt.number}`);
  }, [streaming, dispatchMessage]);

  const stream = useMemo(
    () => [...messages].sort((a, b) => a.timestamp - b.timestamp),
    [messages],
  );

  const isEmpty =
    stream.length === 0 &&
    !(isGuided && pipelineStatus.pendingChain);

  return (
    <div className="flex flex-col h-full bg-white border-l border-neutral-200">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-neutral-100 bg-neutral-900 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-white/70" />
          <span className="text-[12px] font-bold text-white">Orchestrator</span>
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-white/10 text-white/80 border border-white/15">
            {isGuided ? "GUIDED" : "AUTO"}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              if (window.confirm(
                t("Reset the conversation and the pipeline progress (completed stages, pending cards, recommendations)?\n\n") +
                t("Datasets, models and saved artifacts are not deleted.")
              )) {
                void resetSession();
              }
            }}
            title={t("Reset conversation and progress")}
            className="p-1 rounded text-white/40 hover:text-white hover:bg-white/10"
          >
            <Eraser className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Guided progress strip — only in Guided mode */}
      {isGuided && <GuidedProgressStrip status={pipelineStatus} />}

      {/* Unified scroll area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full text-neutral-300">
            <Bot className="w-8 h-8 opacity-30 mb-2" />
            <p className="text-[11px] font-medium">
              {isGuided ? t("Start the pipeline") : t("Ask the agent")}
            </p>
            <p className="text-[10px] text-neutral-200 mt-1 text-center">
              {isGuided
                ? <>{t("Upload a document, or")}<br />{t("start a step-by-step run from the chat")}</>
                : <>{t("Data analysis, training setup,")}<br />{t("and help interpret evaluation results")}</>}
            </p>
          </div>
        ) : (
          <>
            {stream.map((msg) =>
              renderMessage(msg, {
                    isActivePicker: msg.id === activePickerMsgId,
                    pickerBusy: streaming,
                    onPick: handleOptionPick,
                  })
            )}
            {isGuided &&
              pipelineStatus.pendingChain &&
              !pipelineStatus.chainPaused && (
                <GuidedChainCard
                  pending={pipelineStatus.pendingChain}
                  chainPaused={pipelineStatus.chainPaused}
                  isBusy={streaming}
                  onApprove={handleApprove}
                  onSkip={handleSkip}
                  onPauseToggle={handlePauseToggle}
                />
              )}
            {isGuided &&
              pipelineStatus.pendingChain &&
              pipelineStatus.chainPaused && (
                <GuidedPausedBanner
                  pending={pipelineStatus.pendingChain}
                  isBusy={streaming}
                  onResume={resumeChain}
                />
              )}
          </>
        )}
        {streaming && (
          <div className="flex gap-2 items-center">
            <div className="w-6 h-6 rounded-lg bg-neutral-100 flex items-center justify-center flex-shrink-0">
              <Bot className="w-3 h-3 text-neutral-500" />
            </div>
            <div className="border-l-2 border-neutral-200 pl-3 py-1 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-neutral-400" />
              {isGuided && pipelineStatus.busyAgent ? (
                <span className="text-[10px] text-neutral-400">{tp("{0} working...", pipelineStatus.busyAgent)}</span>
              ) : null}
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="px-3 py-2.5 border-t border-neutral-100 flex-shrink-0">
        <div className="relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder={isGuided ? t("Send a message to the agent...") : t("Ask the agent...")}
            rows={2}
            disabled={streaming}
            className="w-full pr-10 rounded-xl border border-neutral-200 px-3 py-2 text-[12px] resize-none focus:outline-none focus:border-neutral-400 focus:ring-1 focus:ring-neutral-200 disabled:opacity-50"
          />
          {streaming && isGuided ? (
            <button onClick={() => storeAbort()} title={t("Stop")}
              className="absolute right-2 bottom-2 w-7 h-7 flex items-center justify-center rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors">
              <StopCircle className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim() || streaming}
              className="absolute right-2 bottom-2 w-7 h-7 flex items-center justify-center rounded-lg bg-neutral-900 text-white disabled:opacity-30 hover:bg-neutral-700 transition-colors">
              <Send className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Parameter confirmation modal — pops up when the user confirms on a
          chain card that carries structured recommended_params. */}
      {isGuided && confirmOpen && confirmSnapshot && (
        <GuidedConfirmParamsModal
          stage={confirmSnapshot.stage}
          params={confirmSnapshot.params}
          announce={confirmSnapshot.announce}
          busy={streaming}
          onConfirm={handleConfirmParams}
          onCancel={closeConfirmModal}
        />
      )}

      {isGuided && pipelineStatus.pendingExecution && (
        <GuidedConfirmParamsModal
          stage={pipelineStatus.pendingExecution.stage}
          params={pipelineStatus.pendingExecution.params}
          announce={t("Proceed with these parameters?")}
          busy={streaming}
          onConfirm={handleConfirmExecution}
          onCancel={handleCancelExecution}
        />
      )}

    </div>
  );
}

/* ── Markdown renderer for assistant messages ── */
function AgentMarkdown({ content }: { content: string }) {
  return (
    <div className={[
      "prose prose-sm max-w-none text-[12px]",
      "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
      "prose-p:text-[12px] prose-p:leading-relaxed prose-p:text-neutral-700 prose-p:my-1",
      "prose-headings:font-bold prose-headings:text-neutral-900",
      "prose-h1:text-[13px] prose-h1:my-1 prose-h2:text-[12.5px] prose-h2:my-1 prose-h3:text-[12px] prose-h3:my-0.5",
      "prose-strong:text-neutral-900 prose-strong:font-semibold",
      "prose-em:text-neutral-600 prose-em:italic",
      "prose-code:text-[10.5px] prose-code:bg-neutral-100 prose-code:text-neutral-700 prose-code:rounded prose-code:px-1 prose-code:font-mono prose-code:before:content-none prose-code:after:content-none",
      "prose-pre:bg-neutral-50 prose-pre:border prose-pre:border-neutral-200 prose-pre:rounded-lg prose-pre:text-[10.5px] prose-pre:my-1.5 prose-pre:p-3",
      "prose-li:text-[12px] prose-li:text-neutral-700 prose-li:my-0",
      "prose-ul:my-1.5 prose-ol:my-1.5",
      "prose-a:text-blue-600 prose-a:no-underline prose-a:font-normal",
      "prose-blockquote:border-neutral-300 prose-blockquote:text-neutral-500 prose-blockquote:not-italic",
      "prose-hr:border-neutral-200 prose-hr:my-2",
    ].join(" ")}>
      <Markdown>{content}</Markdown>
    </div>
  );
}

/* ── Message renderer ── */
interface RenderMessageCtx {
  isActivePicker: boolean;
  pickerBusy: boolean;
  onPick: (opt: AgentOption) => void;
}

function renderMessage(msg: ChatMessage, ctx?: RenderMessageCtx) {
  if (msg.role === "user") {
    return (
      <div key={msg.id} className="flex gap-2 justify-end group">
        <div className="max-w-[85%] bg-neutral-900 text-white rounded-2xl rounded-br-sm px-3 py-2">
          <p className="text-[12px] leading-relaxed whitespace-pre-wrap text-white">{msg.content}</p>
          <p className="text-[9px] mt-1 text-white/40">{formatTime(msg.timestamp)}</p>
        </div>
        <div className="w-6 h-6 rounded-lg bg-neutral-200 flex items-center justify-center flex-shrink-0 mt-0.5">
          <User className="w-3 h-3 text-neutral-600" />
        </div>
      </div>
    );
  }

  if (msg.role === "system") {
    return (
      <div key={msg.id} className="flex gap-1 group mx-2 my-1">
        <div className="flex-1 px-2.5 py-1.5 rounded-lg bg-neutral-50 border border-neutral-100">
          <p className="text-[11px] leading-relaxed whitespace-pre-wrap text-neutral-600">{msg.content}</p>
        </div>
      </div>
    );
  }

  if (msg.role === "tool_result") {
    return (
      <div key={msg.id} className="flex gap-2 group">
        <div className="w-6 h-6 rounded-lg bg-amber-100 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Wrench className="w-3 h-3 text-amber-600" />
        </div>
        <div className="flex-1 min-w-0 border-l-2 border-amber-200 pl-2">
          <p className="text-[11px] leading-relaxed whitespace-pre-wrap text-neutral-500 line-clamp-6">{msg.content}</p>
        </div>
      </div>
    );
  }

  // assistant
  const options = parseAgentOptions(msg.content);
  return (
    <div key={msg.id} className="flex gap-2 group">
      <div className="w-6 h-6 rounded-lg bg-neutral-100 flex items-center justify-center flex-shrink-0 mt-0.5">
        <Bot className="w-3 h-3 text-neutral-500" />
      </div>
      <div className="flex-1 min-w-0 border-l-2 border-neutral-200 pl-3 py-1">
        {msg.agentName && (
          <p className="text-[9px] font-bold uppercase tracking-wide text-neutral-400 mb-0.5">{msg.agentName}</p>
        )}
        <AgentMarkdown content={msg.content} />
        {options.length > 0 && ctx && (
          <OptionPicker
            options={options}
            disabled={!ctx.isActivePicker || ctx.pickerBusy}
            onPick={ctx.onPick}
          />
        )}
        <p className="text-[9px] mt-1 text-neutral-300">{formatTime(msg.timestamp)}</p>
      </div>
    </div>
  );
}

