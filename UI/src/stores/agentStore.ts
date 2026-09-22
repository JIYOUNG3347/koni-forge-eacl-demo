
import { create } from "zustand";
import { apiFetch } from "../lib/apiFetch";
import { safeRandomUUID } from "../utils/uuid";
import { t, tp } from "../i18n";

export interface ChatMessage {
  id: string;
  threadId?: string;  // optional — kept for ChatView compatibility
  role: "user" | "assistant" | "system" | "tool_result";
  content: string;
  toolUse?: { name: string; input: string; output?: string }[];
  thinking?: string;
  agentName?: string;
  timestamp: number;
}

export interface PendingChain {
  nextAgent: string;
  announce: string;
  /** Structured recommended params — populated for corpus/tuning stages */
  params?: Record<string, unknown> | null;
}

export const EXECUTION_LABELS: Record<string, string> = {
  start_training_job: "Starting training",
};

export interface PendingExecution {
  tool: string;
  agent: string;
  stage: string;
  params: Record<string, unknown>;
}


export interface PipelineSessionStage {
  stage: string;
  skipped?: boolean;
  completed_at?: string;
  skipped_at?: string;
  [key: string]: unknown;
}

export interface PipelineStatus {
  currentStage: string | null;
  completedStages: string[];
  pendingChain: PendingChain | null;
  pendingExecution: PendingExecution | null;
  chainPaused: boolean;
  isBusy: boolean;
  busyAgent: string | null;
  session: {
    sessionId: string | null;
    mode: string | null;
    stages: PipelineSessionStage[];
  };
}

const EMPTY_PIPELINE_STATUS: PipelineStatus = {
  currentStage: null,
  completedStages: [],
  pendingChain: null,
  pendingExecution: null,
  chainPaused: false,
  isBusy: false,
  busyAgent: null,
  session: { sessionId: null, mode: null, stages: [] },
};

interface AgentState {
  messages: ChatMessage[];
  isStreaming: boolean;
  abortController: AbortController | null;
  pipelineStatus: PipelineStatus;
  navigateTarget: string | null;

  fetchAgentMessages: () => Promise<void>;
  clearAgentMessages: () => Promise<void>;
  addMessage: (msg: ChatMessage) => void;
  resetSession: () => Promise<void>;
  dispatchMessage: (content: string, agent?: string) => Promise<void>;
  advanceChain: (action: "approve" | "skip", overrides?: Record<string, unknown>) => Promise<void>;
  confirmExecution: (
    action: "approve" | "cancel",
    overrides?: Record<string, unknown>,
  ) => Promise<void>;
  pauseChain: () => Promise<void>;
  resumeChain: () => Promise<void>;
  fetchPipelineStatus: () => Promise<void>;
  abort: () => void;
  clearNavigateTarget: () => void;
}

/* ── Pipeline status parsing helpers ───────────────────────── */

interface RawPipelineStatus {
  current_stage?: string | null;
  completed_stages?: string[];
  pending_chain?: string | null;
  pending_chain_agent?: string | null;
  pending_chain_params?: Record<string, unknown> | null;
  pending_execution?: PendingExecution | null;
  chain_paused?: boolean;
  is_busy?: boolean;
  busy_agent?: string | null;
  session?: { session_id?: string | null; mode?: string | null; stages?: PipelineSessionStage[] };
}

/* Pull `[chain_target:<agent>]` out of a dispatcher system message. */
function extractChainTarget(content: string): string | null {
  const m = /\[chain_target:([a-z_]+)\]/.exec(content);
  return m ? m[1] : null;
}

function extractNavigateTarget(content: string): string | null {
  const m = /\[navigate:([a-z_]+)\]/.exec(content);
  return m ? m[1] : null;
}

function normalizePipelineStatus(raw: RawPipelineStatus, fallbackAnnounce?: string): PipelineStatus {
  const pending = raw.pending_chain;
  let pendingChain: PendingChain | null = null;
  if (pending && typeof pending === "string") {
    pendingChain = {
      nextAgent: raw.pending_chain_agent ?? "",
      announce: pending,
      params: raw.pending_chain_params ?? null,
    };
  }
  return {
    currentStage: raw.current_stage ?? null,
    completedStages: raw.completed_stages ?? [],
    pendingChain: pendingChain
      ? { ...pendingChain, announce: pendingChain.announce || fallbackAnnounce || "" }
      : null,
    pendingExecution: raw.pending_execution ?? null,
    chainPaused: !!raw.chain_paused,
    isBusy: !!raw.is_busy,
    busyAgent: raw.busy_agent ?? null,
    session: {
      sessionId: raw.session?.session_id ?? null,
      mode: raw.session?.mode ?? null,
      stages: raw.session?.stages ?? [],
    },
  };
}

/* Save a message to the server (fire-and-forget). */
function saveAgentMessage(role: string, content: string, agentName?: string) {
  apiFetch("/api/agent/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content, agent_name: agentName ?? null }),
  }).catch(() => {});
}

/* Consume the SSE stream from /api/agent/dispatch or /api/agent/chain/advance.
   Updates messages, pipelineStatus, navigateTarget as events arrive.
   Returns IDs of messages added during the stream. */
async function consumeDispatchStream(
  res: Response,
  set: (partial: Partial<AgentState> | ((s: AgentState) => Partial<AgentState>)) => void,
  get: () => AgentState,
): Promise<Set<string>> {
  const addedIds = new Set<string>();
  const reader = res.body?.getReader();
  if (!reader) return addedIds;
  const decoder = new TextDecoder();
  let buffer = "";
  let latestChainTarget: string | null = null;
  let streamingAssistant: { id: string; content: string; agent?: string } | null = null;

  const pushOrUpdateMsg = (msg: ChatMessage) => {
    addedIds.add(msg.id);
    const current = get().messages;
    const idx = current.findIndex((m) => m.id === msg.id);
    const next = idx >= 0 ? current.map((m, i) => (i === idx ? msg : m)) : [...current, msg];
    set({ messages: next });
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const raw = dataLine.slice(5).trim();
      if (!raw) continue;

      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(raw);
      } catch {
        continue;
      }

      const type = evt.type as string | undefined;

      if (type === "event") {
        const role = (evt.role as ChatMessage["role"]) ?? "assistant";
        const content = (evt.content as string) ?? "";
        const agentName = (evt.agent_name as string) ?? undefined;

        // Track `[chain_target:X]` markers so we can fill nextAgent when the
        // final /status snapshot arrives (which only carries the announce text).
        const ct = extractChainTarget(content);
        if (ct) {
          latestChainTarget = ct;
          void get().fetchPipelineStatus();
        }
        const nav = extractNavigateTarget(content);
        if (nav) set({ navigateTarget: nav });

        if (role === "assistant") {
          // Coalesce consecutive assistant chunks from the same logical response
          if (!streamingAssistant || streamingAssistant.agent !== agentName) {
            streamingAssistant = {
              id: (evt.id as string) ?? safeRandomUUID(),
              content,
              agent: agentName,
            };
          } else {
            streamingAssistant = {
              ...streamingAssistant,
              content: streamingAssistant.content + content,
            };
          }
          pushOrUpdateMsg({
            id: streamingAssistant.id,
            role: "assistant",
            content: streamingAssistant.content,
            agentName,
            timestamp: Date.now(),
          });
        } else if (role === "system") {
          // Hide pure marker lines ([chain_target:...] etc) from the chat log.
          const stripped = content
            .replace(/\[chain_target:[a-z_]+\]/g, "")
            .replace(/\[navigate:[a-z_]+\]/g, "")
            .replace(/\[stage_complete:[a-z_]+\]/g, "")
            .trim();
          if (stripped) {
            pushOrUpdateMsg({
              id: (evt.id as string) ?? safeRandomUUID(),
              role: "system",
              content: stripped,
              agentName,
              timestamp: Date.now(),
            });
          }
          // Reset assistant coalescing — system messages separate responses
          streamingAssistant = null;
        } else if (role === "tool_result") {
          streamingAssistant = null;
        }
      } else if (type === "status") {
        const status = normalizePipelineStatus(evt as RawPipelineStatus);
        if (status.pendingChain && latestChainTarget) {
          status.pendingChain.nextAgent = latestChainTarget;
        }
        set({ pipelineStatus: status });
      } else if (type === "error") {
        pushOrUpdateMsg({
          id: safeRandomUUID(),
          role: "system",
          content: tp("Error: {0}", (evt.message as string) ?? "unknown"),
          timestamp: Date.now(),
        });
      } else if (type === "done") {
        return addedIds;
      }
    }
  }
  return addedIds;
}

export const useAgentStore = create<AgentState>((set, get) => ({
  messages: [],
  isStreaming: false,
  abortController: null,
  pipelineStatus: EMPTY_PIPELINE_STATUS,
  navigateTarget: null,

  fetchAgentMessages: async () => {
    try {
      const res = await apiFetch("/api/agent/messages");
      if (!res.ok) return;
      const data = await res.json() as Array<{
        id: string; role: string; content: string;
        agent_name?: string | null; thinking?: string | null; created_at: string;
      }>;
      const mapped: ChatMessage[] = data.map((m) => ({
        id: m.id,
        role: m.role as ChatMessage["role"],
        content: m.content,
        agentName: m.agent_name ?? undefined,
        thinking: m.thinking ?? undefined,
        timestamp: new Date(m.created_at).getTime(),
      }));
      set({ messages: mapped });
    } catch {
      // Silent — store keeps current state
    }
  },

  clearAgentMessages: async () => {
    // Optimistic update
    set({ messages: [] });
    try {
      const res = await apiFetch("/api/agent/messages", { method: "DELETE" });
      if (!res.ok) {
        await get().fetchAgentMessages();
      }
    } catch {
      await get().fetchAgentMessages();
    }
  },

  resetSession: async () => {
    // 1) Abort any in-flight stream
    const ctrl = get().abortController;
    if (ctrl) {
      try { ctrl.abort(); } catch { /* ignore */ }
    }

    // 2) Clear frontend state
    set({
      messages: [],
      pipelineStatus: EMPTY_PIPELINE_STATUS,
      navigateTarget: null,
      isStreaming: false,
      abortController: null,
    });

    // 3) Delete server-side agent messages (Eraser)
    try {
      await apiFetch("/api/agent/messages", { method: "DELETE" });
    } catch {
      // Best-effort
    }

    // 4) Wipe backend dispatcher (pending_chain, completed_stages, session)
    try {
      await apiFetch("/api/agent/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    } catch {
      // Best-effort
    }

    // 5) Refetch fresh pipeline status (new empty session)
    try { await get().fetchPipelineStatus(); } catch { /* ignore */ }
  },

  /* ── Guided mode: dispatcher SSE stream ────────────────────── */
  dispatchMessage: async (content, agent) => {
    const userMsg: ChatMessage = {
      id: safeRandomUUID(),
      role: "user",
      content,
      timestamp: Date.now(),
    };
    set({ messages: [...get().messages, userMsg] });

    // Save user message to server
    saveAgentMessage("user", content);

    const controller = new AbortController();
    set({ isStreaming: true, abortController: controller });

    try {
      const res = await apiFetch("/api/agent/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, agent }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(tp("Request failed ({0})", res.status));
      // Backend (_dispatcher_sse_saving_agent_msgs) saves assistant messages — no frontend save needed
      await consumeDispatchStream(res, set, get);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const errorMsg: ChatMessage = {
          id: safeRandomUUID(),
          role: "system",
          content: tp("Error: {0}", (err as Error).message),
          timestamp: Date.now(),
        };
        set({ messages: [...get().messages, errorMsg] });
      }
    } finally {
      set({ isStreaming: false, abortController: null });
    }
  },

  advanceChain: async (action, overrides) => {
    // Show a user-visible acknowledgement so the chat log makes sense
    const hasOverrides = overrides && Object.keys(overrides).length > 0;
    const label = action === "approve"
      ? (hasOverrides ? t("Proceeding with the edited parameters") : t("Proceeding"))
      : t("Skipping");
    const ackMsg: ChatMessage = {
      id: safeRandomUUID(),
      role: "user",
      content: label,
      timestamp: Date.now(),
    };
    set({ messages: [...get().messages, ackMsg] });
    saveAgentMessage("user", label);

    // Optimistically clear the pending chain so the chain card hides right away.
    set({
      pipelineStatus: { ...get().pipelineStatus, pendingChain: null },
    });

    const controller = new AbortController();
    set({ isStreaming: true, abortController: controller });

    const postAdvance = () =>
      apiFetch("/api/agent/chain/advance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, overrides: overrides ?? null }),
        signal: controller.signal,
      });

    try {
      let res = await postAdvance();
      // 409 = backend has no pending chain. This happens on a server restart
      // (in-memory _pending_chain lost) or a brief desync with the 8s status
      // poll. Resync status, then retry once before surfacing an error.
      if (res.status === 409) {
        console.warn("[advanceChain] 409 — resyncing pipeline status and retrying once");
        await get().fetchPipelineStatus();
        await new Promise((r) => setTimeout(r, 500));
        res = await postAdvance();
      }
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error((detail as { detail?: string }).detail ?? tp("Request failed ({0})", res.status));
      }
      // Backend (_dispatcher_sse_saving_agent_msgs) saves assistant messages — no frontend save needed
      await consumeDispatchStream(res, set, get);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const errorMsg: ChatMessage = {
          id: safeRandomUUID(),
          role: "system",
          content: tp("Error: {0}", (err as Error).message),
          timestamp: Date.now(),
        };
        set({ messages: [...get().messages, errorMsg] });
      }
    } finally {
      set({ isStreaming: false, abortController: null });
    }
  },

  confirmExecution: async (action, overrides) => {
    const pending = get().pipelineStatus.pendingExecution;
    // Hide the modal optimistically; the next status poll resyncs.
    set({ pipelineStatus: { ...get().pipelineStatus, pendingExecution: null } });

    if (action === "cancel") {
      try {
        await apiFetch("/api/agent/execution/confirm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "cancel" }),
        });
      } catch {
        /* A failed cancel is ignored; the next status poll resyncs. */
      }
      return;
    }

    const label = EXECUTION_LABELS[pending?.tool ?? ""] ?? "Proceeding";
    const ackMsg: ChatMessage = {
      id: safeRandomUUID(),
      role: "user",
      content: t(label),
      timestamp: Date.now(),
    };
    set({ messages: [...get().messages, ackMsg] });
    saveAgentMessage("user", t(label));

    const controller = new AbortController();
    set({ isStreaming: true, abortController: controller });

    const postConfirm = () =>
      apiFetch("/api/agent/execution/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve", overrides: overrides ?? {} }),
        signal: controller.signal,
      });

    try {
      let res = await postConfirm();
      // 409 means the backend has no pending execution (a restart or poll desync). Resync and retry once.
      if (res.status === 409) {
        await get().fetchPipelineStatus();
        await new Promise((r) => setTimeout(r, 500));
        res = await postConfirm();
      }
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error((detail as { detail?: string }).detail ?? tp("Request failed ({0})", res.status));
      }
      await consumeDispatchStream(res, set, get);
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        set({
          messages: [
            ...get().messages,
            {
              id: safeRandomUUID(),
              role: "system",
              content: tp("Error: {0}", (err as Error).message),
              timestamp: Date.now(),
            },
          ],
        });
      }
    } finally {
      set({ isStreaming: false, abortController: null });
    }
  },

  pauseChain: async () => {
    await apiFetch("/api/agent/chain/pause", { method: "POST" });
    set({ pipelineStatus: { ...get().pipelineStatus, chainPaused: true } });
  },

  resumeChain: async () => {
    await apiFetch("/api/agent/chain/resume", { method: "POST" });
    set({ pipelineStatus: { ...get().pipelineStatus, chainPaused: false } });
  },

  fetchPipelineStatus: async () => {
    try {
      const res = await apiFetch("/api/agent/pipeline");
      if (!res.ok) return;
      const raw = (await res.json()) as RawPipelineStatus;
      const status = normalizePipelineStatus(raw);
      set({ pipelineStatus: status });
    } catch {
      // Silent — polling will retry
    }
  },

  clearNavigateTarget: () => set({ navigateTarget: null }),

  addMessage: (msg) => {
    const messages = [...get().messages, msg];
    set({ messages });
  },

  abort: () => {
    const controller = get().abortController;
    if (controller) {
      controller.abort();
      set({ isStreaming: false, abortController: null });
    }
  },
}));
