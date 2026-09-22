/**
 * ChatView (Auto mode) -- Full-featured chat interface.
 *
 * Layout:
 *   Top:    horizontal thread tabs
 *   Middle: settings panel (left, w-72) | chat area (right, flex-1)
 *   AgentSidebar rendered separately by Shell (right edge)
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Send,
  StopCircle,
  Database,
  Bot,
  Thermometer,
  Plus,
  X,
  Loader2,
  Settings2,
  MessageSquare,
  ChevronDown,
  Pencil,
  Trash2,
  Download,
  } from "lucide-react";
import { apiFetch } from "../../lib/apiFetch";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Textarea } from "../../components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
import { Slider } from "../../components/ui/slider";
import { Label } from "../../components/ui/label";
import { safeRandomUUID } from "../../utils/uuid";
import { getUserItem, setUserItem } from "../../utils/storage";
import { tp, useT } from "../../i18n";
import { formatDateTime } from "../../i18n/locale";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */


interface ChatMessage {
  id?: string;              // server-assigned id (present after server fetch)
  role: string;
  content: string;
  /** Milliseconds from request to end of stream, measured client side. */
  elapsedMs?: number;
}

interface Thread {
  id: string;
  title: string;
}

interface HfModel {
  name: string;    // value passed to model load API (path for checkpoints)
  label?: string;  // display label (may differ from name for checkpoint rounds)
  kind?: "base" | "finetuned" | "checkpoint";
}

interface RagCollection {
  name: string;
}

/* ------------------------------------------------------------------ */
/*  localStorage persistence (settings only)                          */
/* ------------------------------------------------------------------ */

const CHAT_STORAGE_KEY = "koni_chat_v1";


interface ChatSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  useRag: boolean;
  ragCollection: string;
}

function loadStoredSettings(): Partial<ChatSettings> {
  try {
    const raw = getUserItem(CHAT_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Partial<ChatSettings>) : {};
  } catch {
    return {};
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function ChatView() {
  const t = useT();
  /* ---- state ---- */
  const initialStored = useMemo(() => loadStoredSettings(), []);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [model, setModel] = useState(initialStored.model ?? "");
  const [temperature, setTemperature] = useState(initialStored.temperature ?? 0.7);
  const [maxTokens, setMaxTokens] = useState(initialStored.maxTokens ?? 512);
  const [systemPrompt, setSystemPrompt] = useState(initialStored.systemPrompt ?? "");
  const [useRag, setUseRag] = useState(initialStored.useRag ?? false);
  const [ragCollection, setRagCollection] = useState(initialStored.ragCollection ?? "");
  const [collections, setCollections] = useState<RagCollection[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThread, setActiveThread] = useState("");
  const [modelStatus, setModelStatus] = useState<
    "loaded" | "loading" | "unloaded"
  >("unloaded");
  const [availableModels, setAvailableModels] = useState<HfModel[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [threadMenuOpen, setThreadMenuOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  // Per-thread message cache (seeded from server on thread switch)
  const threadMessagesRef = useRef<Record<string, ChatMessage[]>>({});

  const chatEndRef = useRef<HTMLDivElement>(null);

  /* ---- thread helpers ---- */
  const syncMessages = useCallback(() => {
    setMessages([...(threadMessagesRef.current[activeThread] ?? [])]);
  }, [activeThread]);

  const setThreadMessages = useCallback(
    (updater: (prev: ChatMessage[]) => ChatMessage[]) => {
      const prev = threadMessagesRef.current[activeThread] ?? [];
      const next = updater(prev);
      threadMessagesRef.current[activeThread] = next;
      setMessages([...next]);
    },
    [activeThread],
  );

  /* ---- auto-unload on page leave ---- */
  useEffect(() => {
    return () => {
      apiFetch("/api/hf/unload", { method: "POST" }).catch(() => {});
    };
  }, []);


  /* ---- persist settings to localStorage (threads/messages on server) ---- */
  useEffect(() => {
    try {
      setUserItem(
        CHAT_STORAGE_KEY,
        JSON.stringify({
          model,
          temperature,
          maxTokens,
          systemPrompt,
          useRag,
          ragCollection,
        } as ChatSettings)
      );
    } catch {
      /* ignore */
    }
  }, [model, temperature, maxTokens, systemPrompt, useRag, ragCollection]);

  /* ---- API calls on mount ---- */
  useEffect(() => {
    // Sync actual backend model state on mount (handles page refresh / server restart)
    apiFetch("/api/hf/status").then((r) =>
      r.ok &&
      r.json().then((d: any) => {
        setModelStatus(d.loaded ? "loaded" : "unloaded");
      }),
    );

    // HF base models + fine-tuned outputs from /storage/models
    apiFetch("/api/hf/models").then((r) =>
      r.ok &&
      r.json().then((d: any) => {
        const bases: HfModel[] = (d.base_models || []).map((m: any) => ({ name: m.name, kind: "base" as const }));
        const finetuned: HfModel[] = (d.trained_outputs || []).map((m: any) => ({ name: m.name, kind: "finetuned" as const }));
        setAvailableModels([...bases, ...finetuned]);
      }),
    );

    // Train checkpoints (completed fine-tuned models)
    apiFetch("/api/train/checkpoints").then((r) =>
      r.ok &&
      r.json().then((d: any) => {
        const seen = new Set<string>();
        const checkpoints: HfModel[] = (d.checkpoints || [])
          .map((c: any) => {
            // Use path as the load value so each round resolves to its own weights.
            // label: "job_id · Round N" for rounds, plain "job_id" for final.
            const roundMatch = typeof c.name === "string" && c.name.match(/^round_(\d+)$/);
            const label = roundMatch
              ? `${c.job_id} · Round ${roundMatch[1]}`
              : c.name === "final" || !c.name
                ? c.job_id
                : `${c.job_id} · ${c.name}`;
            const value = c.path ?? c.job_id;
            return { name: value, label, kind: "checkpoint" as const };
          })
          .filter((c: HfModel) => {
            if (seen.has(c.name)) return false;
            seen.add(c.name);
            return true;
          });
        setAvailableModels((prev) => {
          const existingNames = new Set(prev.map((m) => m.name));
          return [...prev, ...checkpoints.filter((c) => !existingNames.has(c.name))];
        });
      }),
    );

    apiFetch("/api/rag/collections").then((r) =>
      r.ok &&
      r.json().then((d: any) => setCollections(d.collections || [])),
    );
  }, []);

  /* ---- load threads from server on mount ---- */
  useEffect(() => {
    apiFetch("/api/chat/threads")
      .then((r) => r.ok && r.json())
      .then((data: any) => {
        if (!Array.isArray(data)) return;
        // Filter to Auto mode threads only
        const autoThreads: Thread[] = data
          .filter((t: any) => t.mode === "auto")
          .map((t: any) => ({ id: t.id, title: t.title }));
        if (autoThreads.length > 0) {
          setThreads(autoThreads);
          setActiveThread(autoThreads[0].id);
        } else {
          // No auto threads yet — create a default one
          const id = safeRandomUUID();
          const title = t("Default thread");
          threadMessagesRef.current[id] = [];
          setThreads([{ id, title }]);
          setActiveThread(id);
          apiFetch("/api/chat/threads", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, title, mode: "auto" }),
          }).catch(() => {});
        }
      })
      .catch(() => {
        // Fallback: start with a local default thread
        const id = safeRandomUUID();
        threadMessagesRef.current[id] = [];
        setThreads([{ id, title: t("Default thread") }]);
        setActiveThread(id);
      });
  }, [t]);

  /* ---- load messages from server when thread changes ---- */
  useEffect(() => {
    if (!activeThread) return;
    // Use cached version first for immediate display
    syncMessages();
    // Then fetch from server (updates cache + display)
    apiFetch(`/api/chat/threads/${activeThread}/messages`)
      .then((r) => r.ok && r.json())
      .then((data: any) => {
        if (!Array.isArray(data)) return;
        const mapped: ChatMessage[] = data.map((m: any) => ({
          id: m.id as string | undefined,
          role: m.role,
          content: m.content,
        }));
        threadMessagesRef.current[activeThread] = mapped;
        syncMessages();
      })
      .catch(() => {});
  }, [activeThread, syncMessages]);

  /* ---- auto-scroll ---- */
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  /* ---- send message ---- */
  const handleSend = useCallback(async () => {
    if (!input.trim() || isStreaming) return;
    if (modelStatus !== "loaded") {
      setThreadMessages((prev) => [
        ...prev,
        { role: "assistant", content: t("⚠️ No model is loaded. Pick a model in the left panel and press Load.") },
      ]);
      return;
    }

    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    setThreadMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);

    // Snapshot history (includes the just-added user message via ref)
    const history = [...(threadMessagesRef.current[activeThread] ?? [])];

    try {
      const res = await apiFetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history,
          temperature,
          max_tokens: maxTokens,
          system_prompt: systemPrompt,
          rag_enabled: useRag,
          rag_collection: ragCollection,
          thread_id: activeThread || null,
        }),
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as Record<string, string>).detail || t("Request failed"));
      }

      // Add empty assistant placeholder, then stream tokens into it
      setThreadMessages((prev) => [...prev, { role: "assistant", content: "" }]);
      const startedAt = Date.now();

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";
      let buf = "";

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") break outer;
          try {
            const data = JSON.parse(payload);
            if (data.token) {
              accumulated += data.token;
              setThreadMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { role: "assistant", content: accumulated };
                return next;
              });
            }
            if (data.error) throw new Error(data.error);
          } catch (parseErr) {
            if (parseErr instanceof SyntaxError) continue;
            throw parseErr;
          }
        }
      }
      // Attach the elapsed time to the finished answer.
      setThreadMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = {
            ...last,
            elapsedMs: Date.now() - startedAt,
          };
        }
        return next;
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : t("Unknown error");
      setThreadMessages((prev) => {
        const last = prev[prev.length - 1];
        const errContent = tp("Error: {0}", msg);
        // Replace empty placeholder if present, otherwise append
        if (last?.role === "assistant" && last.content === "") {
          return [...prev.slice(0, -1), { role: "assistant", content: errContent }];
        }
        return [...prev, { role: "assistant", content: errContent }];
      });
    } finally {
      setIsStreaming(false);
    }
  }, [
    input,
    isStreaming,
    modelStatus,
    activeThread,
    temperature,
    maxTokens,
    systemPrompt,
    useRag,
    ragCollection,
    setThreadMessages,
    t,
  ]);

  /* ---- clear current thread ---- */
  const handleClearThread = useCallback(async () => {
    if (!activeThread) return;

    // Optimistic update
    threadMessagesRef.current[activeThread] = [];
    setMessages([]);

    const resync = () => {
      apiFetch(`/api/chat/threads/${activeThread}/messages`)
        .then((r) => r.ok && r.json())
        .then((data: unknown) => {
          if (!Array.isArray(data)) return;
          const mapped: ChatMessage[] = (data as Array<{ id?: string; role: string; content: string }>).map(
            (m) => ({ id: m.id, role: m.role, content: m.content }),
          );
          threadMessagesRef.current[activeThread] = mapped;
          syncMessages();
        })
        .catch(() => {});
    };

    try {
      const res = await apiFetch(
        `/api/chat/threads/${activeThread}/messages`,
        { method: "DELETE" },
      );
      if (!res.ok) resync();
    } catch {
      resync();
    }
  }, [activeThread, syncMessages]);

  /* ---- delete single message ---- */
  const handleDeleteMessage = useCallback(async (idx: number) => {
    const msgs = threadMessagesRef.current[activeThread] ?? [];
    const msg = msgs[idx];

    // Optimistic update
    const updated = msgs.filter((_, i) => i !== idx);
    threadMessagesRef.current[activeThread] = updated;
    setMessages([...updated]);

    if (msg?.id && activeThread) {
      try {
        await apiFetch(
          `/api/chat/threads/${activeThread}/messages/${msg.id}`,
          { method: "DELETE" },
        );
      } catch {
        // Re-sync from server on failure
        apiFetch(`/api/chat/threads/${activeThread}/messages`)
          .then((r) => r.ok && r.json())
          .then((data: any) => {
            if (!Array.isArray(data)) return;
            const remapped: ChatMessage[] = data.map((m: any) => ({
              id: m.id as string | undefined,
              role: m.role,
              content: m.content,
            }));
            threadMessagesRef.current[activeThread] = remapped;
            setMessages([...remapped]);
          })
          .catch(() => {});
      }
    }
  }, [activeThread]);

  /* ---- download chat as text file ---- */
  const handleDownload = useCallback(() => {
    const msgs = threadMessagesRef.current[activeThread] ?? [];
    if (msgs.length === 0) return;
    const lines: string[] = [
      t("=== KONI Forge chat history ==="),
      tp("Model: {0}", model || "Unspecified"),
      tp("Date: {0}", formatDateTime(Date.now())),
      `RAG: ${useRag ? `ON (${ragCollection || "No collections"})` : "OFF"}`,
      ``,
    ];
    for (const m of msgs) {
      lines.push(m.role === "user" ? t("[user]") : t("[model]"));
      lines.push(m.content);
      lines.push("");
    }
    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeThread, model, useRag, ragCollection, t]);


  /* ---- keyboard ---- */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ---- thread management ---- */
  const createThread = () => {
    const id = safeRandomUUID();
    const title = tp("Thread {0}", threads.length + 1);
    threadMessagesRef.current[id] = [];
    setThreads((prev) => [...prev, { id, title }]);
    setActiveThread(id);

    // Fire-and-forget: create on server with client-generated UUID
    apiFetch("/api/chat/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, title, mode: "auto" }),
    }).catch(() => {});
  };

  const closeThread = (threadId: string) => {
    if (threads.length <= 1) return;
    delete threadMessagesRef.current[threadId];
    setThreads((prev) => prev.filter((t) => t.id !== threadId));
    if (activeThread === threadId) {
      const remaining = threads.filter((t) => t.id !== threadId);
      setActiveThread(remaining[0]?.id ?? "");
    }

    // Fire-and-forget: delete on server
    apiFetch(`/api/chat/threads/${threadId}`, { method: "DELETE" }).catch(() => {});
  };

  const startRename = (t: Thread) => {
    setRenamingId(t.id);
    setRenameValue(t.title);
  };

  /** Commit a rename. The PATCH is fire-and-forget; a failure leaves the screen as is. */
  const commitRename = (threadId: string) => {
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title) return;
    const current = threads.find((t) => t.id === threadId);
    if (!current || current.title === title) return;
    setThreads((prev) => prev.map((t) => (t.id === threadId ? { ...t, title } : t)));
    apiFetch(`/api/chat/threads/${threadId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }).catch(() => {});
  };

  /* ---- model groups ---- */
  const baseModels      = availableModels.filter((m) => m.kind === "base");
  const finetunedModels = availableModels.filter((m) => m.kind === "finetuned");
  const checkpoints     = availableModels.filter((m) => m.kind === "checkpoint");

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* ============================================================ */}
      {/*  TOP: context bar — thread picker, setting chips, actions        */}
      {/* ============================================================ */}
      <div className="flex items-center gap-2 px-3 py-2 border-b flex-shrink-0">
        {/* Thread picker */}
        <div className="relative flex-shrink-0 z-30">
          <button
            onClick={() => setThreadMenuOpen((v) => !v)}
            className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-neutral-200 text-[12px] font-bold text-neutral-900 hover:bg-neutral-50 transition-colors max-w-[220px]"
          >
            <MessageSquare className="w-3.5 h-3.5 text-neutral-400 flex-shrink-0" />
            <span className="truncate">{threads.find((t) => t.id === activeThread)?.title ?? t("New thread")}</span>
            <span className="text-[10px] font-semibold text-neutral-400 tabular-nums">{threads.length}</span>
            <ChevronDown className={`w-3.5 h-3.5 text-neutral-400 flex-shrink-0 transition-transform ${threadMenuOpen ? "rotate-180" : ""}`} />
          </button>
          {threadMenuOpen && (
            <div className="absolute z-30 mt-1 w-[260px] rounded-xl border border-neutral-200 bg-white shadow-lg py-1">
              {threads.map((th) => (
                <div
                  key={th.id}
                  className={`group flex items-center gap-2 px-3 py-2 text-[12px] cursor-pointer hover:bg-neutral-50 ${
                    activeThread === th.id ? "font-bold text-neutral-900" : "text-neutral-600"
                  }`}
                  onClick={() => {
                    if (renamingId === th.id) return;
                    setActiveThread(th.id);
                    setThreadMenuOpen(false);
                  }}
                >
                  {renamingId === th.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onBlur={() => commitRename(th.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") { e.preventDefault(); commitRename(th.id); }
                        if (e.key === "Escape") { e.preventDefault(); setRenamingId(null); }
                      }}
                      className="flex-1 min-w-0 px-1.5 py-0.5 rounded border border-neutral-300 text-[12px] font-normal text-neutral-900 outline-none focus:border-neutral-700"
                    />
                  ) : (
                    <>
                      <span className="flex-1 truncate" onDoubleClick={(e) => { e.stopPropagation(); startRename(th); }}>
                        {th.title}
                      </span>
                      <span
                        role="button"
                        title={t("Rename")}
                        onClick={(e) => { e.stopPropagation(); startRename(th); }}
                        className="p-0.5 rounded opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-neutral-900 transition-opacity"
                      >
                        <Pencil className="w-3 h-3" />
                      </span>
                      {threads.length > 1 && (
                        <span
                          role="button"
                          title={t("Close thread")}
                          onClick={(e) => { e.stopPropagation(); closeThread(th.id); }}
                          className="p-0.5 rounded opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-red-500 transition-opacity"
                        >
                          <X className="w-3 h-3" />
                        </span>
                      )}
                    </>
                  )}
                </div>
              ))}
              <div className="h-px bg-neutral-100 my-1" />
              <button
                onClick={() => { createThread(); setThreadMenuOpen(false); }}
                className="flex items-center gap-1.5 w-full px-3 py-2 text-[12px] text-neutral-500 hover:bg-neutral-50 hover:text-neutral-900 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />{t("New thread")}
              </button>
            </div>
          )}
        </div>

        {/* Setting chips — clicking one opens the settings panel.
            Only the chips scroll horizontally; overflow on the bar would clip the dropdown. */}
        <div className="flex items-center gap-2 min-w-0 overflow-x-auto">
        <button onClick={() => setSettingsOpen(true)}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-neutral-50 border border-neutral-100 text-[11px] font-semibold text-neutral-600 hover:bg-neutral-100 transition-colors flex-shrink-0">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
            modelStatus === "loaded" ? "bg-green-500"
            : modelStatus === "loading" ? "bg-yellow-400 animate-pulse"
            : "bg-neutral-300"}`} />
          <span className="max-w-[180px] truncate">{model || t("No model selected")}</span>
        </button>
        {useRag && (
          <button onClick={() => setSettingsOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-neutral-900 text-white text-[11px] font-semibold flex-shrink-0">
            <Database className="w-3 h-3 opacity-70" />
            RAG{ragCollection ? ` · ${ragCollection}` : ""}
          </button>
        )}
        <span className="px-2.5 py-1.5 rounded-lg bg-neutral-50 border border-neutral-100 text-[11px] font-semibold text-neutral-500 flex-shrink-0">
          T {temperature}
        </span>
        </div>

        <div className="flex-1" />

        <button onClick={() => setSettingsOpen((v) => !v)} title={t("Settings")}
          className={`p-1.5 rounded-lg transition-colors flex-shrink-0 ${settingsOpen ? "bg-neutral-900 text-white" : "text-neutral-400 hover:bg-neutral-50 hover:text-neutral-700"}`}>
          <Settings2 className="w-4 h-4" />
        </button>
        {messages.length > 0 && !isStreaming && (
          <>
            <button onClick={handleDownload} title={t("Download conversation")}
              className="p-1.5 rounded-lg text-neutral-400 hover:bg-neutral-50 hover:text-neutral-700 transition-colors flex-shrink-0">
              <Download className="w-4 h-4" />
            </button>
            <button onClick={handleClearThread} title={t("Delete conversation")}
              className="p-1.5 rounded-lg text-neutral-400 hover:bg-red-50 hover:text-red-500 transition-colors flex-shrink-0">
              <Trash2 className="w-4 h-4" />
            </button>
          </>
        )}
      </div>

      {/* ============================================================ */}
      {/*  MIDDLE: Settings (left) + Chat (right)                      */}
      {/* ============================================================ */}
      <div className="flex flex-1 min-h-0">

        {/* ---------------------------------------------------------- */}
        {/*  RIGHT (slide-over): settings, opened only when needed        */}
        {/*  The conversation is the focus, so this takes no permanent width. */}
        {/* ---------------------------------------------------------- */}
        {settingsOpen && (
        <div className="order-2 w-[300px] border-l flex-shrink-0 overflow-y-auto bg-white shadow-[-8px_0_24px_rgba(0,0,0,0.06)]">
          <div className="flex items-center justify-between px-4 py-3 border-b sticky top-0 bg-white z-10">
            <span className="text-[12.5px] font-extrabold tracking-[-0.02em]">{t("Settings")}</span>
            <button onClick={() => setSettingsOpen(false)} title={t("Close")}
              className="p-1 rounded-md text-neutral-400 hover:bg-neutral-50 hover:text-neutral-700 transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
          {/* Model settings */}
          <div className="px-4 pt-4 pb-2">
            <h3 className="text-xs font-semibold flex items-center gap-1.5 text-foreground">
              <Settings2 className="w-3.5 h-3.5" />
              {t("Model settings")}
            </h3>
          </div>

          <div className="px-4 pb-4 space-y-4">

            {/* 1. Model selection */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-[11px] font-medium">{t("Model")}</Label>
                <span className="flex items-center gap-1 text-[10px]">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    modelStatus === "loaded"
                      ? "bg-green-500"
                      : modelStatus === "loading"
                        ? "bg-yellow-400 animate-pulse"
                        : "bg-neutral-300"
                  }`} />
                  <span className={`font-medium ${
                    modelStatus === "loaded"
                      ? "text-green-600"
                      : modelStatus === "loading"
                        ? "text-yellow-600"
                        : "text-muted-foreground"
                  }`}>
                    {modelStatus === "loaded" ? t("Loaded") : modelStatus === "loading" ? t("Loading...") : t("Queued")}
                  </span>
                </span>
              </div>
              <Select value={model} onValueChange={(v) => { setModel(v); setModelStatus("unloaded"); }}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder={t("Select a model")} />
                </SelectTrigger>
                <SelectContent>
                  {baseModels.length > 0 && (
                    <>
                      <div className="px-2 py-1 text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{t("Base model")}</div>
                      {baseModels.map((m) => (
                        <SelectItem key={m.name} value={m.name} className="text-xs">{m.name}</SelectItem>
                      ))}
                    </>
                  )}
                  {finetunedModels.length > 0 && (
                    <>
                      <div className="px-2 py-1 text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{t("Fine-tuned model")}</div>
                      {finetunedModels.map((m) => (
                        <SelectItem key={m.name} value={m.name} className="text-xs">{m.name}</SelectItem>
                      ))}
                    </>
                  )}
                  {checkpoints.length > 0 && (
                    <>
                      <div className="px-2 py-1 text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{t("Training checkpoints")}</div>
                      {checkpoints.map((m) => (
                        <SelectItem key={m.name} value={m.name} className="text-xs">{m.label ?? m.name}</SelectItem>
                      ))}
                    </>
                  )}
                  {availableModels.length === 0 && (
                    <div className="px-2 py-1.5 text-xs text-muted-foreground">
                      {t("No model (check /storage/models)")}
                    </div>
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* 2. RAG toggle and its options */}
            <div className="space-y-2">
              <button
                type="button"
                onClick={() => setUseRag((v) => !v)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-md border text-[11px] font-medium transition-colors ${
                  useRag
                    ? "bg-neutral-100 border-neutral-400 text-neutral-800"
                    : "bg-muted/40 border-transparent text-muted-foreground hover:bg-muted/70"
                }`}
              >
                <span className="flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5" />
                  {t("RAG retrieval")}
                </span>
                <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${useRag ? "bg-neutral-800 text-white" : "bg-muted text-muted-foreground"}`}>
                  {useRag ? "ON" : "OFF"}
                </span>
              </button>

              {useRag && (
                <div className="space-y-2 pl-1 border-l-2 border-neutral-200 ml-1">
                  {/* Collection selection */}
                  <div className="space-y-1 pl-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-[11px] font-medium">{t("Collection")}</Label>
                      <button
                        type="button"
                        className="text-[10px] text-muted-foreground hover:text-foreground"
                        onClick={() => {
                          apiFetch("/api/rag/collections").then(r => r.ok && r.json().then((d: any) => setCollections(d.collections || [])));
                        }}
                      >
                        {t("Refresh")}
                      </button>
                    </div>
                    <Select value={ragCollection} onValueChange={setRagCollection}>
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue placeholder={t("Select a collection")} />
                      </SelectTrigger>
                      <SelectContent>
                        {collections.map((c) => (
                          <SelectItem key={c.name} value={c.name} className="text-xs">{c.name}</SelectItem>
                        ))}
                        {collections.length === 0 && (
                          <div className="px-2 py-1.5 text-xs text-muted-foreground">{t("No indexed collections")}</div>
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {ragCollection && (
                    <p className="text-[10px] text-neutral-600 pl-2">✓ {ragCollection}</p>
                  )}
                  {!ragCollection && (
                    <p className="text-[10px] text-neutral-400 pl-2">{t("Select a collection")}</p>
                  )}
                </div>
              )}
            </div>

            {/* 3. Load / unload */}
            <div className="flex items-center gap-1.5">
              <Button
                size="sm"
                variant={modelStatus === "loaded" ? "outline" : "default"}
                className={`flex-1 h-7 text-[11px] gap-1 ${modelStatus === "loaded" ? "border-green-500 text-green-700 hover:bg-green-50" : ""}`}
                disabled={!model || modelStatus === "loading"}
                onClick={async () => {
                  setModelStatus("loading");
                  const res = await apiFetch("/api/hf/load", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_name: model }),
                  }).catch(() => null);
                  setModelStatus(res?.ok ? "loaded" : "unloaded");
                }}
              >
                {modelStatus === "loading" ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
                {modelStatus === "loaded" ? t("● loaded") : t("Load")}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-7 text-[11px] gap-1"
                disabled={!model || modelStatus === "loading" || modelStatus !== "loaded"}
                onClick={async () => {
                  setModelStatus("loading");
                  await apiFetch("/api/hf/unload", { method: "POST" }).catch(() => {});
                  setModelStatus("unloaded");
                }}
              >
                {t("Clear")}
              </Button>
            </div>

            {/* ④ Temperature */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-[11px] font-medium flex items-center gap-1">
                  <Thermometer className="w-3 h-3" />
                  Temperature
                </Label>
                <span className="text-[11px] font-mono font-medium">{temperature.toFixed(1)}</span>
              </div>
              <Slider value={[temperature]} onValueChange={([v]) => setTemperature(v)} min={0} max={2} step={0.1} className="w-full" />
            </div>

            {/* ⑤ Max Tokens */}
            <div className="space-y-1.5">
              <Label className="text-[11px] font-medium">Max Tokens</Label>
              <Input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Math.max(1, parseInt(e.target.value) || 1))}
                min={1}
                max={32768}
                className="h-8 text-xs"
              />
            </div>

            {/* 6. System prompt */}
            <div className="space-y-1.5">
              <Label className="text-[11px] font-medium">{t("System prompt")}</Label>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder={t("Enter a system prompt...")}
                rows={3}
                className="resize-none text-xs"
              />
            </div>
          </div>
        </div>
        )}

        {/* ---------------------------------------------------------- */}
        {/*  MAIN: conversation and composer                             */}
        {/* ---------------------------------------------------------- */}
        <div className="order-1 flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Messages — answers are full-width text, only questions are bubbles */}
          <div className="flex-1 overflow-y-auto min-h-0">
            {messages.length === 0 && !isStreaming ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[300px] gap-3.5 px-6">
                <h2 className="text-[20px] font-black tracking-[-0.03em] text-neutral-900">{t("What would you like to look at?")}</h2>
                <p className="text-[12.5px] text-neutral-400">{t("A screen for asking the trained model questions and checking its answers.")}</p>
                {modelStatus !== "loaded" && (
                  <p className="text-[11px] text-neutral-300">{t("Select and load a model in settings first")}</p>
                )}
              </div>
            ) : (
              <div className="w-full px-6 py-6 flex flex-col gap-6">
                {messages.map((msg, idx) => {
                  const isUser = msg.role === "user";
                  const isLast = idx === messages.length - 1;

                  if (isUser) {
                    return (
                      <div key={idx} className="group flex justify-end gap-2">
                        {!isStreaming && (
                          <button onClick={() => handleDeleteMessage(idx)} title={t("Delete message")}
                            className="self-start mt-1 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity text-neutral-300 hover:text-red-500">
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                        <div className="max-w-[78%] bg-neutral-100 text-neutral-800 rounded-2xl rounded-br-md px-3.5 py-2.5 text-[13.5px] leading-relaxed whitespace-pre-wrap">
                          {msg.content}
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div key={idx} className="group flex gap-3">
                      <div className="flex-shrink-0 w-[26px] h-[26px] rounded-full bg-neutral-900 text-white flex items-center justify-center">
                        <Bot className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[13.5px] leading-[1.75] text-neutral-800 whitespace-pre-wrap">
                          {msg.content}
                          {isStreaming && isLast && (
                            <span className="inline-flex items-center ml-1.5 align-middle">
                              <span className="inline-block w-[7px] h-[15px] bg-neutral-900 align-[-2px] animate-pulse" />
                            </span>
                          )}
                        </div>

                        {/* Evidence chips — how this answer was produced */}
                        {msg.elapsedMs && (
                          <div className="flex items-center gap-1.5 flex-wrap mt-2.5">
                            {typeof msg.elapsedMs === "number" && (
                              <span className="inline-flex items-center px-2.5 py-1 rounded-full border border-neutral-200 text-[10.5px] font-medium text-neutral-500 tabular-nums">
                                {tp("{0}s", (msg.elapsedMs / 1000).toFixed(1))}
                              </span>
                            )}
                            {!isStreaming && (
                              <button onClick={() => handleDeleteMessage(idx)} title={t("Delete message")}
                                className="px-2.5 py-1 rounded-full border border-neutral-200 text-[10.5px] font-medium text-neutral-400 opacity-0 group-hover:opacity-100 hover:text-red-500 hover:border-red-200 transition-all">
                                {t("Delete")}
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}

                {/* Streaming placeholder */}
                {isStreaming &&
                  messages.length > 0 &&
                  messages[messages.length - 1].role === "user" && (
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-[26px] h-[26px] rounded-full bg-neutral-900 text-white flex items-center justify-center">
                        <Bot className="w-3.5 h-3.5" />
                      </div>
                      <div className="inline-flex items-center gap-2 text-neutral-400 text-[13px] pt-0.5">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>{t("Generating response...")}</span>
                      </div>
                    </div>
                  )}

                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {/* Composer — centred card. RAG and model can be switched here */}
          <div className="flex-shrink-0 px-6 pb-4 pt-3">
            <div className="max-w-[680px] mx-auto">
              <div className="border border-neutral-200 rounded-2xl shadow-[0_2px_10px_rgba(0,0,0,0.05)] bg-white px-3.5 py-3">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={t("Type a message...")}
                  rows={1}
                  className="min-h-[24px] max-h-[160px] resize-none border-0 shadow-none px-0 py-0 text-[13.5px] focus-visible:ring-0"
                />
                <div className="flex items-center gap-1.5 mt-2.5">
                  <button
                    onClick={() => setUseRag((v) => !v)}
                    title={t("Use RAG retrieval")}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold transition-colors ${
                      useRag ? "bg-neutral-900 text-white" : "bg-neutral-50 border border-neutral-100 text-neutral-500 hover:bg-neutral-100"
                    }`}
                  >
                    <Database className={`w-3 h-3 ${useRag ? "opacity-70" : "text-neutral-400"}`} />
                    RAG
                  </button>
                  <button onClick={() => setSettingsOpen(true)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-neutral-50 border border-neutral-100 text-[11px] font-semibold text-neutral-500 hover:bg-neutral-100 transition-colors max-w-[220px]">
                    <span className="truncate">{model || t("Select a model")}</span>
                  </button>
                  {isStreaming ? (
                    <button onClick={() => setIsStreaming(false)} title={t("Stop generation")}
                      className="ml-auto inline-flex items-center justify-center w-[30px] h-[30px] rounded-[9px] bg-red-500 text-white hover:bg-red-600 transition-colors">
                      <StopCircle className="w-3.5 h-3.5" />
                    </button>
                  ) : (
                    <button onClick={handleSend} disabled={!input.trim() || modelStatus !== "loaded"} title={t("Send")}
                      className="ml-auto inline-flex items-center justify-center w-[30px] h-[30px] rounded-[9px] bg-neutral-900 text-white hover:bg-neutral-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
                      <Send className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
              <p className="text-center text-[10px] text-neutral-300 mt-2">
                {t("Enter to send · Shift+Enter for a new line")}
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
