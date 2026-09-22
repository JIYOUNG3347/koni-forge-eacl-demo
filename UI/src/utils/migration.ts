
import { apiFetch } from "../lib/apiFetch";
import { getUserItem, setUserItem } from "./storage";

const MIGRATION_DONE_KEY = "kf-migration-v1-done";

const LEGACY_KEYS = {
  GUIDED_THREADS:  "kf-agent-threads",
  GUIDED_MESSAGES: "kf-agent-messages",
  AUTO_CHAT:       "koni_chat_v1",
};

interface LegacyThread {
  id: string;
  title: string;
  created_at?: number;
  updated_at?: number;
}

interface LegacyMessage {
  id?: string;
  threadId?: string;
  role: string;
  content: string;
  timestamp?: number;
}

interface LegacyAutoStorage {
  threads?: Array<{ id: string; title: string }>;
  threadMessages?: Record<string, Array<{ role: string; content: string }>>;
}

async function migrateThread(thread: LegacyThread, mode: "guided" | "auto"): Promise<boolean> {
  try {
    const res = await apiFetch("/api/chat/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: thread.id, title: thread.title, mode }),
    });
    return res.ok || res.status === 409; // 409 = already exists, OK
  } catch {
    return false;
  }
}

async function migrateMessage(threadId: string, msg: LegacyMessage): Promise<void> {
  try {
    await apiFetch(`/api/chat/threads/${threadId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: msg.role as "user" | "assistant" | "system" | "tool_result",
        content: msg.content || "",
      }),
    });
  } catch {
    // Best-effort — skip individual message failures
  }
}

/**
 * Run localStorage → server migration once per user per install.
 * Safe to call on every login; no-ops after first successful run.
 */
export async function runMigration(): Promise<void> {
  if (getUserItem(MIGRATION_DONE_KEY) === "true") return;

  // ── Guided mode threads ──
  try {
    const rawThreads = getUserItem(LEGACY_KEYS.GUIDED_THREADS);
    const rawMessages = getUserItem(LEGACY_KEYS.GUIDED_MESSAGES);
    if (rawThreads) {
      const threads: LegacyThread[] = JSON.parse(rawThreads);
      const messages: LegacyMessage[] = rawMessages ? JSON.parse(rawMessages) : [];

      for (const thread of threads) {
        const ok = await migrateThread(thread, "guided");
        if (!ok) continue;
        const threadMsgs = messages.filter((m) => m.threadId === thread.id);
        for (const msg of threadMsgs) {
          await migrateMessage(thread.id, msg);
        }
      }
    }
  } catch {
    // parse error — skip guided migration
  }

  // ── Auto mode threads ──
  try {
    const rawAuto = getUserItem(LEGACY_KEYS.AUTO_CHAT);
    if (rawAuto) {
      const stored: LegacyAutoStorage = JSON.parse(rawAuto);
      const threads = stored.threads ?? [];
      const threadMessages = stored.threadMessages ?? {};

      for (const thread of threads) {
        if (!thread.id || thread.id === "default") continue; // skip placeholder
        const ok = await migrateThread({ id: thread.id, title: thread.title }, "auto");
        if (!ok) continue;
        const msgs = threadMessages[thread.id] ?? [];
        for (const msg of msgs) {
          await migrateMessage(thread.id, msg);
        }
      }
    }
  } catch {
    // parse error — skip auto migration
  }

  // Mark done — don't re-run on next login
  setUserItem(MIGRATION_DONE_KEY, "true");
}
