
import { readFileSync } from "fs";
import { join } from "path";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";

/* ─── path constants ─────────────────────────────────────────────── */

const SRC_DIR = join(__dirname, "..");
const AGENT_STORE_PATH = join(SRC_DIR, "stores", "agentStore.ts");
const AUTO_CHATVIEW_PATH = join(SRC_DIR, "views", "auto", "ChatView.tsx");

/* ══════════════════════════════════════════════════════════════════════
   Suite 1: regression — the localStorage legacy symbols are gone
   ══════════════════════════════════════════════════════════════════════ */

describe("Suite 1: agentStore.ts — legacy localStorage symbols removed", () => {
  let source: string;

  beforeEach(() => {
    source = readFileSync(AGENT_STORE_PATH, "utf-8");
  });

  it("agentStore.ts has no persistThreads symbol", () => {
    expect(source).not.toMatch(/persistThreads/);
  });

  it("agentStore.ts has no persistMessages symbol", () => {
    expect(source).not.toMatch(/persistMessages/);
  });

  it("agentStore.ts has no sendMessage action (the WebSocket one aside)", () => {
    const actionDeclaration = /sendMessage\s*[:=]\s*(async\s+)?\(/.test(source);
    expect(actionDeclaration).toBe(false);
  });

  it("agentStore.ts has fetchAgentMessages", () => {
    expect(source).toMatch(/fetchAgentMessages/);
  });

  it("agentStore.ts has clearAgentMessages", () => {
    expect(source).toMatch(/clearAgentMessages/);
  });

  it("thread symbols are gone", () => {
    // createThread, deleteThread, loadThreadsFromServer, syncMessagesFromServer
    // All removed from agentStore.
    expect(source).not.toMatch(/createThread\s*:/);
    expect(source).not.toMatch(/deleteThread\s*:/);
    expect(source).not.toMatch(/loadThreadsFromServer\s*:/);
    expect(source).not.toMatch(/syncMessagesFromServer\s*:/);
  });
});

/* ══════════════════════════════════════════════════════════════════════
   Suite 2: migration.ts — pure logic
   ══════════════════════════════════════════════════════════════════════ */

vi.mock("../utils/storage", () => ({
  getUserItem: vi.fn(),
  setUserItem: vi.fn(),
}));

vi.mock("../lib/apiFetch", () => ({
  apiFetch: vi.fn(),
}));

describe("Suite 2: migration.ts — pure logic", () => {
  afterEach(() => {
    vi.resetAllMocks();
    vi.resetModules();
  });

  async function getModules() {
    const storageMod = await import("../utils/storage");
    const apiFetchMod = await import("../lib/apiFetch");
    const migrationMod = await import("../utils/migration");
    return {
      getUserItem: storageMod.getUserItem as ReturnType<typeof vi.fn>,
      setUserItem: storageMod.setUserItem as ReturnType<typeof vi.fn>,
      apiFetch: apiFetchMod.apiFetch as ReturnType<typeof vi.fn>,
      runMigration: migrationMod.runMigration,
    };
  }

  function makeOkResponse(status = 200): Response {
    return { ok: true, status } as unknown as Response;
  }

  it("runMigration does nothing once the done flag is set", async () => {
    const { getUserItem, apiFetch, runMigration } = await getModules();
    getUserItem.mockReturnValue("true");

    await runMigration();

    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("without the flag it POSTs /api/chat/threads for each guided thread", async () => {
    const { getUserItem, apiFetch, runMigration } = await getModules();

    const threads = [
      { id: "t1", title: "Thread 1" },
      { id: "t2", title: "Thread 2" },
    ];

    getUserItem.mockImplementation((key: string) => {
      if (key === "kf-migration-v1-done") return null;
      if (key === "kf-agent-threads") return JSON.stringify(threads);
      if (key === "kf-agent-messages") return JSON.stringify([]);
      if (key === "koni_chat_v1") return null;
      return null;
    });

    apiFetch.mockResolvedValue(makeOkResponse());

    await runMigration();

    const postCalls = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.filter(
      (call: unknown[]) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(postCalls.length).toBeGreaterThanOrEqual(2);

    const urls = postCalls.map((call: unknown[]) => call[0] as string);
    expect(urls.every((u) => u === "/api/chat/threads")).toBe(true);

    const bodies = postCalls.map((call: unknown[]) =>
      JSON.parse((call[1] as RequestInit).body as string),
    );
    expect(bodies.some((b: { id: string }) => b.id === "t1")).toBe(true);
    expect(bodies.some((b: { id: string }) => b.id === "t2")).toBe(true);
  });

  it("runMigration sets MIGRATION_DONE_KEY to 'true' when it finishes", async () => {
    const { getUserItem, setUserItem, apiFetch, runMigration } = await getModules();

    getUserItem.mockImplementation((key: string) => {
      if (key === "kf-migration-v1-done") return null;
      return null;
    });
    apiFetch.mockResolvedValue(makeOkResponse());

    await runMigration();

    expect(setUserItem).toHaveBeenCalledWith("kf-migration-v1-done", "true");
  });

  it("the auto thread with id='default' is skipped", async () => {
    const { getUserItem, apiFetch, runMigration } = await getModules();

    const autoStorage = {
      threads: [
        { id: "default", title: "Default thread" },
        { id: "real-thread", title: "Real Thread" },
      ],
      threadMessages: {},
    };

    getUserItem.mockImplementation((key: string) => {
      if (key === "kf-migration-v1-done") return null;
      if (key === "kf-agent-threads") return null;
      if (key === "kf-agent-messages") return null;
      if (key === "koni_chat_v1") return JSON.stringify(autoStorage);
      return null;
    });
    apiFetch.mockResolvedValue(makeOkResponse());

    await runMigration();

    const postCalls = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.filter(
      (call: unknown[]) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    const bodies = postCalls.map((call: unknown[]) =>
      JSON.parse((call[1] as RequestInit).body as string),
    );

    expect(bodies.some((b: { id: string }) => b.id === "default")).toBe(false);
    expect(bodies.some((b: { id: string }) => b.id === "real-thread")).toBe(true);
  });

  it("runMigration is idempotent: calling it twice calls the API once", async () => {
    const { getUserItem, apiFetch, runMigration } = await getModules();

    let callCount = 0;
    getUserItem.mockImplementation((key: string) => {
      if (key === "kf-migration-v1-done") {
        callCount++;
        return callCount === 1 ? null : "true";
      }
      if (key === "kf-agent-threads") return null;
      if (key === "kf-agent-messages") return null;
      if (key === "koni_chat_v1") return null;
      return null;
    });
    apiFetch.mockResolvedValue(makeOkResponse());

    await runMigration();
    const firstCallCount = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.length;

    await runMigration();
    const secondCallCount = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.length;

    expect(secondCallCount).toBe(firstCallCount);
  });
});

describe("Suite 3: agentStore — API calls", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  async function freshStore() {
    const { useAgentStore } = await import("../stores/agentStore");
    useAgentStore.setState({ messages: [] });
    return useAgentStore;
  }

  async function getMockedApiFetch() {
    const apiFetchMod = await import("../lib/apiFetch");
    return apiFetchMod.apiFetch as ReturnType<typeof vi.fn>;
  }

  function makeJsonResponse(data: unknown, ok = true): Response {
    return {
      ok,
      status: ok ? 200 : 500,
      json: async () => data,
    } as unknown as Response;
  }

  it("fetchAgentMessages() calls GET /api/agent/messages and fills messages", async () => {
    const apiFetch = await getMockedApiFetch();
    const serverMessages = [
      {
        id: "m1", role: "user", content: "hello",
        agent_name: null, thinking: null,
        created_at: "2024-01-01T00:00:00Z",
      },
      {
        id: "m2", role: "assistant", content: "Hello!",
        agent_name: "corpus", thinking: null,
        created_at: "2024-01-01T00:01:00Z",
      },
    ];
    apiFetch.mockResolvedValue(makeJsonResponse(serverMessages));

    const store = await freshStore();
    await store.getState().fetchAgentMessages();

    const messages = store.getState().messages;
    expect(messages).toHaveLength(2);
    expect(messages[0].id).toBe("m1");
    expect(messages[0].role).toBe("user");
    expect(messages[1].id).toBe("m2");
    expect(messages[1].agentName).toBe("corpus");

    const getCall = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.find(
      (call: unknown[]) => call[0] === "/api/agent/messages",
    );
    expect(getCall).toBeDefined();
  });

  it("clearAgentMessages() calls DELETE /api/agent/messages and empties messages", async () => {
    const apiFetch = await getMockedApiFetch();
    apiFetch.mockResolvedValue(makeJsonResponse({}, true));

    const store = await freshStore();
    store.setState({
      messages: [
        { id: "m1", role: "user" as const, content: "hello", timestamp: 1000 },
      ],
    });

    await store.getState().clearAgentMessages();

    // Optimistic update: emptied at once.
    expect(store.getState().messages).toHaveLength(0);

    const deleteCall = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.find(
      (call: unknown[]) =>
        call[0] === "/api/agent/messages" &&
        (call[1] as RequestInit | undefined)?.method === "DELETE",
    );
    expect(deleteCall).toBeDefined();
  });

  it("a failed clearAgentMessages() resyncs with fetchAgentMessages()", async () => {
    const apiFetch = await getMockedApiFetch();

    // A failed DELETE resyncs by returning an empty array.
    apiFetch.mockImplementation((url: string, init?: RequestInit) => {
      if (init?.method === "DELETE") return makeJsonResponse({}, false);
      return makeJsonResponse([], true);
    });

    const store = await freshStore();
    store.setState({
      messages: [
        { id: "m1", role: "user" as const, content: "hello", timestamp: 1000 },
      ],
    });

    await store.getState().clearAgentMessages();

    // Check that fetchAgentMessages ran (the GET).
    const getCalls = (apiFetch as ReturnType<typeof vi.fn>).mock.calls.filter(
      (call: unknown[]) =>
        call[0] === "/api/agent/messages" &&
        !(call[1] as RequestInit | undefined)?.method,
    );
    expect(getCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("resetSession() calls DELETE /api/agent/messages then POST /api/agent/reset", async () => {
    const apiFetch = await getMockedApiFetch();
    apiFetch.mockResolvedValue(makeJsonResponse({}));

    const store = await freshStore();
    await store.getState().resetSession();

    const calls = (apiFetch as ReturnType<typeof vi.fn>).mock.calls;
    const deleteMsgCall = calls.find(
      (call: unknown[]) =>
        call[0] === "/api/agent/messages" &&
        (call[1] as RequestInit | undefined)?.method === "DELETE",
    );
    const resetCall = calls.find(
      (call: unknown[]) =>
        call[0] === "/api/agent/reset" &&
        (call[1] as RequestInit | undefined)?.method === "POST",
    );

    expect(deleteMsgCall).toBeDefined();
    expect(resetCall).toBeDefined();

    // Check that messages were emptied.
    expect(store.getState().messages).toHaveLength(0);
  });
});

/* ══════════════════════════════════════════════════════════════════════
   Suite 4: auto vs guided mode, by static analysis
   ══════════════════════════════════════════════════════════════════════ */

describe("Suite 4: auto and guided thread modes — static analysis", () => {
  let autoChatViewSource: string;
  let agentStoreSource: string;

  beforeEach(() => {
    autoChatViewSource = readFileSync(AUTO_CHATVIEW_PATH, "utf-8");
    agentStoreSource = readFileSync(AGENT_STORE_PATH, "utf-8");
  });

  it("the createThread call in auto/ChatView.tsx includes mode: \"auto\"", () => {
    expect(autoChatViewSource).toMatch(/mode:\s*["']auto["']/);
  });

  it("auto/ChatView.tsx filters server threads by mode === \"auto\"", () => {
    expect(autoChatViewSource).toMatch(/mode\s*===\s*["']auto["']/);
  });

  it("agentStore.ts makes no thread-based API calls", () => {
    // The /api/chat/threads path must not appear in agentStore.
    expect(agentStoreSource).not.toMatch(/\/api\/chat\/threads/);
  });

  it("agentStore.ts uses /api/agent/messages", () => {
    expect(agentStoreSource).toMatch(/\/api\/agent\/messages/);
  });
});
