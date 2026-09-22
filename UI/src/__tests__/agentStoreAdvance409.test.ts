
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../i18n", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../i18n")>();
  const id = (k: string) => k;
  const fmt = (k: string, ...a: unknown[]) => mod.formatArgs(k, a);
  return { ...mod, t: id, useT: () => id, tp: fmt, useTp: () => fmt };
});

vi.mock("../lib/apiFetch", () => ({ apiFetch: vi.fn() }));
// Bypass the storage backend so saveAgentMessage does not touch it.
vi.mock("../utils/storage", () => ({
  getUserItem: vi.fn(() => null),
  setUserItem: vi.fn(),
}));

const ADVANCE_URL = "/api/agent/chain/advance";
const PIPELINE_URL = "/api/agent/pipeline";

function makeResponse(status: number, bodyObj: unknown = {}): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    body: null, // consumeDispatchStream returns at once without a body
    json: async () => bodyObj,
  } as unknown as Response;
}

async function getStore() {
  const { useAgentStore } = await import("../stores/agentStore");
  useAgentStore.setState({
    messages: [],
    isStreaming: false,
    abortController: null,
  });
  return useAgentStore;
}

async function getMockedApiFetch() {
  const mod = await import("../lib/apiFetch");
  return mod.apiFetch as ReturnType<typeof vi.fn>;
}

describe("advanceChain — graceful retry on 409", () => {
  beforeEach(async () => {
    const apiFetch = await getMockedApiFetch();
    apiFetch.mockReset();
  });

  it("409, resync, retry, success (no error message)", async () => {
    const store = await getStore();
    const apiFetch = await getMockedApiFetch();

    let advanceCalls = 0;
    let pipelineCalls = 0;
    apiFetch.mockImplementation((url: string) => {
      if (url === ADVANCE_URL) {
        advanceCalls += 1;
        return Promise.resolve(
          advanceCalls === 1
            ? makeResponse(409, { detail: "No pending chain to advance" })
            : makeResponse(200),
        );
      }
      if (url === PIPELINE_URL) {
        pipelineCalls += 1;
        return Promise.resolve(makeResponse(200, {}));
      }
      return Promise.resolve(makeResponse(200));
    });

    await store.getState().advanceChain("approve");

    expect(advanceCalls).toBe(2); // initial call plus one retry
    expect(pipelineCalls).toBeGreaterThanOrEqual(1); // the resync ran
    const errMsgs = store.getState().messages.filter((m) => m.content.startsWith("Error:"));
    expect(errMsgs).toHaveLength(0);
  }, 10000);

  it("409, retry also 409, error message shown", async () => {
    const store = await getStore();
    const apiFetch = await getMockedApiFetch();

    let advanceCalls = 0;
    apiFetch.mockImplementation((url: string) => {
      if (url === ADVANCE_URL) {
        advanceCalls += 1;
        return Promise.resolve(makeResponse(409, { detail: "No pending chain to advance" }));
      }
      if (url === PIPELINE_URL) return Promise.resolve(makeResponse(200, {}));
      return Promise.resolve(makeResponse(200));
    });

    await store.getState().advanceChain("approve");

    expect(advanceCalls).toBe(2); // initial call plus exactly one retry
    const errMsgs = store.getState().messages.filter((m) => m.content.startsWith("Error:"));
    expect(errMsgs).toHaveLength(1);
  }, 10000);

  it("200 means no resync and no retry", async () => {
    const store = await getStore();
    const apiFetch = await getMockedApiFetch();

    let advanceCalls = 0;
    let pipelineCalls = 0;
    apiFetch.mockImplementation((url: string) => {
      if (url === ADVANCE_URL) {
        advanceCalls += 1;
        return Promise.resolve(makeResponse(200));
      }
      if (url === PIPELINE_URL) {
        pipelineCalls += 1;
        return Promise.resolve(makeResponse(200, {}));
      }
      return Promise.resolve(makeResponse(200));
    });

    await store.getState().advanceChain("skip");

    expect(advanceCalls).toBe(1);
    expect(pipelineCalls).toBe(0);
  });
});
