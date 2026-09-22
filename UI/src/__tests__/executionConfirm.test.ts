import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { EXECUTION_LABELS } from "../stores/agentStore";

const store = readFileSync(resolve(__dirname, "../stores/agentStore.ts"), "utf-8");
const sidebar = readFileSync(resolve(__dirname, "../components/AgentSidebar.tsx"), "utf-8");

describe("pending execution wiring", () => {
  it("maps the backend field into the store", () => {
    expect(store).toContain("pending_execution?: PendingExecution | null");
    expect(store).toContain("pendingExecution: raw.pending_execution ?? null");
  });

  it("posts to the execution endpoint, not the chain one", () => {
    expect(store).toContain('"/api/agent/execution/confirm"');
  });

  it("hides the modal optimistically so it cannot double-fire", () => {
    const block = store.slice(store.indexOf("confirmExecution: async"));
    expect(block.slice(0, 900)).toContain("pendingExecution: null");
  });

  it("retries once on 409 — a restart can desync the pending state", () => {
    const block = store.slice(store.indexOf("confirmExecution: async"), store.indexOf("confirmEvalSelection: async"));
    expect(block).toContain("res.status === 409");
    expect(block).toContain("fetchPipelineStatus()");
  });

  it("cancel does not stream or post overrides", () => {
    const block = store.slice(store.indexOf("confirmExecution: async"), store.indexOf("confirmEvalSelection: async"));
    const cancel = block.slice(block.indexOf('action === "cancel"'), block.indexOf("const label"));
    expect(cancel).toContain('"cancel"');
    expect(cancel).not.toContain("consumeDispatchStream");
  });
});

describe("modal reuse", () => {
  it("renders the same modal the chain path uses", () => {
    // Two different confirmation screens would make the user learn both.
    const block = sidebar.slice(sidebar.indexOf("pipelineStatus.pendingExecution &&"));
    expect(block.slice(0, 400)).toContain("GuidedConfirmParamsModal");
  });

  it("only shows in guided mode", () => {
    expect(sidebar).toContain("isGuided && pipelineStatus.pendingExecution");
  });

  it("passes the tool's own params, not the chain's", () => {
    const block = sidebar.slice(sidebar.indexOf("pipelineStatus.pendingExecution &&"));
    expect(block.slice(0, 400)).toContain("pipelineStatus.pendingExecution.params");
  });
});

describe("labels", () => {
  it("names what was approved for every editable tool", () => {
    // With only "Proceeding" in the history, what was approved is lost.
    expect(Object.keys(EXECUTION_LABELS).sort()).toEqual(["start_training_job"]);
  });

  it("every label is a non-empty English string", () => {
    for (const label of Object.values(EXECUTION_LABELS)) {
      expect(label.trim()).not.toBe("");
      expect(label).not.toMatch(/[\uac00-\ud7a3]/);
    }
  });
});

describe("modal labels", () => {
  const modal = readFileSync(
    resolve(__dirname, "../components/agent/GuidedConfirmParamsModal.tsx"),
    "utf-8",
  );

  it("no raw Korean sits in a rendered JSX attribute", () => {
    // label=, title= and placeholder= go straight to the screen.
    const raw = [...modal.matchAll(/\b(?:label|title|placeholder)="([^"]*[\uac00-\ud7a3][^"]*)"/g)];
    expect(raw.map((m) => m[1])).toEqual([]);
  });

  it("no raw Korean is passed to the row builder", () => {
    // rows.push(r("...", ...)) is a screen label too.
    const raw = [...modal.matchAll(/\br\(\s*"([^"]*[\uac00-\ud7a3][^"]*)"/g)];
    expect(raw.map((m) => m[1])).toEqual([]);
  });

  it("labels go through the translate seam", () => {
    const used = [...modal.matchAll(/\bt\(\s*"([^"]*)"/g)].map((m) => m[1]);
    expect(used.length).toBeGreaterThan(20);
  });
});
