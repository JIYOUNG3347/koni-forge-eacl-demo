import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..");
const DATA_VIEW = readFileSync(join(SRC, "views/guided/DataView.tsx"), "utf-8");
const DATA_STORE = readFileSync(join(SRC, "stores/dataStore.ts"), "utf-8");

function codeOnly(src: string): string {
  return src
    .split("\n")
    .filter((l) => {
      const t = l.trim();
      return !(t.startsWith("//") || t.startsWith("*") || t.startsWith("/*"));
    })
    .join("\n");
}

describe("where the upload handoff text comes from", () => {
  it("the front end no longer assembles the sentence", () => {
    const code = codeOnly(DATA_VIEW);
    expect(code).not.toContain("uploaded a QA dataset (JSON/JSONL) to the folder");
    expect(code).not.toContain("uploaded documents to the folder");
  });

  it("no tool-call fragment is left in the front end", () => {
    // It moved to where the backend `TOOL_TOKENS` guard can reach it.
    expect(codeOnly(DATA_VIEW)).not.toContain("start_preprocessing(job_name=");
  });

  it("the front end does not decide on TWIST", () => {
    // Whether augmentation is possible is backend knowledge.
    // The on-screen tooltip is display text and out of scope —
    // only sentences that instruct the agent are checked.
    expect(codeOnly(DATA_VIEW)).not.toContain("Run preprocessing and augmentation (filtering + TWIST)");
  });

  it("it sends the text the server gave, unchanged", () => {
    expect(DATA_VIEW).toContain(
      'void dispatchMessage(agentHandoff, target === "gen" ? "retrieval" : "tuning")',
    );
  });

  it("with no text, nothing is sent", () => {
    // The moment the screen invents one, there are two copies again.
    expect(DATA_VIEW).toContain('agentAutonomy === "guided" && agentHandoff');
  });

  it("the handoff agent depends on the target", () => {
    // gen → retrieval (indexing) / train → tuning (the QA dataset is used as-is)
    expect(DATA_VIEW).toContain('target === "gen" ? "retrieval" : "tuning"');
  });
});

describe("the store reads the text from the response", () => {
  it("uploadFiles returns folder and agentHandoff together", () => {
    expect(DATA_STORE).toContain("return { folder: uploadedFolder, agentHandoff };");
    expect(DATA_STORE).toContain("export interface UploadResult");
  });

  it("a non-string value becomes null", () => {
    // An older server omits the field; undefined must not be forwarded.
    expect(DATA_STORE).toContain(
      'typeof resp.agent_handoff === "string" ? resp.agent_handoff : null',
    );
  });

  it("folder still comes from the response", () => {
    expect(DATA_STORE).toContain("uploadedFolder = resp.folder ?? \"\";");
  });
});
