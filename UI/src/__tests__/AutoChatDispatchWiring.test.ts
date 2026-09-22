import { readFileSync } from "fs";
import { join } from "path";

import { describe, it, expect } from "vitest";

const SIDEBAR = join(__dirname, "..", "components", "AgentSidebar.tsx");
const source = readFileSync(SIDEBAR, "utf-8");

/** Executable code only, so a path mentioned in a comment cannot pollute the check. */
const code = source
  .replace(/\/\*[\s\S]*?\*\//g, "")
  .split("\n")
  .filter((l) => !l.trim().startsWith("//"))
  .join("\n");

/** The body of one useCallback definition, so the next handler is not mixed in. */
function callbackBody(name: string): string {
  const start = code.indexOf(`const ${name}`);
  if (start < 0) return "";
  const end = code.indexOf("]);", start);
  return end < 0 ? code.slice(start) : code.slice(start, end + 3);
}

describe("auto chat dispatch wiring", () => {
  it("auto send no longer calls /api/agent/ask", () => {
    expect(code).not.toContain("/api/agent/ask");
  });

  it("handleSendAuto calls dispatchMessage", () => {
    const body = callbackBody("handleSendAuto");
    expect(body).not.toBe("");
    expect(body).toContain("dispatchMessage(text)");
  });

  it("streaming state lives only in the store (local autoStreaming removed)", () => {
    expect(code).not.toContain("autoStreaming");
    expect(code).toContain("const streaming = storeStreaming;");
  });

  it("auto does not use the guided confirmation modal intercept", () => {
    expect(callbackBody("handleSendAuto")).not.toContain("openConfirmModal");
    // It must still be there on the guided side.
    expect(callbackBody("handleSendGuided")).toContain("openConfirmModal");
    expect(code).toContain("const handleSendGuided");
  });
});
