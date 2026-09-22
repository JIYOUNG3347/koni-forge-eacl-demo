import { readFileSync, readdirSync, statSync } from "fs";
import { join, relative } from "path";
import { describe, it, expect } from "vitest";

const SRC_DIR = join(__dirname, "..");

function walkTs(dir: string): string[] {
  const results: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "__tests__" || entry === "node_modules") continue;
      results.push(...walkTs(full));
    } else if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
      if (entry === "test-setup.ts") continue;
      results.push(full);
    }
  }
  return results;
}

// Scope: every .ts and .tsx under src/, excluding the test directory.
const SOURCE_FILES = walkTs(SRC_DIR);

// Real call patterns containing a /v1/ path.
// Only /v1/ inside a string literal counts; absolute external URLs are excluded.
const V1_CALL_RE = /["'`](?!https?:\/\/)([^"'`]*\/v1\/[^"'`]*?)["'`]/g;

describe("no /v1/ API calls in the front end", () => {
  it("no source file under src/ calls a /v1/ API", () => {
    const violations: { file: string; line: number; match: string }[] = [];

    for (const filePath of SOURCE_FILES) {
      const content = readFileSync(filePath, "utf-8");
      const lines = content.split("\n");

      lines.forEach((line, idx) => {
        // Skip comment lines.
        const trimmed = line.trimStart();
        if (trimmed.startsWith("//") || trimmed.startsWith("*")) return;

        let m: RegExpExecArray | null;
        V1_CALL_RE.lastIndex = 0;
        while ((m = V1_CALL_RE.exec(line)) !== null) {
          violations.push({
            file: relative(SRC_DIR, filePath),
            line: idx + 1,
            match: m[0],
          });
        }
      });
    }

    if (violations.length > 0) {
      const report = violations
        .map((v) => `  ${v.file}:${v.line}  ${v.match}`)
        .join("\n");
      expect.fail(
        `Found ${violations.length} /v1/ API call(s) in the front end:\n${report}\n` +
          `Use the /api/ path instead.`
      );
    }

    expect(violations).toHaveLength(0);
  });

  it("at least one file is scanned (catches a broken glob)", () => {
    // A broken glob would find no files and make the previous test pass forever.
    expect(SOURCE_FILES.length).toBeGreaterThan(10);
  });
});
