import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..");

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "__tests__" || entry === "node_modules") continue;
      walk(full, out);
    } else if (/\.tsx?$/.test(entry)) {
      // The seam itself *defines* `t`; its own body is not a call site.
      if (full.endsWith(join("i18n", "index.ts"))) continue;
      out.push(full);
    }
  }
  return out;
}

/**
 * Lines where a `t(` / `tp(` call sits at module scope.
 *
 * Depth is tracked by counting braces, ignoring those inside strings, template
 * literals and comments — a naive count trips over `{` in JSX className strings.
 */
function moduleScopeTCalls(source: string): number[] {
  const hits: number[] = [];
  let depth = 0;
  let inBlockComment = false;

  source.split("\n").forEach((line, i) => {
    const trimmed = line.trim();

    if (inBlockComment) {
      if (trimmed.includes("*/")) inBlockComment = false;
      return;
    }
    if (trimmed.startsWith("/*")) {
      if (!trimmed.includes("*/")) inBlockComment = true;
      return;
    }
    if (trimmed.startsWith("//") || trimmed.startsWith("*")) return;

    // Strip strings/templates so their braces and `t(` text don't count.
    const stripped = line
      .replace(/`(?:[^`\\]|\\.)*`/g, "``")
      .replace(/"(?:[^"\\]|\\.)*"/g, '""')
      .replace(/'(?:[^'\\]|\\.)*'/g, "''")
      .replace(/\/\/.*$/, "");

    // A `t(` / `tp(` at depth 0 runs when the module loads.
    if (depth === 0 && /(?<![\w.$])tp?\(/.test(stripped)) hits.push(i + 1);

    for (const ch of stripped) {
      if (ch === "{" || ch === "(") depth++;
      else if (ch === "}" || ch === ")") depth = Math.max(0, depth - 1);
    }
  });

  return hits;
}

const FILES = walk(SRC);

describe("module-scope t() guard", () => {
  it("finds the source tree", () => {
    expect(FILES.length).toBeGreaterThan(50);
  });

  it("detects a module-scope call in a synthetic sample", () => {
    // The guard has to actually catch the shape it exists for.
    const bad = ['import { t } from "./i18n";', "", 'const TABS = [{ label: t("Home") }];'].join("\n");
    expect(moduleScopeTCalls(bad)).toEqual([3]);
  });

  it("detects a module-scope tp() too", () => {
    const bad = ['import { tp } from "./i18n";', "", 'const MSG = tp("{0}", 3);'].join("\n");
    expect(moduleScopeTCalls(bad)).toEqual([3]);
  });

  it("does not flag calls inside a component", () => {
    const good = ["export function View() {", "  const t = useT();", '  return <p>{t("Home")}</p>;', "}"].join(
      "\n",
    );
    expect(moduleScopeTCalls(good)).toEqual([]);
  });

  it("no source file calls t() at module scope", () => {
    const offenders: string[] = [];
    for (const file of FILES) {
      const lines = moduleScopeTCalls(readFileSync(file, "utf-8"));
      for (const line of lines) {
        offenders.push(`${file.replace(SRC, "src")}:${line}`);
      }
    }
    expect(
      offenders,
      "A module-scope t()/tp() freezes the language at import time — the screen will " +
        "only half update when the user switches. Keep the Korean source as the key " +
        "in the constant and translate where it is used, e.g. t(tab.label).",
    ).toEqual([]);
  });
});
