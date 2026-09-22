import { describe, expect, it } from "vitest";

import { formatGb, groupStorage } from "../lib/storageGroups";

const MB = 1024; // 1GB

describe("groupStorage", () => {
  it("groups the storage directories into three", () => {
    const { groups } = groupStorage(
      {
        raw_corpus: { size_mb: 10 * MB },
        corpus: { size_mb: 8 * MB },
        chroma: { size_mb: 2 * MB },
        models: { size_mb: 40 * MB },
        checkpoints: { size_mb: 20 * MB },
        outputs: { size_mb: 5 * MB },
        logs: { size_mb: 1 * MB },
      },
      100,
      900,
    );
    expect(groups.map((g) => [g.key, g.gb])).toEqual([
      ["dataset", 20],
      ["model", 60],
      ["output", 6],
    ]);
  });

  it("disk usage not explained by storage stays as 'Other'", () => {
    const { otherGb } = groupStorage({ models: { size_mb: 10 * MB } }, 30, 100);
    expect(otherGb).toBe(20);
  });

  it("a storage total above disk usage never exceeds 100%", () => {
    const { groups } = groupStorage({ models: { size_mb: 200 * MB } }, 50, 10);
    expect(groups.find((g) => g.key === "model")!.pct).toBeCloseTo(100);
  });

  it("missing categories or keys count as 0", () => {
    const { groups, otherGb } = groupStorage(null, 0, 0);
    expect(groups.every((g) => g.gb === 0)).toBe(true);
    expect(otherGb).toBe(0);
  });

  it("a negative disk value is clamped to 0", () => {
    const { otherGb, freeGb } = groupStorage({}, -5, -1);
    expect(otherGb).toBe(0);
    expect(freeGb).toBe(0);
  });
});

describe("formatGb", () => {
  it("reduces the precision as the size grows", () => {
    expect(formatGb(1178)).toBe("1,178");
    expect(formatGb(42.7)).toBe("43");
    expect(formatGb(4.27)).toBe("4.3");
    expect(formatGb(0.42)).toBe("0.42");
  });
});
