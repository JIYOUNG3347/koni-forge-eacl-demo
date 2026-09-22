import { describe, it, expect } from "vitest";

import { groupByBaseId, type DatasetItem } from "../stores/dataStore";

// Mirrors the inline helpers in TrainView so they can be verified here.
function versionedFolderName(baseId: string, ver: number): string {
  return ver <= 1 ? baseId : `${baseId}_v${ver}`;
}

const mk = (over: Partial<DatasetItem>): DatasetItem =>
  ({
    id: over.name ?? "x",
    name: over.name ?? "x",
    type: "training",
    size: 0,
    format: "JSON",
    created_at: over.created_at ?? 0,
    status: "ready",
    base_id: over.base_id ?? "dataset_xxx",
    version: over.version ?? 1,
    qa_count: over.qa_count,
  }) as DatasetItem;

describe("versionedFolderName", () => {
  it("v1 (carry-over) is the bare base_id", () => {
    expect(versionedFolderName("dataset_xxx", 1)).toBe("dataset_xxx");
  });

  it("v2+ adds the _v{N} suffix", () => {
    expect(versionedFolderName("dataset_xxx", 2)).toBe("dataset_xxx_v2");
    expect(versionedFolderName("dataset_xxx", 10)).toBe("dataset_xxx_v10");
  });
});

describe("corpus groupByBaseId for TrainView", () => {

  it("several versions sort by version descending, newest first", () => {
    const items = [
      mk({ name: "dataset_xxx", version: 1, created_at: 100 }),
      mk({ name: "dataset_xxx_v2", base_id: "dataset_xxx", version: 2, created_at: 200 }),
      mk({ name: "dataset_xxx_v3", base_id: "dataset_xxx", version: 3, created_at: 300 }),
    ];
    const groups = groupByBaseId(items);
    expect(groups[0].versions.map((v) => v.version)).toEqual([3, 2, 1]);
    // With the default (selectedVersion=null) the latest (versions[0]) applies.
    const effectiveVersion = groups[0].versions[0].version;
    expect(effectiveVersion).toBe(3);
  });

  it("a single version (carry-over) needs no secondary selector", () => {
    const items = [mk({ name: "dataset_xxx", version: 1 })];
    const groups = groupByBaseId(items);
    expect(groups[0].versions).toHaveLength(1);
    const isMultiVersion = groups[0].versions.length > 1;
    expect(isMultiVersion).toBe(false);
  });

  it("selecting v1 sends the bare base_id (carry-over)", () => {
    const groups = groupByBaseId([mk({ name: "dataset_xxx", version: 1 })]);
    const ver = groups[0].versions[0].version ?? 1;
    expect(versionedFolderName(groups[0].base_id, ver)).toBe("dataset_xxx");
  });

  it("selecting v2 sends base_id_v2", () => {
    const items = [
      mk({ name: "dataset_xxx", version: 1 }),
      mk({ name: "dataset_xxx_v2", base_id: "dataset_xxx", version: 2 }),
    ];
    const groups = groupByBaseId(items);
    expect(versionedFolderName(groups[0].base_id, 2)).toBe("dataset_xxx_v2");
  });
});
