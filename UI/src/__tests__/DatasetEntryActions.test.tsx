import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect } from "vitest";

import { DatasetEntry } from "../views/guided/DataView";

vi.mock("../i18n", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../i18n")>();
  const id = (k: string) => k;
  const fmt = (k: string, ...a: unknown[]) => mod.formatArgs(k, a);
  return { ...mod, t: id, useT: () => id, tp: fmt, useTp: () => fmt };
});

const item = {
  id: "ds1",
  name: "doc.pdf",
  type: "raw",
  size: 1024,
  format: "pdf",
  created_at: 1780000000000,
  status: "ready",
} as never;

describe("DatasetEntry — detail and delete buttons are independent", () => {
  it("clicking detail calls onOpenDetail(item)", () => {
    const onOpenDetail = vi.fn();
    render(<DatasetEntry item={item} onOpenDetail={onOpenDetail} onDelete={() => {}} />);
    fireEvent.click(screen.getByTitle("Details"));
    expect(onOpenDetail).toHaveBeenCalledTimes(1);
    expect(onOpenDetail).toHaveBeenCalledWith(item);
  });

  it("clicking the bin calls only onDelete and does not open the modal", () => {
    const onOpenDetail = vi.fn();
    const onDelete = vi.fn();
    render(<DatasetEntry item={item} onOpenDetail={onOpenDetail} onDelete={onDelete} />);
    fireEvent.click(screen.getByTitle("Delete"));
    expect(onDelete).toHaveBeenCalledWith("ds1");
    expect(onOpenDetail).not.toHaveBeenCalled();
  });

  it("the QA preview (eye) button is absent from processed and training cards", () => {
    const processedItem = { id: "ds1", name: "qa.json", type: "training", size: 1024, format: "json", created_at: 1780000000000, status: "ready" } as never;
    render(<DatasetEntry item={processedItem} onOpenDetail={() => {}} onDelete={() => {}} />);
    expect(screen.queryByTitle("QA preview")).not.toBeInTheDocument();
  });
});
