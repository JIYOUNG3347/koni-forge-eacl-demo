import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect } from "vitest";

import { GuidedPausedBanner } from "../components/agent/GuidedPausedBanner";
import type { PendingChain } from "../stores/agentStore";

vi.mock("../i18n", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../i18n")>();
  const id = (k: string) => k;
  const fmt = (k: string, ...a: unknown[]) => mod.formatArgs(k, a);
  return { ...mod, t: id, useT: () => id, tp: fmt, useTp: () => fmt };
});

const KBD_PENDING: PendingChain = {
  nextAgent: "boundary",
  announce: "RAG indexing complete — start the KBD analysis?",
  params: null,
};

describe("GuidedPausedBanner", () => {
  it("shows the paused title, guidance and a dynamic example for the waiting stage", () => {
    render(<GuidedPausedBanner pending={KBD_PENDING} isBusy={false} onResume={vi.fn()} />);
    expect(screen.getByText("Paused")).toBeInTheDocument();
    expect(screen.getByText(/Describe the next step in plain language/)).toBeInTheDocument();
    // boundary maps to the agent label "KBD Analysis", which appears in the example.
    expect(screen.getByText(/run KBD Analysis/)).toBeInTheDocument();
  });

  it("updates the example for a different waiting stage", () => {
    const tuningPending: PendingChain = {
      nextAgent: "tuning",
      announce: "Start training?",
      params: null,
    };
    render(<GuidedPausedBanner pending={tuningPending} isBusy={false} onResume={vi.fn()} />);
    expect(screen.getByText(/run Training/)).toBeInTheDocument();
  });

  it("calls onResume when the resume button is clicked", () => {
    const onResume = vi.fn();
    render(<GuidedPausedBanner pending={KBD_PENDING} isBusy={false} onResume={onResume} />);
    fireEvent.click(screen.getByRole("button", { name: /Resume/ }));
    expect(onResume).toHaveBeenCalledTimes(1);
  });

  it("disables the resume button while busy", () => {
    render(<GuidedPausedBanner pending={KBD_PENDING} isBusy={true} onResume={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Resume/ })).toBeDisabled();
  });
}, 10000);
