/**
 * Shell -- Application shell with header bar, centered tabs, user menu.
 *
 * Single-line header: logo (left) + tabs (center) + status/settings/user (right).
 */

import React, { useState, useRef, useEffect } from "react";
import { useNavigate, NavLink } from "react-router-dom";
import {
  Loader2, Clock,
  CircleDot, LogOut, User, ChevronDown,
  Home, DatabaseZap, Brain, MessageCircle,
} from "lucide-react";
import { apiFetch } from "../lib/apiFetch";
import { usePolling } from "../hooks/usePolling";
import { useTrainActive } from "../hooks/useTrainActive";
import { KoniForgeIcon } from "./icons/KoniForgeIcon";
import { Button } from "./ui/button";
import { usePipelineStore } from "../stores/pipelineStore";
import { useAuthStore } from "../stores/authStore";
import { AgentSidebar } from "./AgentSidebar";
import { tp, useT } from "../i18n";

/* =========================
   Tab definitions
   ========================= */
const TABS = [
  { to: "/home",  label: "Home",     icon: Home },
  { to: "/data",  label: "Upload", icon: DatabaseZap },
  { to: "/train", label: "Train",   icon: Brain },
  { to: "/chat",  label: "Chat",   icon: MessageCircle },
];

/* =========================
   QueueStatus (compact pill)
   ========================= */
function QueueStatus() {
  const t = useT();
  const jobs = usePipelineStore((s) => s.jobs);
  const runningJob = usePipelineStore((s) => s.runningJob);

  const activeTrainingCount = useTrainActive().total;
  const [globalQueueBusy, setGlobalQueueBusy] = useState(false);
  const [globalQueueTotal, setGlobalQueueTotal] = useState(0);


  // Global GPU queue state, including other users' jobs.
  usePolling(async () => {
    try {
      const r = await apiFetch("/api/system/gpu-queue");
      if (!r.ok) return;
      const d = await r.json();
      setGlobalQueueBusy(Boolean(d?.busy));
      setGlobalQueueTotal(Number(d?.total ?? 0));
    } catch { /* ignore */ }
  }, 5000);


  const pillBase =
    "inline-flex items-center gap-1 h-6 px-2.5 rounded-md text-[10px] font-bold border whitespace-nowrap shrink-0";

  if (runningJob) {
    return (
      <span className={`${pillBase} bg-red-500 text-white border-red-600`}>
        <Loader2 className="w-3 h-3 animate-spin" />
        {t("Training")}
      </span>
    );
  }

  if (activeTrainingCount > 0) {
    return (
      <span className={`${pillBase} bg-red-500 text-white border-red-600`}>
        <Loader2 className="w-3 h-3 animate-spin" />
        {tp("Training{0}", activeTrainingCount > 1 ? ` (${activeTrainingCount})` : "")}
      </span>
    );
  }




  const queuedJobs = jobs.filter((j) => j.status === "queued");
  if (queuedJobs.length > 0) {
    return (
      <span className={`${pillBase} bg-neutral-200 text-neutral-800 border-neutral-300`}>
        <Clock className="w-3 h-3" />
        {tp("{0} waiting", queuedJobs.length)}
      </span>
    );
  }

  // Another user is using the GPU.
  if (globalQueueBusy && activeTrainingCount === 0) {
    return (
      <span className={`${pillBase} bg-neutral-200 text-neutral-800 border-neutral-300`}>
        <Loader2 className="w-3 h-3 animate-spin" />
        {tp("GPU busy ({0})", globalQueueTotal)}
      </span>
    );
  }

  return (
    <span className={`${pillBase} bg-neutral-900 text-white border-neutral-900`}>
      <CircleDot className="w-3 h-3" />{t("Queued")}</span>
  );
}

/* =========================
   UserMenu (dropdown)
   ========================= */
function UserMenu({ userId }: { userId: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const logout = useAuthStore((s) => s.logout);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <Button
        variant={open ? "secondary" : "ghost"}
        size="sm"
        onClick={() => setOpen(o => !o)}
        className="gap-1.5 h-8"
      >
        <User className="w-3.5 h-3.5" />
        <span className="text-xs font-medium max-w-[80px] truncate">{userId}</span>
        <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </Button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-48 bg-popover rounded-lg border shadow-lg py-1 z-50">
          <button
            onClick={() => { setOpen(false); logout(); }}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />{t("Sign out")}</button>
        </div>
      )}
    </div>
  );
}

/* =========================
   Navigation Item types
   ========================= */
/* =========================
   Main Layout
   ========================= */
export function Shell({
  children,
  userId,
  lite = false,
}: {
  children: React.ReactNode;
  userId?: string;
  lite?: boolean;
}) {
  const t = useT();
  const navigate = useNavigate();
  const jobs = usePipelineStore((s) => s.jobs);
  const runningJob = usePipelineStore((s) => s.runningJob);

  const mostRelevantJob = runningJob ?? (jobs.length > 0 ? [...jobs].sort((a, b) => b.created_at - a.created_at)[0] : null);

  const tabs = TABS;

  return (
    <div className="min-h-screen w-full bg-background flex flex-col">
      {/* Job Progress Bar (thin, animated) */}
      {mostRelevantJob?.status === "running" && (
        <div className="h-[2px] w-full bg-muted overflow-hidden fixed top-0 left-0 z-50">
          {mostRelevantJob.progress ? (
            <div
              className="h-full bg-primary transition-all duration-500 ease-out"
              style={{ width: `${mostRelevantJob.progress}%` }}
            />
          ) : (
            <div
              className="h-full w-1/3 bg-gradient-to-r from-transparent via-primary to-transparent"
              style={{ animation: "progressSlide 1.5s ease-in-out infinite" }}
            />
          )}
        </div>
      )}
      <style>{`
        @keyframes progressSlide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
      `}</style>

      {/* Header Bar — single line */}
      <header className="sticky top-0 z-40 bg-background/80 backdrop-blur-xl border-b">
        <div className="flex items-center h-14 px-5 relative">
          {/* Left: Logo */}
          <button
            onClick={() => navigate("/home")}
            className="flex items-center gap-2 flex-shrink-0 group"
          >
            <KoniForgeIcon size={24} strokeWidth={1.6} className="text-foreground group-hover:scale-105 transition-transform duration-300" />
            <span className="text-sm tracking-tight hidden sm:inline">
              <span className="font-extrabold text-foreground">KONI</span>
              <span className="font-light text-muted-foreground/40 mx-px">-</span>
              <span className="font-normal text-muted-foreground">Forge</span>
            </span>
          </button>

          {/* Center: Tabs (absolute center) */}
          {!lite && (
            <nav className="absolute left-1/2 -translate-x-1/2 flex items-center gap-0.5">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <NavLink
                    key={tab.to}
                    to={tab.to}
                    className={({ isActive }) =>
                      `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold transition-all duration-150 whitespace-nowrap ${
                        isActive
                          ? "bg-neutral-900 text-white shadow-sm"
                          : "text-neutral-400 hover:text-neutral-900 hover:bg-neutral-100"
                      }`
                    }
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {t(tab.label)}
                  </NavLink>
                );
              })}
            </nav>
          )}

          {/* Right: Status + Settings + User */}
          <div className="ml-auto flex items-center gap-2 flex-shrink-0">
            <QueueStatus />

            {userId && <UserMenu userId={userId} />}
          </div>
        </div>
      </header>

      {/* Main content and agent sidebar (always expanded) */}
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 p-6 overflow-y-auto">{children}</main>
        {!lite && (
          <aside className="w-96 flex-shrink-0 h-[calc(100vh-57px)]">
            <AgentSidebar />
          </aside>
        )}
      </div>

    </div>
  );
}
