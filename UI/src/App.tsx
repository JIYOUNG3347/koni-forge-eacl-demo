/**
 * Root component with React Router.
 *
 * - BrowserRouter with routes.
 * - Auth guard: redirect to /login if not authenticated.
 * - Auto/Guided layout selection based on systemStore.agentAutonomy.
 * - Role-based setup wizards: admin → /onboarding/admin, member → /onboarding/member.
 * - NO Context Provider nesting — Zustand handles state.
 */
import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { Toaster } from "sonner";
import { useAuthStore } from "./stores/authStore";
import { runMigration } from "./utils/migration";
import { FaultBoundary } from "./components/FaultBoundary";
import { LoginPage } from "./components/LoginPage";
import { GuidedLayout } from "./layouts/GuidedLayout";

// ── Views ─────────────────────────────────────────────
import ChatView from "./views/auto/ChatView";
import DataView from "./views/guided/DataView";
import TrainView from "./views/guided/TrainView";
import HomeView from "./views/guided/HomeView";

// ── Loading Spinner ────────────────────────────────────

function LoadingScreen({ message }: { message: string }) {
  return (
    <div className="min-h-screen bg-[#FAFAFA] flex items-center justify-center">
      <div className="flex items-center gap-3 text-neutral-400">
        <div className="w-5 h-5 border-2 border-neutral-300 border-t-neutral-600 rounded-full animate-spin" />
        <span className="text-[13px] font-medium">{message}</span>
      </div>
    </div>
  );
}

// ── Auth Guard ─────────────────────────────────────────

function AuthGuard() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isLoading = useAuthStore((s) => s.isLoading);

  // Migrate legacy localStorage chat data to server once after login
  useEffect(() => {
    if (isAuthenticated) {
      runMigration().catch(() => {});
    }
  }, [isAuthenticated]);

  if (isLoading) {
    return <LoadingScreen message="Checking authentication..." />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}










// ── Login Wrapper ──────────────────────────────────────

function LoginWrapper() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const setAuth = useAuthStore((s) => s.setAuth);

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return (
    <LoginPage
      onLogin={(uid, role) => {
        const token = localStorage.getItem("kf-token") ?? "";
        setAuth(token, uid, role as "admin" | "user");
      }}
    />
  );
}

// ── Settings Layout (standalone, outside nav tabs) ────


// ── App Root ───────────────────────────────────────────

export default function App() {
  const checkSession = useAuthStore((s) => s.checkSession);

  // Validate stored session on mount
  useEffect(() => {
    checkSession();
  }, [checkSession]);

  return (
    <>
      <FaultBoundary>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginWrapper />} />
            <Route element={<AuthGuard />}>
              <Route element={<GuidedLayout />}>
                <Route path="/home" element={<HomeView />} />
                <Route path="/data" element={<DataView />} />
                <Route path="/train" element={<TrainView />} />
                <Route path="/chat" element={<ChatView />} />
                <Route index element={<Navigate to="/home" replace />} />
                <Route path="*" element={<Navigate to="/home" replace />} />
              </Route>
            </Route>
          </Routes>
        </BrowserRouter>
      </FaultBoundary>
      <Toaster position="bottom-left" richColors closeButton duration={3000} />
    </>
  );
}




