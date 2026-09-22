/**
 * GuidedLayout — Shell handles tabs + agent sidebar.
 */

import React from "react";
import { Outlet } from "react-router-dom";
import { Shell } from "../components/Shell";
import { useAuthStore } from "../stores/authStore";

export function GuidedLayout() {
  const userId = useAuthStore((s) => s.userId);

  return (
    <Shell userId={userId}>
      <Outlet />
    </Shell>
  );
}
