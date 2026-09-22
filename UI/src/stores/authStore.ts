/**
 * authStore — Zustand store for authentication state.
 *
 * State: token, userId, role, isAuthenticated, isLoading
 * Actions: login, logout, checkSession, setAuth
 */

import { create } from "zustand";
import { apiFetch } from "../lib/apiFetch";
import { clearCurrentUserStorage } from "../utils/storage";

interface AuthState {
  token: string;
  userId: string;
  role: "admin" | "user" | "";
  isAuthenticated: boolean;
  isLoading: boolean;

  setAuth: (token: string, userId: string, role: "admin" | "user") => void;
  login: (userId: string, password: string) => Promise<void>;
  logout: () => void;
  checkSession: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: "",
  userId: "",
  role: "",
  isAuthenticated: false,
  isLoading: true,

  setAuth: (token, userId, role) => {
    localStorage.setItem("kf-token", token);
    localStorage.setItem("kf-user-id", userId);
    localStorage.setItem("kf-user-role", role);
    set({ token, userId, role, isAuthenticated: true, isLoading: false });
  },

  login: async (userId: string, password: string) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: userId, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Incorrect ID or password.");
    }
    const data = await res.json();
    localStorage.setItem("kf-token", data.token);
    localStorage.setItem("kf-user-id", data.user_id);
    localStorage.setItem("kf-user-role", data.role);
    set({
      token: data.token,
      userId: data.user_id,
      role: data.role,
      isAuthenticated: true,
      isLoading: false,
    });
  },

  logout: () => {
    const token = localStorage.getItem("kf-token");
    if (token) {
      fetch("/api/auth/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    clearCurrentUserStorage(); // must run before kf-user-id is removed
    localStorage.removeItem("kf-token");
    localStorage.removeItem("kf-user-id");
    localStorage.removeItem("kf-user-role");
    set({ token: "", userId: "", role: "", isAuthenticated: false, isLoading: false });
    window.location.href = "/login";
  },

  checkSession: () => {
    const token = localStorage.getItem("kf-token");
    const userId = localStorage.getItem("kf-user-id");
    const role = localStorage.getItem("kf-user-role");
    if (token && userId) {
      // Verify token is still valid
      apiFetch("/api/auth/me")
        .then((res) => {
          if (res.ok) {
            set({
              token,
              userId,
              role: (role as "admin" | "user") || "user",
              isAuthenticated: true,
              isLoading: false,
            });
          } else {
            localStorage.removeItem("kf-token");
            localStorage.removeItem("kf-user-id");
            localStorage.removeItem("kf-user-role");
            set({ token: "", userId: "", role: "", isAuthenticated: false, isLoading: false });
          }
        })
        .catch(() => {
          // Network error: keep the session rather than logging the user out.
          set({
            token,
            userId,
            role: (role as "admin" | "user") || "user",
            isAuthenticated: true,
            isLoading: false,
          });
        });
    } else {
      set({ isAuthenticated: false, isLoading: false });
    }
  },
}));
