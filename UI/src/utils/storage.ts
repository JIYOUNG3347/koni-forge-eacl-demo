/**
 * User-scoped localStorage utilities.
 *
 * To prevent data leakage between users on the same browser
 * (e.g., shared workstation), all user data localStorage keys
 * must be prefixed with the current user ID.
 *
 * Format: "{baseKey}:{userId}"
 *
 * Auth keys (kf-token, kf-user-id, kf-user-role) are NOT scoped
 * because they identify the current user.
 */

import type { StateStorage } from "zustand/middleware";

const KEY_SEPARATOR = ":";

const AUTH_USER_ID_KEY = "kf-user-id";

/**
 * Keys from the legacy (pre-scoping) implementation.
 * Cleaned up on app startup to prevent leakage.
 */
const LEGACY_KEYS = [
  "kf-agent-threads",
  "kf-agent-messages",
  "kf-pipeline-events",
  "koni_chat_v1",
  "koni-eval-state",
] as const;

export function getCurrentUserId(): string | null {
  try {
    return localStorage.getItem(AUTH_USER_ID_KEY);
  } catch {
    return null;
  }
}

/**
 * Generate a user-scoped storage key.
 * Returns null if no user is logged in.
 *
 * @example userKey("kf-agent-threads") // "kf-agent-threads:alice"
 */
export function userKey(baseKey: string): string | null {
  const userId = getCurrentUserId();
  if (!userId) return null;
  return `${baseKey}${KEY_SEPARATOR}${userId}`;
}

export function getUserItem(baseKey: string): string | null {
  const key = userKey(baseKey);
  if (!key) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function setUserItem(baseKey: string, value: string): void {
  const key = userKey(baseKey);
  if (!key) return;
  try {
    localStorage.setItem(key, value);
  } catch {
    // Quota exceeded or storage disabled — fail silently
  }
}

export function removeUserItem(baseKey: string): void {
  const key = userKey(baseKey);
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

/**
 * Clear all localStorage entries belonging to the current user.
 * Must be called BEFORE removing kf-user-id from localStorage.
 */
export function clearCurrentUserStorage(): void {
  const userId = getCurrentUserId();
  if (!userId) return;

  const suffix = `${KEY_SEPARATOR}${userId}`;
  const keysToRemove: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.endsWith(suffix)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((k) => localStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}

/**
 * One-time cleanup of legacy (pre-scoping) keys.
 * Call on app startup to remove insecure shared keys
 * written before this PR.
 */
export function cleanupLegacyKeys(): void {
  try {
    LEGACY_KEYS.forEach((key) => {
      if (localStorage.getItem(key) !== null) {
        localStorage.removeItem(key);
      }
    });
  } catch {
    /* ignore */
  }
}

/**
 * Zustand-compatible StateStorage for user-scoped persistence.
 *
 * @example
 *   persist(fn, { name: "kf-state", storage: createJSONStorage(() => userScopedStorage) })
 */
export const userScopedStorage: StateStorage = {
  getItem: (name) => getUserItem(name),
  setItem: (name, value) => setUserItem(name, value),
  removeItem: (name) => removeUserItem(name),
};
