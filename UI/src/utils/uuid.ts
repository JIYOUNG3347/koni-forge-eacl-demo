/**
 * Safe UUID generator with fallback for non-secure contexts.
 *
 * `crypto.randomUUID()` is only available in Secure Contexts (HTTPS or localhost).
 * When the app is accessed via HTTP + IP (e.g., http://192.168.1.100:8080),
 * browsers like Safari block this API. This module provides a fallback using
 * Math.random for such environments.
 *
 * Security note: The Math.random fallback is NOT cryptographically secure.
 * It is suitable for client-side identifiers (chat thread IDs, message IDs)
 * but MUST NOT be used for security tokens, session IDs, or anything that
 * requires unpredictability.
 */

export function safeRandomUUID(): string {
  // Prefer native crypto.randomUUID when available
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    try {
      return crypto.randomUUID();
    } catch {
      // Fall through to fallback (some browsers throw in non-secure context)
    }
  }

  // Fallback: RFC 4122 version 4 UUID using Math.random
  // Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
