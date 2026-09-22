import { useCallback } from "react";

/**
 * Single-language build. The source strings are already English, so translation
 * is the identity and only the `{0}` interpolation does real work.
 *
 * The `t` / `tp` seam is kept so a second language can be added by swapping in
 * a dictionary here, without touching the 400+ call sites.
 */

export type Lang = "en";

export function isLang(value: unknown): value is Lang {
  return value === "en";
}

/** Current UI language, for non-React callers. */
export function getLang(): Lang {
  return "en";
}

/** Translate a source string. */
export function translate(key: string, _lang: Lang = "en"): string {
  return key;
}

export function useT(): (key: string) => string {
  return useCallback((key: string) => key, []);
}

/** Non-reactive translate, for module-level constants and non-React code. */
export function t(key: string): string {
  return key;
}

export function formatArgs(text: string, args: readonly unknown[]): string {
  return text.replace(/\{(\d+)\}/g, (whole, idx: string) => {
    const arg = args[Number(idx)];
    return arg === undefined || arg === null ? whole : String(arg);
  });
}

/**
 * Hook form of the parameterised translate.
 * `useTp()("{0} remaining", n)` renders `"3 remaining"`.
 */
export function useTp(): (key: string, ...args: unknown[]) => string {
  return useCallback((key: string, ...args: unknown[]) => formatArgs(key, args), []);
}

/** Non-reactive parameterised translate, for module-level helpers. */
export function tp(key: string, ...args: unknown[]): string {
  return formatArgs(key, args);
}
