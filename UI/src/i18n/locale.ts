import { getLang, type Lang } from "./index";

const LOCALES: Record<Lang, string> = {
  en: "en-US",
};

/** BCP 47 locale for the UI language. */
export function getLocale(): string {
  return LOCALES[getLang()] ?? LOCALES.en;
}

/** Component form of `getLocale`. */
export function useLocale(): string {
  return LOCALES[getLang()] ?? LOCALES.en;
}

/** Shown for an invalid timestamp. A blank would not distinguish absent from broken. */
export const INVALID_DATE = "—";

const NAIVE_ISO = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/;

export function parseServerDate(
  value: number | string | Date | null | undefined,
): Date | null {
  if (value === null || value === undefined || value === "") return null;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
  let raw: number | string = value;
  if (typeof raw === "string" && NAIVE_ISO.test(raw.trim())) {
    raw = raw.trim().replace(" ", "T") + "Z";
  }
  const d = new Date(raw);
  return Number.isNaN(d.getTime()) ? null : d;
}

function toDate(value: number | string | Date | null | undefined): Date | null {
  return parseServerDate(value);
}

/** Date and time formatting. Only the locale follows the UI language; options come from the caller. */
export function formatDateTime(
  value: number | string | Date | null | undefined,
  options?: Intl.DateTimeFormatOptions,
  locale?: string,
): string {
  const d = toDate(value);
  if (!d) return INVALID_DATE;
  try {
    return d.toLocaleString(locale ?? getLocale(), options);
  } catch {
    return d.toISOString();
  }
}

/** Date only. */
export function formatDate(
  value: number | string | Date | null | undefined,
  options?: Intl.DateTimeFormatOptions,
  locale?: string,
): string {
  const d = toDate(value);
  if (!d) return INVALID_DATE;
  try {
    return d.toLocaleDateString(locale ?? getLocale(), options);
  } catch {
    return d.toISOString().slice(0, 10);
  }
}

/** Time only. */
export function formatTime(
  value: number | string | Date | null | undefined,
  options?: Intl.DateTimeFormatOptions,
  locale?: string,
): string {
  const d = toDate(value);
  if (!d) return INVALID_DATE;
  try {
    return d.toLocaleTimeString(locale ?? getLocale(), options);
  } catch {
    return d.toISOString().slice(11, 19);
  }
}

/** Number formatting with thousands separators. */
export function formatNumber(value: number | null | undefined, locale?: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return INVALID_DATE;
  try {
    return value.toLocaleString(locale ?? getLocale());
  } catch {
    return String(value);
  }
}
