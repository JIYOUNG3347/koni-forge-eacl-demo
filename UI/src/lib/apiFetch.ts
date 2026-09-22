
import { getLang } from "../i18n";

const LANG_HEADER = "X-KONI-Lang";

const TOKEN_KEY = "kf-token";
const USER_ID_KEY = "kf-user-id";
const USER_ROLE_KEY = "kf-user-role";

function clearAuthTokens(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_ID_KEY);
  localStorage.removeItem(USER_ROLE_KEY);
}

function injectAuth(opts: RequestInit): RequestInit {
  const token = localStorage.getItem(TOKEN_KEY) ?? "";
  const headers: Record<string, string> = {
    ...((opts.headers as Record<string, string>) ?? {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  // An explicit value from the caller wins. A failed language lookup must not
  // block the request, so the header is simply omitted and the server falls back.
  if (!headers[LANG_HEADER]) {
    try {
      headers[LANG_HEADER] = getLang();
    } catch {
      /* The request goes out even if the language is unknown. */
    }
  }
  return { ...opts, headers };
}

export async function apiFetch(
  url: string,
  opts: RequestInit = {},
): Promise<Response> {
  const authedOpts = injectAuth(opts);

  let response: Response;
  try {
    response = await fetch(url, authedOpts);
  } catch (networkError: unknown) {
    // Network failure (offline, DNS, CORS preflight, etc.) — retry once.
    // Only retry when the signal has NOT been aborted (user-initiated cancel).
    if (opts.signal?.aborted) {
      throw networkError;
    }
    response = await fetch(url, authedOpts);
  }

  // 401 outside auth endpoints → clear session, redirect to /login
  if (response.status === 401 && !url.includes("/auth/")) {
    clearAuthTokens();
    window.location.href = "/login";
  }

  return response;
}
