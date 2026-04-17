const DEFAULT_LOCAL_API_BASE_URL = "http://localhost:8000";

const normalizeBaseUrl = (value: string) => value.trim().replace(/\/+$/, "");

const uniqueStrings = (values: string[]) => Array.from(new Set(values));

const envPrimaryBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
const envFallbackBaseUrl = import.meta.env.VITE_API_FALLBACK_BASE_URL as string | undefined;
const envTimeoutMs = import.meta.env.VITE_API_TIMEOUT_MS as string | undefined;

export const API_BASE_URLS = uniqueStrings(
  [envPrimaryBaseUrl, envFallbackBaseUrl, DEFAULT_LOCAL_API_BASE_URL]
    .filter((value): value is string => Boolean(value && value.trim()))
    .map(normalizeBaseUrl),
);

export const API_BASE_URL = API_BASE_URLS[0] ?? DEFAULT_LOCAL_API_BASE_URL;

const joinUrl = (baseUrl: string, path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${normalizedPath}`;
};

const DEFAULT_TIMEOUT_MS_REMOTE = 2500;
const DEFAULT_TIMEOUT_MS_LOCAL = 100000;
const LONG_RUNNING_TIMEOUT_MS_LOCAL = 20 * 60 * 1000;

const isLocalBaseUrl = (baseUrl: string) =>
  baseUrl.includes("localhost") || baseUrl.includes("127.0.0.1");

const getTimeoutMs = (baseUrl: string) => {
  const fromEnv = envTimeoutMs ? Number(envTimeoutMs) : NaN;
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv;
  return isLocalBaseUrl(baseUrl) ? DEFAULT_TIMEOUT_MS_LOCAL : DEFAULT_TIMEOUT_MS_REMOTE;
};

const getRequestTimeoutMs = (baseUrl: string, path: string) => {
  if (!isLocalBaseUrl(baseUrl)) return getTimeoutMs(baseUrl);

  const normalizedPath = path.toLowerCase();
  const isLongRunningEndpoint =
    normalizedPath.startsWith("/query") ||
    normalizedPath.startsWith("/debug/query") ||
    normalizedPath.startsWith("/upload") ||
    normalizedPath.startsWith("/reset");

  return isLongRunningEndpoint ? LONG_RUNNING_TIMEOUT_MS_LOCAL : getTimeoutMs(baseUrl);
};

const fetchWithTimeout = async (url: string, init: RequestInit | undefined, timeoutMs: number) => {
  // If caller already provides a signal, respect it and do not override with our own timeout.
  if (init?.signal) return fetch(url, init);

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
};

// Try primary base URL first; if it looks unreachable/misconfigured, fall back to localhost:8000.
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  let lastError: unknown = undefined;

  for (let i = 0; i < API_BASE_URLS.length; i += 1) {
    const baseUrl = API_BASE_URLS[i]!;
    const url = joinUrl(baseUrl, path);

    try {
      const res = await fetchWithTimeout(url, init, getRequestTimeoutMs(baseUrl, path));

      if (res.ok) return res;

      const hasFallback = i < API_BASE_URLS.length - 1;
      const shouldFallback = res.status === 404 || res.status >= 500;

      if (hasFallback && shouldFallback) continue;

      return res;
    } catch (err) {
      lastError = err;
      const hasFallback = i < API_BASE_URLS.length - 1;
      if (hasFallback) continue;
      throw err;
    }
  }

  // Should not happen; defensive return.
  throw (
    lastError ??
    new Error("Unable to connect to the backend (tried remote and localhost:8000).")
  );
}
