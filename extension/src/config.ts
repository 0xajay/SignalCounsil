/**
 * SignalCouncil API Configuration
 *
 * Sourced from Vite environment variables (VITE_API_URL) with fallback
 * to runtime chrome.storage / localStorage override so users can configure
 * their deployed Cloud Run URL directly in the extension.
 */

export const DEFAULT_API_URL = (
  import.meta.env.VITE_API_URL || 'http://localhost:8080'
).replace(/\/+$/, '');

const STORAGE_KEY = 'signal_council_api_url';

/**
 * Clean and normalize a base URL (strips trailing slashes).
 */
export function normalizeUrl(url: string): string {
  if (!url) return '';
  return url.trim().replace(/\/+$/, '');
}

/**
 * Retrieves the active API base URL.
 * Checks chrome.storage first, then falls back to VITE_API_URL / localhost.
 */
export async function getApiBaseUrl(): Promise<string> {
  try {
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
      const result = await chrome.storage.sync.get([STORAGE_KEY]) as Record<string, string>;
      if (result && result[STORAGE_KEY]) {
        return normalizeUrl(String(result[STORAGE_KEY]));
      }
    }
  } catch (err) {
    console.warn('Could not read from chrome.storage.sync, checking localStorage fallback:', err);
  }

  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const local = window.localStorage.getItem(STORAGE_KEY);
      if (local) {
        return normalizeUrl(local);
      }
    }
  } catch (err) {
    console.warn('Could not read from localStorage:', err);
  }

  return DEFAULT_API_URL;
}

/**
 * Persists an API base URL override to chrome.storage and localStorage.
 */
export async function setApiBaseUrl(url: string): Promise<void> {
  const cleaned = normalizeUrl(url);

  try {
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
      await chrome.storage.sync.set({ [STORAGE_KEY]: cleaned });
    }
  } catch (err) {
    console.warn('Could not persist to chrome.storage.sync:', err);
  }

  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      if (cleaned) {
        window.localStorage.setItem(STORAGE_KEY, cleaned);
      } else {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    }
  } catch (err) {
    console.warn('Could not persist to localStorage:', err);
  }
}
