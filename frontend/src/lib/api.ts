// Central API base URL — reads from VITE_API_BASE_URL at build time.
// In dev, this is empty so Vite's proxy handles /api and /storage.
// In production on Vercel, set VITE_API_BASE_URL to your Railway backend URL,
// e.g. https://trace-backend.up.railway.app
export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

export function apiUrl(path: string): string {
  // path should start with /api/... or /storage/...
  return `${API_BASE}${path}`;
}
