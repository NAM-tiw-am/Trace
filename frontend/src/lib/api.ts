// Central API helper — reads VITE_API_BASE_URL at build time.
// Dev: empty string (Vite proxy handles /api and /storage)
// Vercel: set to Railway backend URL
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function mediaUrl(rawPath: string): string {
  if (!rawPath) return '';
  // Normalise Windows backslashes
  const normalised = rawPath.replace(/\\/g, '/');
  // Find 'storage/' and return from there
  const idx = normalised.indexOf('storage/');
  if (idx !== -1) return apiUrl('/' + normalised.slice(idx));
  // Already absolute URL or /storage/... path
  if (normalised.startsWith('http') || normalised.startsWith('/storage')) return apiUrl(normalised);
  return apiUrl('/storage/' + normalised);
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  model: string;
  database: { engine: string };
  thresholds: { match: number; high_confidence: number; cooldown_seconds: number };
}

export interface Case {
  id: number;
  case_number: string;
  title?: string;
  description?: string;
  status?: string;
  created_at: string;
}

export interface PersonPhoto {
  id: number;
  file_path: string;
}

export interface Embedding {
  id: number;
}

export interface Person {
  id: number;
  name: string;
  age?: number;
  gender?: string;
  last_known_location?: string;
  description?: string;
  case_id: number;
  photos: PersonPhoto[];
  embeddings: Embedding[];
  created_at: string;
}

export interface Video {
  id: number;
  camera_name: string;
  location?: string;
  file_path: string;
  status: 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  case_id?: number;
  created_at: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
}

export interface Job {
  id: number;
  video_id: number;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress: number;
  current_frame?: number;
  total_frames?: number;
  faces_detected?: number;
  matches_found?: number;
  error_message?: string;
  created_at: string;
}

export type MatchStatus = 'POTENTIAL_MATCH' | 'REVIEWED' | 'CONFIRMED' | 'REJECTED';

export interface Sighting {
  id: number;
  video_id: number;
  person_id: number;
  similarity_score: number;
  timestamp_seconds: number;
  timestamp_formatted?: string;
  frame_number: number;
  bbox_x1?: number;
  bbox_y1?: number;
  bbox_x2?: number;
  bbox_y2?: number;
  snapshot_path?: string;
  match_status: MatchStatus;
  notes?: string;
  person?: Person;
  video?: Video;
  created_at: string;
}

// ── API calls ──────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'API error');
  }
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),

  // Cases
  cases: {
    list: () => request<Case[]>('/api/cases'),
    create: (data: { case_number: string; title?: string; description?: string }) =>
      request<Case>('/api/cases', { method: 'POST', body: JSON.stringify(data) }),
    get: (id: number) => request<Case>(`/api/cases/${id}`),
  },

  // Persons
  persons: {
    list: (caseId?: number) =>
      request<Person[]>(`/api/persons${caseId ? `?case_id=${caseId}` : ''}`),
    create: (data: { name: string; case_id: number; age?: number; gender?: string; last_known_location?: string; description?: string }) =>
      request<Person>('/api/persons', { method: 'POST', body: JSON.stringify(data) }),
    get: (id: number) => request<Person>(`/api/persons/${id}`),
    uploadPhoto: async (personId: number, file: File): Promise<PersonPhoto> => {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(apiUrl(`/api/persons/${personId}/photos`), {
        method: 'POST',
        body: fd,
      });
      const json = await res.json().catch(() => ({ detail: res.statusText }));
      if (!res.ok) throw Object.assign(new Error(json.detail ?? 'Upload failed'), { status: res.status });
      return json;
    },
    delete: (id: number) => request(`/api/persons/${id}`, { method: 'DELETE' }),
  },

  // Videos
  videos: {
    list: () => request<Video[]>('/api/videos'),
    get: (id: number) => request<Video>(`/api/videos/${id}`),
    upload: async (data: { camera_name: string; location?: string; case_id?: number; file: File }): Promise<Video> => {
      const fd = new FormData();
      fd.append('camera_name', data.camera_name);
      if (data.location) fd.append('location', data.location);
      if (data.case_id) fd.append('case_id', String(data.case_id));
      fd.append('file', data.file);
      const res = await fetch(apiUrl('/api/videos'), { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail ?? 'Upload failed'); }
      return res.json();
    },
    process: (id: number) =>
      fetch(apiUrl(`/api/videos/${id}/process`), { method: 'POST' }).then(async r => {
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail ?? 'Process failed');
        return j as Job;
      }),
    sightings: (id: number) => request<Sighting[]>(`/api/videos/${id}/sightings`),
  },

  // Jobs
  jobs: {
    get: (id: number) => request<Job>(`/api/jobs/${id}`),
  },

  // Sightings
  sightings: {
    list: (params?: { person_id?: number; video_id?: number; match_status?: string }) => {
      const qs = new URLSearchParams();
      if (params?.person_id) qs.set('person_id', String(params.person_id));
      if (params?.video_id) qs.set('video_id', String(params.video_id));
      if (params?.match_status) qs.set('match_status', params.match_status);
      return request<Sighting[]>(`/api/sightings?${qs}`);
    },
    get: (id: number) => request<Sighting>(`/api/sightings/${id}`),
    updateStatus: (id: number, match_status: MatchStatus, notes?: string) =>
      request<Sighting>(`/api/sightings/${id}/status`, {
        method: 'PUT',
        body: JSON.stringify({ match_status, notes: notes ?? '' }),
      }),
  },
};
