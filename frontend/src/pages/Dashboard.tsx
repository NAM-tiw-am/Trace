import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Folder, Users, Video as VideoIcon, Eye, CheckCircle, XCircle, Search as SearchIcon, Camera, Clock, ChevronRight } from 'lucide-react';
import { api, mediaUrl, type Sighting, type MatchStatus } from '../lib/api';

interface Props { tab: 'overview' | 'cases' | 'persons' | 'footage' | 'sightings'; }

// ── Score badge ─────────────────────────────────────────────────────────────
function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.55 ? 'bg-green-500' : score >= 0.35 ? 'bg-amber-500' : 'bg-slate-400';
  return (
    <span className={`${color} text-white text-xs font-bold px-2 py-0.5 rounded-full flex items-center gap-1`}>
      ✓ {pct}%
    </span>
  );
}

// ── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: MatchStatus }) {
  const map: Record<MatchStatus, string> = {
    POTENTIAL_MATCH: 'bg-amber-100 text-amber-800 border border-amber-300',
    REVIEWED: 'bg-blue-100 text-blue-800 border border-blue-300',
    CONFIRMED: 'bg-green-100 text-green-800 border border-green-300',
    REJECTED: 'bg-red-100 text-red-800 border border-red-300',
  };
  return (
    <span className={`${map[status]} text-xs font-semibold px-2 py-0.5 rounded`}>
      {status.replace('_', ' ')}
    </span>
  );
}

// ── Sighting card ────────────────────────────────────────────────────────────
function SightingCard({ sighting, onStatusChange }: { sighting: Sighting; onStatusChange?: () => void }) {
  const qc = useQueryClient();
  const [notes, setNotes] = useState(sighting.notes ?? '');

  const mutate = useMutation({
    mutationFn: ({ status, n }: { status: MatchStatus; n: string }) =>
      api.sightings.updateStatus(sighting.id, status, n),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sightings'] });
      onStatusChange?.();
    },
  });

  const snap = sighting.snapshot_path ? mediaUrl(sighting.snapshot_path) : null;
  const ts = sighting.timestamp_formatted ?? `${Math.floor(sighting.timestamp_seconds / 3600).toString().padStart(2,'0')}:${Math.floor((sighting.timestamp_seconds % 3600)/60).toString().padStart(2,'0')}:${Math.floor(sighting.timestamp_seconds % 60).toString().padStart(2,'0')}`;

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* Snapshot */}
      <div className="relative aspect-video bg-slate-800 overflow-hidden">
        {snap
          ? <img src={snap} alt="Sighting snapshot" className="w-full h-full object-cover" />
          : <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm">No snapshot</div>
        }
        {/* Score overlay */}
        <div className="absolute top-2 right-2">
          <ScoreBadge score={sighting.similarity_score} />
        </div>
        {/* Timestamp overlay */}
        <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded font-mono">
          {ts}
        </div>
        {/* Scan bracket corners */}
        <div className="absolute inset-3 pointer-events-none">
          <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-blue-400 rounded-tl" />
          <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-blue-400 rounded-tr" />
          <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-blue-400 rounded-bl" />
          <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-blue-400 rounded-br" />
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          <span className="font-semibold text-slate-900">{sighting.person?.name ?? `Person #${sighting.person_id}`}</span>
          <StatusBadge status={sighting.match_status} />
        </div>
        <div className="text-xs text-slate-500 space-y-0.5 mb-3">
          <div className="flex items-center gap-1.5">
            <Camera size={11} />
            {sighting.video?.camera_name ?? `Video #${sighting.video_id}`}
            {sighting.video?.location ? ` — ${sighting.video.location}` : ''}
          </div>
          <div className="flex items-center gap-1.5">
            <Clock size={11} />
            Frame {sighting.frame_number} ({sighting.timestamp_seconds.toFixed(1)}s)
          </div>
        </div>

        {/* Notes */}
        <textarea
          rows={2}
          placeholder="Add notes…"
          value={notes}
          onChange={e => setNotes(e.target.value)}
          className="w-full text-xs border border-slate-200 rounded-lg px-2 py-1.5 resize-none mb-3 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />

        {/* Action buttons */}
        <div className="flex gap-2">
          <Link
            to={`/results/${sighting.video_id}/${sighting.person_id}`}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors text-slate-600"
          >
            <SearchIcon size={12} /> Inspect
          </Link>
          <button
            onClick={() => mutate.mutate({ status: 'CONFIRMED', n: notes })}
            disabled={mutate.isPending || sighting.match_status === 'CONFIRMED'}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-green-500 hover:bg-green-600 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            <CheckCircle size={12} /> Confirm
          </button>
          <button
            onClick={() => mutate.mutate({ status: 'REJECTED', n: notes })}
            disabled={mutate.isPending || sighting.match_status === 'REJECTED'}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-red-500 hover:bg-red-600 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            <XCircle size={12} /> Reject
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Stat tile ────────────────────────────────────────────────────────────────
function StatTile({ label, value, icon: Icon, color }: { label: string; value: number | string; icon: React.ElementType; color: string }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 flex items-center gap-4 shadow-sm">
      <div className={`${color} rounded-xl p-3`}>
        <Icon size={22} className="text-white" />
      </div>
      <div>
        <div className="text-xs text-slate-500 uppercase tracking-wide font-medium">{label}</div>
        <div className="text-3xl font-bold text-slate-900 leading-tight">{value}</div>
      </div>
    </div>
  );
}

// ── Overview tab ────────────────────────────────────────────────────────────
function Overview() {
  const { data: cases } = useQuery({ queryKey: ['cases'], queryFn: api.cases.list });
  const { data: persons } = useQuery({ queryKey: ['persons'], queryFn: () => api.persons.list() });
  const { data: videos } = useQuery({ queryKey: ['videos'], queryFn: api.videos.list });
  const { data: pending } = useQuery({
    queryKey: ['sightings', 'POTENTIAL_MATCH'],
    queryFn: () => api.sightings.list({ match_status: 'POTENTIAL_MATCH' }),
    refetchInterval: 5000,
  });

  const activeCases = cases?.filter(c => c.status !== 'CLOSED').length ?? 0;

  return (
    <div className="p-8 space-y-8">
      {/* Stats */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatTile label="Active Cases" value={activeCases} icon={Folder} color="bg-indigo-500" />
        <StatTile label="Registered Persons" value={persons?.length ?? 0} icon={Users} color="bg-purple-500" />
        <StatTile label="CCTV Feeds Processed" value={videos?.filter(v => v.status === 'COMPLETED').length ?? 0} icon={VideoIcon} color="bg-rose-500" />
        <StatTile label="Potential Sightings" value={pending?.length ?? 0} icon={Eye} color="bg-teal-500" />
      </div>

      {/* Latest sightings */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
            <Eye size={18} className="text-blue-500" /> Latest CCTV Sighting Events
          </h2>
          <Link to="/dashboard/sightings" className="text-sm text-blue-600 hover:underline flex items-center gap-1">
            View All Sightings <ChevronRight size={14} />
          </Link>
        </div>

        {pending && pending.length > 0 ? (
          <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
            {pending.slice(0, 6).map(s => <SightingCard key={s.id} sighting={s} />)}
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-slate-400">
            <Eye size={32} className="mx-auto mb-3 opacity-40" />
            <p>No pending sightings. Run a search to detect matches.</p>
            <Link to="/search" className="mt-4 inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
              <SearchIcon size={14} /> Start a Search
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sightings tab ────────────────────────────────────────────────────────────
function SightingsTab() {
  const [filter, setFilter] = useState<string>('POTENTIAL_MATCH');
  const { data, isLoading } = useQuery({
    queryKey: ['sightings', filter],
    queryFn: () => api.sightings.list(filter ? { match_status: filter } : undefined),
    refetchInterval: 5000,
  });

  const sorted = data ? [...data].sort((a, b) => b.similarity_score - a.similarity_score) : [];

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Sightings & Matches</h2>
        <select
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="">All</option>
          <option value="POTENTIAL_MATCH">Potential Match</option>
          <option value="CONFIRMED">Confirmed</option>
          <option value="REVIEWED">Reviewed</option>
          <option value="REJECTED">Rejected</option>
        </select>
      </div>

      {isLoading ? (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="bg-slate-100 rounded-xl aspect-square animate-pulse" />
          ))}
        </div>
      ) : sorted.length > 0 ? (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {sorted.map(s => <SightingCard key={s.id} sighting={s} />)}
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-slate-400">
          No sightings found for this filter.
        </div>
      )}
    </div>
  );
}

// ── Cases tab ────────────────────────────────────────────────────────────────
function CasesTab() {
  const qc = useQueryClient();
  const { data: cases, isLoading } = useQuery({ queryKey: ['cases'], queryFn: api.cases.list });
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState('');

  const create = useMutation({
    mutationFn: () => {
      const ts = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
      return api.cases.create({ case_number: `CASE-${ts}`, title });
    },
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['cases'] }); setCreating(false); setTitle(''); },
  });

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Cases</h2>
        <button onClick={() => setCreating(v => !v)} className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
          + New Case
        </button>
      </div>
      {creating && (
        <div className="bg-white border border-slate-200 rounded-xl p-4 flex gap-3">
          <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Case title…"
            className="flex-1 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
          <button onClick={() => create.mutate()} disabled={create.isPending}
            className="bg-green-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-green-600 disabled:opacity-50">
            Create
          </button>
        </div>
      )}
      {isLoading ? (
        <div className="space-y-3">{Array.from({ length: 4 }).map((_, i) => <div key={i} className="bg-slate-100 h-16 rounded-xl animate-pulse" />)}</div>
      ) : (
        <div className="space-y-3">
          {cases?.map(c => (
            <div key={c.id} className="bg-white border border-slate-200 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div className="font-semibold">{c.title || c.case_number}</div>
                <div className="text-xs text-slate-500">{c.case_number} · {new Date(c.created_at).toLocaleDateString()}</div>
              </div>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${c.status === 'CLOSED' ? 'bg-slate-100 text-slate-500' : 'bg-green-100 text-green-700'}`}>
                {c.status ?? 'OPEN'}
              </span>
            </div>
          ))}
          {cases?.length === 0 && <div className="text-center text-slate-400 py-12">No cases yet.</div>}
        </div>
      )}
    </div>
  );
}

// ── Persons tab ────────────────────────────────────────────────────────────
function PersonsTab() {
  const { data: persons, isLoading } = useQuery({ queryKey: ['persons'], queryFn: () => api.persons.list() });

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Missing Persons</h2>
        <Link to="/search" className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
          + Add Person
        </Link>
      </div>
      {isLoading ? (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <div key={i} className="bg-slate-100 h-40 rounded-xl animate-pulse" />)}
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {persons?.map(p => {
            const hasEmbedding = p.embeddings?.length > 0;
            const photo = p.photos?.[0]?.file_path;
            return (
              <div key={p.id} className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                <div className="aspect-video bg-slate-100 overflow-hidden">
                  {photo
                    ? <img src={mediaUrl(photo)} alt={p.name} className="w-full h-full object-cover" />
                    : <div className="w-full h-full flex items-center justify-center text-slate-300">
                        <Users size={40} />
                      </div>
                  }
                </div>
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold">{p.name}</span>
                    {hasEmbedding
                      ? <span className="bg-green-100 text-green-700 text-xs px-1.5 py-0.5 rounded font-medium">Ready</span>
                      : <span className="bg-red-100 text-red-600 text-xs px-1.5 py-0.5 rounded font-medium">No face</span>
                    }
                  </div>
                  <div className="text-xs text-slate-500">
                    {p.age && `Age: ${p.age} · `}{p.gender && `${p.gender} · `}{p.last_known_location || 'Location unknown'}
                  </div>
                </div>
              </div>
            );
          })}
          {persons?.length === 0 && <div className="col-span-3 text-center text-slate-400 py-12">No persons registered yet.</div>}
        </div>
      )}
    </div>
  );
}

// ── Footage tab ────────────────────────────────────────────────────────────
function FootageTab() {
  const qc = useQueryClient();
  const { data: videos, isLoading } = useQuery({ queryKey: ['videos'], queryFn: api.videos.list, refetchInterval: 5000 });

  const scan = useMutation({
    mutationFn: (id: number) => api.videos.process(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['videos'] }),
  });

  const statusColor: Record<string, string> = {
    UPLOADED: 'bg-slate-100 text-slate-600',
    PROCESSING: 'bg-blue-100 text-blue-700',
    COMPLETED: 'bg-green-100 text-green-700',
    FAILED: 'bg-red-100 text-red-700',
  };

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">CCTV Footage</h2>
        <Link to="/search" className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
          + Upload Footage
        </Link>
      </div>
      {isLoading ? (
        <div className="space-y-3">{Array.from({ length: 4 }).map((_, i) => <div key={i} className="bg-slate-100 h-20 rounded-xl animate-pulse" />)}</div>
      ) : (
        <div className="space-y-3">
          {videos?.map(v => (
            <div key={v.id} className="bg-white border border-slate-200 rounded-xl p-4 flex items-center justify-between">
              <div>
                <div className="font-semibold">{v.camera_name}</div>
                <div className="text-xs text-slate-500">{v.location || 'No location'} · {new Date(v.created_at).toLocaleString()}</div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor[v.status] ?? statusColor.UPLOADED}`}>
                  {v.status}
                </span>
                {v.status !== 'PROCESSING' && (
                  <button
                    onClick={() => {
                      if (v.status === 'COMPLETED' && !confirm('This video was already scanned. Re-scanning will create duplicate sightings. Continue?')) return;
                      scan.mutate(v.id);
                    }}
                    disabled={scan.isPending}
                    className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50">
                    {v.status === 'COMPLETED' ? 'Re-scan' : 'Scan'}
                  </button>
                )}
              </div>
            </div>
          ))}
          {videos?.length === 0 && <div className="text-center text-slate-400 py-12">No footage uploaded yet.</div>}
        </div>
      )}
    </div>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────
export default function Dashboard({ tab }: Props) {
  return (
    <div className="min-h-screen bg-slate-50">
      {tab === 'overview' && <Overview />}
      {tab === 'sightings' && <SightingsTab />}
      {tab === 'cases' && <CasesTab />}
      {tab === 'persons' && <PersonsTab />}
      {tab === 'footage' && <FootageTab />}
    </div>
  );
}
