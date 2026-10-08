import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  CheckCircle, XCircle, AlertTriangle, Download, FileText, Archive,
  Video as VideoIcon, Camera, Clock, User, ChevronLeft, Eye,
} from 'lucide-react';
import { api, mediaUrl, type Sighting, type MatchStatus } from '../lib/api';

// ── Verdict logic ──────────────────────────────────────────────────────────
function getVerdict(sightings: Sighting[], matchT: number, highT: number) {
  const active = sightings.filter(s => s.match_status !== 'REJECTED');
  if (sightings.some(s => s.match_status === 'CONFIRMED'))
    return { label: 'CONFIRMED MATCH', color: 'green', icon: CheckCircle } as const;
  const best = active.length ? Math.max(...active.map(s => s.similarity_score)) : 0;
  if (best >= highT) return { label: 'MATCH FOUND: HIGH CONFIDENCE', color: 'green', icon: CheckCircle } as const;
  if (best >= matchT) return { label: 'POSSIBLE MATCH: REVIEW REQUIRED', color: 'amber', icon: AlertTriangle } as const;
  return { label: 'NO MATCH FOUND', color: 'red', icon: XCircle } as const;
}

// ── Score badge ─────────────────────────────────────────────────────────────
function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const cls = score >= 0.55 ? 'bg-green-100 text-green-700 border-green-300'
    : score >= 0.35 ? 'bg-amber-100 text-amber-700 border-amber-300'
    : 'bg-slate-100 text-slate-600 border-slate-200';
  return <span className={`${cls} border text-xs font-bold px-2 py-0.5 rounded-full`}>{pct}% match</span>;
}

// ── Detail drawer ─────────────────────────────────────────────────────────────
function SightingDrawer({ sighting, videoPath, personPhotoPath, onClose }: {
  sighting: Sighting; videoPath?: string; personPhotoPath?: string; onClose: () => void;
}) {
  const qc = useQueryClient();
  const [notes, setNotes] = useState(sighting.notes ?? '');
  const [status, setStatus] = useState<MatchStatus>(sighting.match_status);

  const mut = useMutation({
    mutationFn: (s: MatchStatus) => api.sightings.updateStatus(sighting.id, s, notes),
    onSuccess: (updated) => {
      setStatus(updated.match_status);
      qc.invalidateQueries({ queryKey: ['sightings'] });
    },
  });

  const snap = sighting.snapshot_path ? mediaUrl(sighting.snapshot_path) : null;
  const vidUrl = videoPath ? mediaUrl(videoPath) : null;
  const refUrl = personPhotoPath ? mediaUrl(personPhotoPath) : null;
  const seekTime = Math.max(0, sighting.timestamp_seconds - 1);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="p-6 border-b border-slate-100 flex items-center justify-between">
          <h2 className="text-lg font-bold">Sighting Detail — Frame {sighting.frame_number}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 text-xl">&times;</button>
        </div>

        <div className="p-6 grid md:grid-cols-2 gap-6">
          {/* Reference vs Snapshot */}
          <div>
            <div className="text-xs font-semibold text-slate-500 uppercase mb-2">Reference Photo</div>
            {refUrl
              ? <img src={refUrl} alt="Reference" className="w-full rounded-xl object-cover max-h-64" />
              : <div className="w-full h-40 bg-slate-100 rounded-xl flex items-center justify-center text-slate-400"><User size={32} /></div>
            }
          </div>
          <div>
            <div className="text-xs font-semibold text-slate-500 uppercase mb-2">CCTV Snapshot</div>
            {snap
              ? <img src={snap} alt="Snapshot" className="w-full rounded-xl object-cover max-h-64" />
              : <div className="w-full h-40 bg-slate-100 rounded-xl flex items-center justify-center text-slate-400">No snapshot</div>
            }
          </div>

          {/* Video player */}
          {vidUrl && (
            <div className="md:col-span-2">
              <div className="text-xs font-semibold text-slate-500 uppercase mb-2">Original Footage (seeks to sighting time)</div>
              <video
                src={vidUrl}
                controls
                className="w-full rounded-xl"
                onLoadedMetadata={e => { (e.target as HTMLVideoElement).currentTime = seekTime; }}
              />
            </div>
          )}

          {/* Score + info */}
          <div className="md:col-span-2 grid sm:grid-cols-3 gap-4 bg-slate-50 rounded-xl p-4">
            <div><div className="text-xs text-slate-500">Score</div><ScoreBadge score={sighting.similarity_score} /></div>
            <div><div className="text-xs text-slate-500">Timestamp</div><div className="font-mono font-medium">{sighting.timestamp_formatted ?? `${sighting.timestamp_seconds.toFixed(1)}s`}</div></div>
            <div><div className="text-xs text-slate-500">Status</div><div className="font-medium">{status}</div></div>
          </div>

          {/* Review */}
          <div className="md:col-span-2 space-y-3">
            <textarea rows={3} value={notes} onChange={e => setNotes(e.target.value)}
              placeholder="Reviewer notes…"
              className="w-full border border-slate-200 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500" />
            <div className="flex gap-3">
              <button onClick={() => mut.mutate('CONFIRMED')} disabled={mut.isPending}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-green-500 text-white rounded-xl font-semibold hover:bg-green-600 disabled:opacity-50 transition-colors">
                <CheckCircle size={16} /> Confirm Match
              </button>
              <button onClick={() => mut.mutate('REVIEWED')} disabled={mut.isPending}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-blue-500 text-white rounded-xl font-semibold hover:bg-blue-600 disabled:opacity-50 transition-colors">
                <Eye size={16} /> Mark Reviewed
              </button>
              <button onClick={() => mut.mutate('REJECTED')} disabled={mut.isPending}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-red-500 text-white rounded-xl font-semibold hover:bg-red-600 disabled:opacity-50 transition-colors">
                <XCircle size={16} /> Reject
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Download helpers ─────────────────────────────────────────────────────────
function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

async function buildReport(person: { name: string; [key: string]: unknown }, video: { camera_name: string; [key: string]: unknown }, sightings: Sighting[], health: { model: string; thresholds: { match: number; high_confidence: number; cooldown_seconds: number } }, verdict: string) {
  const { default: jsPDF } = await import('jspdf');
  const { default: autoTable } = await import('jspdf-autotable');
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  // Header
  doc.setFillColor(15, 23, 42); doc.rect(0, 0, 210, 28, 'F');
  doc.setTextColor(255, 255, 255); doc.setFontSize(18); doc.setFont('helvetica', 'bold');
  doc.text('TRACE — AI Surveillance Suite', 14, 12);
  doc.setFontSize(9); doc.setFont('helvetica', 'normal');
  doc.text(`Evidence Report · Generated: ${new Date().toLocaleString()}`, 14, 20);

  doc.setTextColor(0, 0, 0); doc.setFontSize(12); doc.setFont('helvetica', 'bold');
  doc.text('Subject & Case Details', 14, 36);
  autoTable(doc, {
    startY: 40, theme: 'grid', headStyles: { fillColor: [51, 65, 85] },
    head: [['Field', 'Value']],
    body: [
      ['Name', String(person.name)],
      ['Camera', String(video.camera_name)],
      ['Verdict', verdict],
      ['Model', health.model],
      ['Match Threshold', String(health.thresholds.match)],
      ['High Confidence Threshold', String(health.thresholds.high_confidence)],
      ['Cooldown (s)', String(health.thresholds.cooldown_seconds)],
    ],
  });

  // Sightings table
  const afterFirst = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 8;
  doc.setFontSize(12); doc.setFont('helvetica', 'bold');
  doc.text('Sightings', 14, afterFirst);
  autoTable(doc, {
    startY: afterFirst + 4, theme: 'striped', headStyles: { fillColor: [51, 65, 85] },
    head: [['#', 'Timestamp', 'Frame', 'Score', 'Status', 'Notes']],
    body: sightings.map((s, i) => [
      i + 1,
      s.timestamp_formatted ?? `${s.timestamp_seconds.toFixed(1)}s`,
      s.frame_number,
      `${Math.round(s.similarity_score * 100)}%`,
      s.match_status,
      s.notes ?? '',
    ]),
  });

  // Disclaimer
  const y2 = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 10;
  doc.setFontSize(8); doc.setFont('helvetica', 'italic'); doc.setTextColor(120, 120, 120);
  doc.text('AI-assisted suggestion. Not a positive identification. Requires human verification.', 14, y2);

  return doc.output('blob');
}

async function buildZip(person: { name: string; [key: string]: unknown }, video: { camera_name: string; [key: string]: unknown }, sightings: Sighting[], health: { model: string; thresholds: { match: number; high_confidence: number; cooldown_seconds: number } }, verdict: string) {
  const JSZip = (await import('jszip')).default;
  const zip = new JSZip();

  // PDF
  const pdfBlob = await buildReport(person, video, sightings, health, verdict);
  zip.file('report.pdf', pdfBlob);

  // JSON
  zip.file('report.json', JSON.stringify({ person, video, verdict, sightings }, null, 2));

  // CSV
  const csv = ['id,timestamp,frame,score,status,notes',
    ...sightings.map(s => `${s.id},${s.timestamp_seconds},${s.frame_number},${s.similarity_score},${s.match_status},"${(s.notes ?? '').replace(/"/g, '""')}"`)
  ].join('\n');
  zip.file('sightings.csv', csv);

  // Snapshots
  const snapsFolder = zip.folder('snapshots')!;
  await Promise.allSettled(sightings.map(async (s, i) => {
    if (!s.snapshot_path) return;
    try {
      const r = await fetch(mediaUrl(s.snapshot_path));
      if (r.ok) snapsFolder.file(`sighting_${i + 1}.jpg`, await r.blob());
    } catch { /* skip */ }
  }));

  const blob = await zip.generateAsync({ type: 'blob' });
  return blob;
}

// ── Main Results page ──────────────────────────────────────────────────────
export default function Results() {
  const { videoId, personId } = useParams<{ videoId: string; personId: string }>();
  const vid = Number(videoId); const pid = Number(personId);

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health });
  const { data: video } = useQuery({ queryKey: ['video', vid], queryFn: () => api.videos.get(vid), enabled: !!vid });
  const { data: person } = useQuery({ queryKey: ['person', pid], queryFn: () => api.persons.get(pid), enabled: !!pid });
  const { data: allSightings, refetch: refetchSightings } = useQuery({
    queryKey: ['sightings', vid, pid],
    queryFn: () => api.sightings.list({ video_id: vid, person_id: pid }),
    enabled: !!vid && !!pid,
    refetchInterval: 5000,
  });

  const [drawer, setDrawer] = useState<Sighting | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  const matchT = health?.thresholds.match ?? 0.35;
  const highT = health?.thresholds.high_confidence ?? 0.55;
  const sightings = allSightings ?? [];
  const verdict = getVerdict(sightings, matchT, highT);

  const verdictBg = { green: 'bg-green-50 border-green-200', amber: 'bg-amber-50 border-amber-200', red: 'bg-slate-50 border-slate-200' }[verdict.color];
  const verdictText = { green: 'text-green-800', amber: 'text-amber-800', red: 'text-slate-700' }[verdict.color];
  const verdictIcon = { green: 'bg-green-100 text-green-600', amber: 'bg-amber-100 text-amber-600', red: 'bg-slate-100 text-slate-500' }[verdict.color];

  const personPhoto = person?.photos?.[0]?.file_path;
  const videoFilePath = video?.file_path;

  const doDownloadPdf = async () => {
    if (!person || !video || !health) return;
    setDownloading('pdf');
    try {
      const blob = await buildReport(person as never, video as never, sightings, health, verdict.label);
      downloadBlob(blob as Blob, `trace-report-${pid}-${vid}.pdf`);
    } finally { setDownloading(null); }
  };

  const doDownloadZip = async () => {
    if (!person || !video || !health) return;
    setDownloading('zip');
    try {
      const blob = await buildZip(person as never, video as never, sightings, health, verdict.label);
      downloadBlob(blob, `trace-evidence-${pid}-${vid}.zip`);
    } finally { setDownloading(null); }
  };

  const doDownloadFootage = () => {
    if (!videoFilePath) return;
    const a = document.createElement('a');
    a.href = mediaUrl(videoFilePath);
    a.download = `footage-${vid}.mp4`;
    a.target = '_blank';
    a.click();
  };

  const doDownloadSnapshot = (s: Sighting) => {
    if (!s.snapshot_path) return;
    const a = document.createElement('a');
    a.href = mediaUrl(s.snapshot_path);
    a.download = `snapshot-${s.id}.jpg`;
    a.target = '_blank';
    a.click();
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Top bar */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-4">
        <Link to="/dashboard" className="text-slate-500 hover:text-slate-800 flex items-center gap-1 text-sm">
          <ChevronLeft size={16} /> Dashboard
        </Link>
        <span className="text-slate-300">/</span>
        <span className="text-slate-800 font-medium text-sm">Results — {person?.name ?? `Person #${pid}`}</span>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Verdict banner */}
        <div className={`${verdictBg} border rounded-2xl p-6 flex items-center gap-5`}>
          <div className={`${verdictIcon} p-4 rounded-full shrink-0`}>
            <verdict.icon size={28} />
          </div>
          <div className="flex-1">
            <h1 className={`text-xl font-bold ${verdictText}`}>{verdict.label}</h1>
            <p className="text-sm text-slate-500 mt-0.5">AI-assisted suggestion. Not a positive identification. Requires human verification.</p>
          </div>
          <div className="text-sm text-slate-500">
            {sightings.filter(s => s.match_status !== 'REJECTED').length} active sightings
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left — evidence */}
          <div className="lg:col-span-2 space-y-6">
            {/* Criteria */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
              <h2 className="font-bold mb-4">Matching Criteria</h2>
              <div className="space-y-3 text-sm">
                {[
                  { label: 'Model', value: health?.model ?? 'ArcFace' },
                  { label: 'Match Threshold', value: matchT, pass: sightings.some(s => s.similarity_score >= matchT) },
                  { label: 'High Confidence Threshold', value: highT, pass: sightings.some(s => s.similarity_score >= highT) },
                  { label: 'Cooldown (s)', value: health?.thresholds.cooldown_seconds ?? 5 },
                  { label: 'Face Embedding', value: person ? (person.embeddings?.length > 0 ? 'Stored ✓' : 'Missing ✗') : '—', pass: person?.embeddings?.length ? true : false },
                ].map(row => (
                  <div key={row.label} className="flex justify-between items-center py-2 border-b border-slate-50">
                    <span className="text-slate-500">{row.label}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{String(row.value)}</span>
                      {row.pass !== undefined && (
                        row.pass
                          ? <CheckCircle size={14} className="text-green-500" />
                          : <XCircle size={14} className="text-red-400" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence gallery */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
              <h2 className="font-bold mb-4">Evidence Gallery ({sightings.length} sightings)</h2>
              {sightings.length === 0 ? (
                <div className="text-center py-10 text-slate-400">No sightings found.</div>
              ) : (
                <div className="grid sm:grid-cols-2 gap-4">
                  {sightings.map(s => {
                    const snap = s.snapshot_path ? mediaUrl(s.snapshot_path) : null;
                    return (
                      <div key={s.id} className="border border-slate-200 rounded-xl overflow-hidden hover:border-blue-400 transition-colors">
                        <div
                          className="aspect-video bg-slate-800 cursor-pointer relative overflow-hidden"
                          onClick={() => setDrawer(s)}
                        >
                          {snap
                            ? <img src={snap} alt="Sighting" className="w-full h-full object-cover" />
                            : <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm">No snapshot</div>
                          }
                          <div className="absolute top-2 right-2"><ScoreBadge score={s.similarity_score} /></div>
                          <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded font-mono">
                            {s.timestamp_formatted ?? `${s.timestamp_seconds.toFixed(1)}s`}
                          </div>
                        </div>
                        <div className="p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-slate-500">Frame {s.frame_number}</span>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                              s.match_status === 'CONFIRMED' ? 'bg-green-100 text-green-700'
                              : s.match_status === 'REJECTED' ? 'bg-red-100 text-red-700'
                              : s.match_status === 'REVIEWED' ? 'bg-blue-100 text-blue-700'
                              : 'bg-amber-100 text-amber-700'
                            }`}>{s.match_status}</span>
                          </div>
                          <div className="flex gap-2">
                            <button onClick={() => setDrawer(s)}
                              className="flex-1 text-xs py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors text-slate-600">
                              Inspect
                            </button>
                            <button onClick={() => doDownloadSnapshot(s)}
                              className="text-xs px-2 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors text-slate-600">
                              <Download size={12} />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Right — details + downloads */}
          <div className="space-y-5">
            {/* Person card */}
            <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
              {personPhoto
                ? <img src={mediaUrl(personPhoto)} alt={person?.name} className="w-full aspect-video object-cover" />
                : <div className="aspect-video bg-slate-100 flex items-center justify-center text-slate-300"><User size={40} /></div>
              }
              <div className="p-4">
                <div className="font-bold text-lg">{person?.name ?? '—'}</div>
                <div className="text-sm text-slate-500 space-y-1 mt-1">
                  {person?.age && <div>Age: {person.age}</div>}
                  {person?.gender && <div>Gender: {person.gender}</div>}
                  {person?.last_known_location && <div className="flex items-center gap-1"><Camera size={12} />{person.last_known_location}</div>}
                  {person?.description && <div className="text-xs text-slate-400 mt-2">{person.description}</div>}
                </div>
              </div>
            </div>

            {/* Video card */}
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
              <div className="font-semibold mb-2 flex items-center gap-2"><VideoIcon size={16} className="text-purple-500" /> Footage Details</div>
              <div className="text-sm text-slate-500 space-y-1">
                <div className="flex items-center gap-1.5"><Camera size={12} />{video?.camera_name ?? '—'}</div>
                {video?.location && <div className="flex items-center gap-1.5"><Clock size={12} />{video.location}</div>}
                <div>Status: <span className="font-medium text-slate-700">{video?.status ?? '—'}</span></div>
                {video?.created_at && <div>Scanned: {new Date(video.created_at).toLocaleString()}</div>}
              </div>
            </div>

            {/* Downloads */}
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
              <div className="font-semibold mb-3">Downloads</div>
              <div className="space-y-2">
                <button onClick={doDownloadPdf} disabled={downloading === 'pdf' || !person || !video}
                  className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 disabled:opacity-50 transition-colors text-left">
                  <div className="flex items-center gap-3">
                    <FileText size={18} className="text-red-500 shrink-0" />
                    <div>
                      <div className="text-sm font-medium">Evidence Report (PDF)</div>
                      <div className="text-xs text-slate-400">Trace-branded report with all sightings</div>
                    </div>
                  </div>
                  {downloading === 'pdf' ? <span className="text-xs text-blue-500">…</span> : <Download size={14} className="text-slate-400 shrink-0" />}
                </button>

                <button onClick={doDownloadZip} disabled={downloading === 'zip' || !person || !video}
                  className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 disabled:opacity-50 transition-colors text-left">
                  <div className="flex items-center gap-3">
                    <Archive size={18} className="text-amber-500 shrink-0" />
                    <div>
                      <div className="text-sm font-medium">Evidence Bundle (ZIP)</div>
                      <div className="text-xs text-slate-400">PDF + JSON + CSV + all snapshots</div>
                    </div>
                  </div>
                  {downloading === 'zip' ? <span className="text-xs text-blue-500">…</span> : <Download size={14} className="text-slate-400 shrink-0" />}
                </button>

                <button onClick={doDownloadFootage} disabled={!videoFilePath}
                  className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 disabled:opacity-50 transition-colors text-left">
                  <div className="flex items-center gap-3">
                    <VideoIcon size={18} className="text-purple-500 shrink-0" />
                    <div>
                      <div className="text-sm font-medium">Original Footage</div>
                      <div className="text-xs text-slate-400">Unmodified CCTV recording</div>
                    </div>
                  </div>
                  <Download size={14} className="text-slate-400 shrink-0" />
                </button>
              </div>
            </div>

            {/* Disclaimer */}
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex gap-3 text-sm text-amber-700">
              <AlertTriangle size={16} className="shrink-0 mt-0.5" />
              <p>Results must be verified by a certified human investigator before use in any official capacity.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Detail drawer */}
      {drawer && (
        <SightingDrawer
          sighting={drawer}
          videoPath={videoFilePath}
          personPhotoPath={personPhoto}
          onClose={() => { setDrawer(null); refetchSightings(); }}
        />
      )}
    </div>
  );
}
