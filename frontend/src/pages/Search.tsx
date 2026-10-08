import { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Upload, FileVideo, FileImage, Play, CheckCircle2, AlertCircle, Loader2, X, ChevronLeft } from 'lucide-react';
import { api } from '../lib/api';

type Stage = 'idle' | 'creating_case' | 'creating_person' | 'uploading_photo' | 'uploading_video' | 'scanning' | 'done' | 'error';

interface StageInfo {
  label: string;
  doneAt: number; // which step index completes this
}

const STAGES: StageInfo[] = [
  { label: 'Reference face detected', doneAt: 3 },
  { label: 'Footage uploaded', doneAt: 4 },
  { label: 'Scanning footage', doneAt: 5 },
  { label: 'Match evaluation', doneAt: 6 },
];

export default function Search() {
  const navigate = useNavigate();

  // Form state
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [personName, setPersonName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [location, setLocation] = useState('');
  const [description, setDescription] = useState('');
  const [cameraName, setCameraName] = useState('');
  const [cameraLocation, setCameraLocation] = useState('');
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Existing case selector
  const { data: existingCases } = useQuery({ queryKey: ['cases'], queryFn: api.cases.list });
  const [selectedCaseId, setSelectedCaseId] = useState<number | 'auto'>('auto');

  // Workflow state
  const [stage, setStage] = useState<Stage>('idle');
  const [stepIdx, setStepIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [progress, setProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState<number | null>(null);
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [facesDetected, setFacesDetected] = useState(0);
  const [matchesFound, setMatchesFound] = useState(0);
  const [resultVideoId, setResultVideoId] = useState<number | null>(null);
  const [resultPersonId, setResultPersonId] = useState<number | null>(null);

  // Restore session
  useEffect(() => {
    const saved = sessionStorage.getItem('trace_session');
    if (saved) {
      const { jobId, videoId, personId, jobStatus } = JSON.parse(saved);
      if (jobId && (jobStatus === 'QUEUED' || jobStatus === 'PROCESSING')) {
        setResultVideoId(videoId);
        setResultPersonId(personId);
        setStage('scanning');
        setStepIdx(4);
        pollJob(jobId, videoId, personId);
      }
    }
  }, []);

  const handlePhotoChange = (file: File | null) => {
    setPhoto(file);
    if (file) {
      const url = URL.createObjectURL(file);
      setPhotoPreview(url);
    } else {
      setPhotoPreview(null);
    }
  };

  const handleVideoChange = (file: File | null) => {
    setVideo(file);
    if (file && videoRef.current) {
      const url = URL.createObjectURL(file);
      videoRef.current.src = url;
    }
  };

  const canStart = photo && video && personName.trim() && cameraName.trim() && stage === 'idle';

  async function pollJob(jobId: number, videoId: number, personId: number) {
    setStage('scanning');
    setStepIdx(4);
    for (;;) {
      await new Promise(r => setTimeout(r, 1800));
      try {
        const job = await api.jobs.get(jobId);
        setProgress(job.progress ?? 0);
        setCurrentFrame(job.current_frame ?? null);
        setTotalFrames(job.total_frames ?? null);
        setFacesDetected(job.faces_detected ?? 0);
        setMatchesFound(job.matches_found ?? 0);
        if (job.status === 'COMPLETED') {
          sessionStorage.removeItem('trace_session');
          setStepIdx(6);
          setStage('done');
          setResultVideoId(videoId);
          setResultPersonId(personId);
          return;
        }
        if (job.status === 'FAILED') {
          sessionStorage.removeItem('trace_session');
          setErrorMsg(job.error_message ?? 'Scan failed');
          setStage('error');
          return;
        }
      } catch {
        // transient — keep polling
      }
    }
  }

  const handleStart = async () => {
    if (!photo || !video) return;
    setStage('creating_case');
    setStepIdx(1);
    setErrorMsg('');
    try {
      // 1. Case
      let caseId: number;
      if (selectedCaseId === 'auto') {
        const ts = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
        const c = await api.cases.create({ case_number: `CASE-${ts}`, title: `Search – ${personName}` });
        caseId = c.id;
      } else {
        caseId = selectedCaseId;
      }

      // 2. Person
      setStage('creating_person');
      setStepIdx(2);
      const p = await api.persons.create({
        name: personName.trim(),
        case_id: caseId,
        age: age ? Number(age) : undefined,
        gender: gender || undefined,
        last_known_location: location || undefined,
        description: description || undefined,
      });

      // 3. Upload reference photo
      setStage('uploading_photo');
      setStepIdx(3);
      try {
        await api.persons.uploadPhoto(p.id, photo);
      } catch (err: unknown) {
        const apiErr = err as { status?: number; message?: string };
        if (apiErr?.status === 400) {
          setErrorMsg('No face detected in the reference photo. Please upload a clear, frontal, well-lit face photo and try again.');
          setStage('error');
          return;
        }
        throw err;
      }

      // 4. Upload video
      setStage('uploading_video');
      setStepIdx(4);
      const v = await api.videos.upload({
        camera_name: cameraName.trim(),
        location: cameraLocation || undefined,
        case_id: caseId,
        file: video,
      });
      setUploadProgress(100);

      // 5. Start scan
      const job = await api.videos.process(v.id);

      sessionStorage.setItem('trace_session', JSON.stringify({
        jobId: job.id, videoId: v.id, personId: p.id, jobStatus: job.status,
      }));

      await pollJob(job.id, v.id, p.id);
    } catch (err: unknown) {
      const e = err as { message?: string };
      setErrorMsg(e?.message ?? 'An error occurred');
      setStage('error');
    }
  };

  const stepNum = (idx: number) => {
    const done = stepIdx > STAGES[idx].doneAt;
    const active = !done && (
      (idx === 0 && (stage === 'creating_case' || stage === 'creating_person' || stage === 'uploading_photo')) ||
      (idx === 1 && stage === 'uploading_video') ||
      (idx === 2 && stage === 'scanning') ||
      (idx === 3 && stage === 'done')
    );
    return { done, active };
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-4">
        <Link to="/dashboard" className="text-slate-500 hover:text-slate-800 flex items-center gap-1 text-sm">
          <ChevronLeft size={16} /> Dashboard
        </Link>
        <span className="text-slate-300">/</span>
        <span className="font-medium text-sm">New Search</span>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10 flex flex-col lg:flex-row gap-8">
        {/* ── Inputs ── */}
        <div className="flex-1 space-y-6">
          <div>
            <h1 className="text-2xl font-bold mb-1">New Search</h1>
            <p className="text-slate-500 text-sm">Upload a reference photo and CCTV footage to begin face-matching.</p>
          </div>

          {/* Person details */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <h2 className="font-semibold flex items-center gap-2"><FileImage size={18} className="text-blue-500" /> Missing Person Details</h2>

            {/* Photo drop zone */}
            <label className="relative border-2 border-dashed border-slate-200 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:border-blue-400 transition-colors overflow-hidden">
              {photoPreview ? (
                <>
                  <img src={photoPreview} alt="Preview" className="max-h-40 rounded-lg object-contain" />
                  <button type="button" onClick={e => { e.preventDefault(); handlePhotoChange(null); }}
                    className="absolute top-2 right-2 bg-white/80 rounded-full p-1 hover:bg-red-50 text-red-500">
                    <X size={14} />
                  </button>
                  <p className="text-xs text-slate-400 mt-2">{photo?.name}</p>
                </>
              ) : (
                <>
                  <Upload size={28} className="text-slate-300 mb-2" />
                  <span className="font-medium text-slate-600">Click or drop reference photo</span>
                  <span className="text-xs text-slate-400 mt-1">Clear, frontal, well-lit face required</span>
                </>
              )}
              <input type="file" className="hidden" accept="image/*"
                onChange={e => handlePhotoChange(e.target.files?.[0] || null)} />
            </label>

            <div className="grid sm:grid-cols-2 gap-4">
              <div className="sm:col-span-2">
                <label className="block text-xs font-medium text-slate-500 mb-1">Full Name *</label>
                <input value={personName} onChange={e => setPersonName(e.target.value)} placeholder="John Doe"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Age</label>
                <input type="number" value={age} onChange={e => setAge(e.target.value)} placeholder="30"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Gender</label>
                <select value={gender} onChange={e => setGender(e.target.value)}
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option value="">Prefer not to say</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="sm:col-span-2">
                <label className="block text-xs font-medium text-slate-500 mb-1">Last Known Location</label>
                <input value={location} onChange={e => setLocation(e.target.value)} placeholder="City, district…"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-xs font-medium text-slate-500 mb-1">Description</label>
                <textarea value={description} onChange={e => setDescription(e.target.value)} rows={2} placeholder="Clothing, distinguishing features…"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-xs font-medium text-slate-500 mb-1">Link to Case</label>
                <select value={selectedCaseId} onChange={e => setSelectedCaseId(e.target.value === 'auto' ? 'auto' : Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option value="auto">Create automatically</option>
                  {existingCases?.map(c => <option key={c.id} value={c.id}>{c.case_number} — {c.title || 'Untitled'}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Footage */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <h2 className="font-semibold flex items-center gap-2"><FileVideo size={18} className="text-purple-500" /> CCTV Footage</h2>

            <label className="border-2 border-dashed border-slate-200 rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:border-purple-400 transition-colors">
              {video ? (
                <div className="w-full">
                  <video ref={videoRef} className="w-full rounded-lg max-h-48"
                    onLoadedMetadata={e => setVideoDuration((e.target as HTMLVideoElement).duration)} />
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs text-slate-500">{video.name} {videoDuration ? `· ${Math.round(videoDuration)}s` : ''}</p>
                    <button type="button" onClick={e => { e.preventDefault(); handleVideoChange(null); setVideoDuration(null); }}
                      className="text-red-500 hover:text-red-700"><X size={14} /></button>
                  </div>
                  {uploadProgress > 0 && uploadProgress < 100 && (
                    <div className="mt-2 w-full bg-slate-100 rounded-full h-1.5">
                      <div className="bg-purple-500 h-1.5 rounded-full transition-all" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <Upload size={28} className="text-slate-300 mb-2" />
                  <span className="font-medium text-slate-600">Click or drop CCTV footage</span>
                  <span className="text-xs text-slate-400 mt-1">MP4, AVI, MKV, MOV</span>
                </>
              )}
              <input type="file" className="hidden" accept="video/*"
                onChange={e => handleVideoChange(e.target.files?.[0] || null)} />
            </label>

            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Camera Name *</label>
                <input value={cameraName} onChange={e => setCameraName(e.target.value)} placeholder="Cam 01 — Entrance"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1">Location</label>
                <input value={cameraLocation} onChange={e => setCameraLocation(e.target.value)} placeholder="Terminal 3, Gate B"
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              </div>
            </div>
          </div>

          {/* Start button */}
          <button onClick={handleStart} disabled={!canStart}
            className="w-full bg-blue-600 disabled:bg-slate-200 disabled:text-slate-400 hover:bg-blue-700 text-white py-4 rounded-2xl font-bold text-base transition-all flex justify-center items-center gap-2">
            {stage !== 'idle' && stage !== 'done' && stage !== 'error'
              ? <><Loader2 className="animate-spin" size={18} /> Processing…</>
              : <><Play size={18} fill="currentColor" /> Run Search</>}
          </button>

          {/* Error message */}
          {stage === 'error' && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex gap-3 text-sm text-red-700">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div>
                <div className="font-medium mb-1">Search failed</div>
                <div>{errorMsg}</div>
                <button onClick={() => { setStage('idle'); setStepIdx(0); }} className="mt-2 underline text-xs">Reset and try again</button>
              </div>
            </div>
          )}
        </div>

        {/* ── Live Results ── */}
        <div className="w-full lg:w-96 shrink-0">
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm sticky top-6 space-y-6">
            <h2 className="font-bold text-lg">Verification Results</h2>

            {stage === 'idle' && stepIdx === 0 ? (
              <div className="flex flex-col items-center justify-center py-10 text-slate-400 text-center">
                <AlertCircle size={36} className="mb-3 opacity-40" />
                <p className="text-sm">Upload a reference photo and footage, then click Run Search.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {STAGES.map((s, i) => {
                  const { done, active } = stepNum(i);
                  return (
                    <div key={i} className={`flex items-center gap-3 transition-colors ${done ? 'text-slate-900' : active ? 'text-blue-600' : 'text-slate-300'}`}>
                      {done ? <CheckCircle2 size={20} className="text-green-500 shrink-0" />
                        : active ? <Loader2 size={20} className="animate-spin text-blue-500 shrink-0" />
                        : <div className="w-5 h-5 border-2 border-current rounded-full shrink-0" />}
                      <span className="font-medium text-sm">{s.label}</span>
                    </div>
                  );
                })}

                {/* Progress ring */}
                {stage === 'scanning' && (
                  <div className="pt-4 border-t border-slate-100 space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="relative w-20 h-20">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#e2e8f0" strokeWidth="3" />
                          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#3b82f6" strokeWidth="3"
                            strokeDasharray={`${progress} 100`} strokeLinecap="round" />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center text-sm font-bold">{progress}%</div>
                      </div>
                      <div className="text-xs text-slate-500 space-y-1">
                        {currentFrame && totalFrames && <div>Frame {currentFrame} / {totalFrames}</div>}
                        <div>Faces detected: <span className="font-medium text-slate-700">{facesDetected}</span></div>
                        <div>Matches found: <span className="font-medium text-blue-600">{matchesFound}</span></div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Done CTA */}
                {stage === 'done' && resultVideoId && resultPersonId && (
                  <div className="pt-4 border-t border-slate-100">
                    <div className="bg-green-50 border border-green-200 rounded-xl p-3 text-sm text-green-700 font-medium text-center mb-3">
                      Scan complete!
                    </div>
                    <button onClick={() => navigate(`/results/${resultVideoId}/${resultPersonId}`)}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl font-bold transition-colors">
                      View Full Result & Report →
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
