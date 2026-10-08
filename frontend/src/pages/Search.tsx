import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileVideo, FileImage, Play, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { apiUrl } from '../lib/api';

export default function Search() {
  const navigate = useNavigate();
  const [photo, setPhoto] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [step, setStep] = useState(0);

  // @ts-expect-error unused
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch(apiUrl('/api/health')).then(res => res.json())
  });

  const handleStartSearch = async () => {
    if (!photo || !video) return;
    setIsScanning(true);
    
    // Mock progression for demonstration without full backend wiring
    setStep(1); // Uploading photo
    await new Promise(r => setTimeout(r, 1000));
    setStep(2); // Uploading video
    await new Promise(r => setTimeout(r, 1500));
    setStep(3); // Scanning
    await new Promise(r => setTimeout(r, 2000));
    setStep(4); // Complete
    
    // In a real app, we'd hit the API endpoints here:
    // POST /api/cases -> POST /api/persons -> POST /api/persons/{id}/photos -> POST /api/videos -> POST /api/videos/{id}/process
    
    navigate('/results/mock-video-id/mock-person-id');
  };

  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-12 flex flex-col md:flex-row gap-8 relative z-10">
      {/* Inputs Side */}
      <div className="flex-1 space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">New Search</h1>
          <p className="text-slate-600">Upload a reference photo and footage to begin.</p>
        </div>

        {/* Photo Upload */}
        <div className="glass p-6 rounded-3xl">
          <h2 className="font-semibold mb-4 flex items-center gap-2"><FileImage size={20} className="text-blue-500" /> Reference Photo</h2>
          <label className="border-2 border-dashed border-slate-300 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-slate-50 transition-colors">
            <Upload size={32} className="text-slate-400 mb-3" />
            <span className="font-medium text-slate-700">Click to upload photo</span>
            <span className="text-sm text-slate-500 mt-1">Clear, frontal face required</span>
            <input type="file" className="hidden" accept="image/*" onChange={(e) => setPhoto(e.target.files?.[0] || null)} />
          </label>
          {photo && <div className="mt-3 text-sm text-blue-600 bg-blue-50 p-2 rounded-lg">{photo.name}</div>}
        </div>

        {/* Video Upload */}
        <div className="glass p-6 rounded-3xl">
          <h2 className="font-semibold mb-4 flex items-center gap-2"><FileVideo size={20} className="text-purple-500" /> CCTV Footage</h2>
          <label className="border-2 border-dashed border-slate-300 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-slate-50 transition-colors">
            <Upload size={32} className="text-slate-400 mb-3" />
            <span className="font-medium text-slate-700">Click to upload video</span>
            <span className="text-sm text-slate-500 mt-1">MP4, AVI, MKV</span>
            <input type="file" className="hidden" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0] || null)} />
          </label>
          {video && <div className="mt-3 text-sm text-purple-600 bg-purple-50 p-2 rounded-lg">{video.name}</div>}
        </div>

        <button 
          onClick={handleStartSearch}
          disabled={!photo || !video || isScanning}
          className="w-full bg-blue-600 disabled:bg-slate-300 hover:bg-blue-700 text-white py-4 rounded-2xl font-bold text-lg transition-all flex justify-center items-center gap-2"
        >
          {isScanning ? <><Loader2 className="animate-spin" /> Processing...</> : <><Play fill="currentColor" size={20} /> Start Search</>}
        </button>
      </div>

      {/* Live Results Side */}
      <div className="flex-1 lg:max-w-md">
        <div className="glass p-8 rounded-3xl h-full flex flex-col">
          <h2 className="font-bold text-xl mb-6">Verification Results</h2>
          
          {!isScanning && step === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-500">
              <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                <AlertCircle size={32} className="text-slate-400" />
              </div>
              <p>Upload a photo and video to see live results.</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className={clsx("flex items-center gap-3", step >= 1 ? "text-slate-900" : "text-slate-400")}>
                {step > 1 ? <CheckCircle2 className="text-green-500" /> : step === 1 ? <Loader2 className="animate-spin text-blue-500" /> : <div className="w-6 h-6 border-2 border-slate-300 rounded-full"></div>}
                <span className="font-medium">Reference face detected</span>
              </div>
              <div className={clsx("flex items-center gap-3", step >= 2 ? "text-slate-900" : "text-slate-400")}>
                {step > 2 ? <CheckCircle2 className="text-green-500" /> : step === 2 ? <Loader2 className="animate-spin text-blue-500" /> : <div className="w-6 h-6 border-2 border-slate-300 rounded-full"></div>}
                <span className="font-medium">Footage uploaded</span>
              </div>
              <div className={clsx("flex items-center gap-3", step >= 3 ? "text-slate-900" : "text-slate-400")}>
                {step > 3 ? <CheckCircle2 className="text-green-500" /> : step === 3 ? <Loader2 className="animate-spin text-blue-500" /> : <div className="w-6 h-6 border-2 border-slate-300 rounded-full"></div>}
                <span className="font-medium">Scanning footage</span>
              </div>
              <div className={clsx("flex items-center gap-3", step >= 4 ? "text-slate-900" : "text-slate-400")}>
                {step >= 4 ? <CheckCircle2 className="text-green-500" /> : <div className="w-6 h-6 border-2 border-slate-300 rounded-full"></div>}
                <span className="font-medium">Match evaluation</span>
              </div>

              {step === 3 && (
                <div className="pt-6 border-t border-slate-200">
                  <div className="text-sm text-slate-500 mb-2 flex justify-between">
                    <span>Progress</span>
                    <span>45%</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div className="bg-blue-600 h-2 rounded-full" style={{ width: '45%' }}></div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
