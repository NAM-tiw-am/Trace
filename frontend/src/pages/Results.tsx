// import { useParams } from 'react-router-dom';
import { Download, CheckCircle, AlertTriangle, FileText, Video, Archive } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

export default function Results() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch('/api/health').then(res => res.json())
  });

  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-12 relative z-10">
      {/* Verdict Banner */}
      <div className="bg-green-50 border border-green-200 rounded-3xl p-6 mb-8 flex items-center gap-6 shadow-sm">
        <div className="bg-green-100 text-green-700 p-4 rounded-full">
          <CheckCircle size={32} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-green-800 mb-1">MATCH FOUND: HIGH CONFIDENCE</h1>
          <p className="text-green-700">AI-assisted suggestion. Not a positive identification. Requires human verification.</p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <div className="md:col-span-2 space-y-8">
          <div className="glass p-8 rounded-3xl">
            <h2 className="text-xl font-bold mb-6">Evidence Gallery</h2>
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="border border-slate-200 rounded-xl overflow-hidden cursor-pointer hover:border-blue-400 transition-colors">
                <div className="bg-slate-100 aspect-video flex items-center justify-center">
                  <span className="text-slate-400">Snapshot Image</span>
                </div>
                <div className="p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">00:14:23</span>
                    <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded-full">0.82 Score</span>
                  </div>
                  <div className="text-sm text-slate-500">Frame: 25841 • Status: Potential Match</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="glass p-8 rounded-3xl">
            <h2 className="text-xl font-bold mb-6">Matching Criteria applied</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">Model</span>
                <span className="font-medium">{health?.model || 'ArcFace'}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">Match Threshold</span>
                <span className="font-medium">{health?.thresholds?.match || 0.35}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">High Confidence Threshold</span>
                <span className="font-medium">{health?.thresholds?.high_confidence || 0.55}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass p-6 rounded-3xl">
            <h2 className="font-bold text-lg mb-4">Export Results</h2>
            <div className="space-y-3">
              <button className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 hover:border-slate-300 transition-colors">
                <div className="flex items-center gap-3">
                  <FileText size={20} className="text-red-500" />
                  <span className="font-medium">Evidence Report</span>
                </div>
                <Download size={16} className="text-slate-400" />
              </button>
              <button className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 hover:border-slate-300 transition-colors">
                <div className="flex items-center gap-3">
                  <Archive size={20} className="text-amber-500" />
                  <span className="font-medium">Evidence Bundle (ZIP)</span>
                </div>
                <Download size={16} className="text-slate-400" />
              </button>
              <button className="w-full flex items-center justify-between p-3 border border-slate-200 rounded-xl hover:bg-slate-50 hover:border-slate-300 transition-colors">
                <div className="flex items-center gap-3">
                  <Video size={20} className="text-purple-500" />
                  <span className="font-medium">Original Footage</span>
                </div>
                <Download size={16} className="text-slate-400" />
              </button>
            </div>
          </div>
          
          <div className="bg-amber-50 border border-amber-200 p-4 rounded-2xl flex gap-3 text-sm text-amber-800">
            <AlertTriangle className="shrink-0 mt-0.5" size={16} />
            <p>Ensure these results are reviewed by a certified human investigator before use in official capacity.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
