import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, ShieldCheck, Zap, Users } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiUrl } from '../lib/api';

export default function Landing() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch(apiUrl('/api/health')).then(res => res.json())
  });

  return (
    <div className="flex flex-col items-center">
      {/* Hero Section */}
      <section className="w-full max-w-6xl mx-auto px-6 py-24 flex flex-col md:flex-row items-center gap-12">
        <div className="flex-1 space-y-8">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl font-extrabold tracking-tight text-slate-900 leading-tight"
          >
            Find them <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">faster.</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-slate-600 max-w-lg leading-relaxed"
          >
            Trace uses motion-aware AI to rapidly scan hours of CCTV footage for missing persons, turning days of manual review into minutes.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-wrap gap-4"
          >
            <Link to="/search" className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-2xl font-semibold flex items-center gap-2 shadow-lg shadow-blue-500/30 transition-all hover:-translate-y-1">
              Start a search <ArrowRight size={20} />
            </Link>
          </motion.div>
        </div>
        
        {/* Floating Cards Visual */}
        <div className="flex-1 relative w-full h-[500px]">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="absolute right-10 top-10 w-80 glass rounded-3xl p-6 shadow-2xl"
          >
            <div className="flex justify-between items-center mb-6">
              <div className="h-2 w-12 bg-slate-200 rounded-full"></div>
              <span className="bg-green-100 text-green-700 text-xs font-bold px-3 py-1 rounded-full">Match found 0.82</span>
            </div>
            <div className="aspect-square bg-slate-100 rounded-2xl mb-4 relative overflow-hidden flex items-center justify-center">
              <div className="absolute inset-0 border-4 border-blue-500/20 m-4 rounded-xl"></div>
              <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-blue-500 rounded-tl-xl m-4"></div>
              <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-blue-500 rounded-tr-xl m-4"></div>
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-blue-500 rounded-bl-xl m-4"></div>
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-blue-500 rounded-br-xl m-4"></div>
              <motion.div 
                animate={{ top: ['0%', '100%', '0%'] }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 right-0 h-1 bg-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.8)] z-10"
              />
              <Users size={64} className="text-slate-300" />
            </div>
            <div className="space-y-2">
              <div className="h-3 w-full bg-slate-100 rounded-full"></div>
              <div className="h-3 w-2/3 bg-slate-100 rounded-full"></div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="w-full bg-white/50 border-y border-white py-12 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="text-4xl font-extrabold text-slate-900 mb-2">
              {health?.thresholds?.match ? health.thresholds.match : '0.35'}
            </div>
            <div className="text-sm font-medium text-slate-500 uppercase tracking-wider">Match Threshold</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-extrabold text-slate-900 mb-2">
              {health?.thresholds?.high_confidence ? health.thresholds.high_confidence : '0.55'}
            </div>
            <div className="text-sm font-medium text-slate-500 uppercase tracking-wider">High Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-extrabold text-slate-900 mb-2">
              {health?.model ? health.model : 'ArcFace'}
            </div>
            <div className="text-sm font-medium text-slate-500 uppercase tracking-wider">AI Model</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-extrabold text-slate-900 mb-2">
              {health?.thresholds?.cooldown_seconds ? health.thresholds.cooldown_seconds + 's' : '5s'}
            </div>
            <div className="text-sm font-medium text-slate-500 uppercase tracking-wider">Detection Cooldown</div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="w-full max-w-6xl mx-auto px-6 py-24">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">How it works</h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">A seamless pipeline from enrollment to actionable intelligence.</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass p-8 rounded-3xl">
            <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center mb-6">
              <Users size={24} />
            </div>
            <h3 className="text-xl font-bold mb-3">1. Enroll Face</h3>
            <p className="text-slate-600">Upload a single clear photo. Trace extracts a 512-dimensional embedding using state-of-the-art models.</p>
          </div>
          <div className="glass p-8 rounded-3xl">
            <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-xl flex items-center justify-center mb-6">
              <Zap size={24} />
            </div>
            <h3 className="text-xl font-bold mb-3">2. Smart Scanning</h3>
            <p className="text-slate-600">Upload CCTV footage. Trace skips static frames, focusing compute only when motion is detected.</p>
          </div>
          <div className="glass p-8 rounded-3xl">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-xl flex items-center justify-center mb-6">
              <ShieldCheck size={24} />
            </div>
            <h3 className="text-xl font-bold mb-3">3. Human Verification</h3>
            <p className="text-slate-600">Review grouped sightings, confirm matches, and generate court-ready evidence reports.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
