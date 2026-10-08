import { HashRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Landing from './pages/Landing';
import Search from './pages/Search';
import Results from './pages/Results';
import Dashboard from './pages/Dashboard';
import './styles/globals.css';

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col relative overflow-hidden">
          {/* Global Gradient Background Blobs */}
          <div className="absolute top-0 left-[-10%] w-96 h-96 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
          <div className="absolute top-0 right-[-10%] w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
          <div className="absolute -bottom-8 left-20 w-96 h-96 bg-pink-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>

          <header className="sticky top-0 z-50 glass border-b border-white/20 px-6 py-4 flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 font-bold text-xl tracking-tight">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="32" height="32" rx="8" fill="#1E293B"/>
                <path d="M16 8C11.5817 8 8 11.5817 8 16C8 20.4183 11.5817 24 16 24C20.4183 24 24 20.4183 24 16" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                <circle cx="16" cy="16" r="3" fill="#3B82F6"/>
              </svg>
              Trace
            </Link>
            <nav className="flex gap-6 items-center font-medium text-sm text-slate-600">
              <Link to="/search" className="hover:text-blue-600 transition-colors">Start a search</Link>
              <Link to="/dashboard" className="hover:text-blue-600 transition-colors">Dashboard</Link>
            </nav>
          </header>
          
          <main className="flex-1 relative z-10 flex flex-col">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/search" element={<Search />} />
              <Route path="/results/:videoId/:personId" element={<Results />} />
              <Route path="/dashboard" element={<Dashboard />} />
            </Routes>
          </main>
          
          <footer className="bg-slate-900 text-slate-400 py-12 px-6 mt-20 relative z-10">
            <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="flex items-center gap-2 text-white font-bold text-lg">
                <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
                  <rect width="32" height="32" rx="8" fill="#3B82F6"/>
                  <circle cx="16" cy="16" r="4" fill="white"/>
                </svg>
                Trace
              </div>
              <div className="text-sm">
                AI-assisted missing person identification. For investigational use only.
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </QueryClientProvider>
  );
}
