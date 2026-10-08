import { HashRouter as Router, Routes, Route, NavLink, Link, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { LayoutDashboard, Folder, Users, Video, Eye, Search, Cpu } from 'lucide-react';
import Landing from './pages/Landing';
import SearchPage from './pages/Search';
import Results from './pages/Results';
import Dashboard from './pages/Dashboard';
import { api } from './lib/api';
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

function StatusChip() {
  const { data, isError } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000 });
  if (isError || !data) return (
    <div className="flex items-center gap-2 text-xs text-red-400">
      <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
      Backend offline
    </div>
  );
  return (
    <div className="text-xs text-slate-400 space-y-0.5">
      <div className="flex items-center gap-2 text-green-400 font-medium">
        <span className="w-2 h-2 rounded-full bg-green-400 inline-block animate-pulse" />
        {data.model || 'InsightFace ArcFace'}
      </div>
      <div className="text-slate-500 pl-4">Engine Ready</div>
    </div>
  );
}

function Sidebar() {
  const loc = useLocation();
  const isDashboard = loc.pathname !== '/' && loc.pathname !== '/search' && !loc.pathname.startsWith('/results');
  if (!isDashboard) return null;

  const links = [
    { to: '/dashboard', icon: LayoutDashboard, label: 'Overview' },
    { to: '/dashboard/cases', icon: Folder, label: 'Cases' },
    { to: '/dashboard/persons', icon: Users, label: 'Missing Persons' },
    { to: '/dashboard/footage', icon: Video, label: 'CCTV Footage' },
    { to: '/dashboard/sightings', icon: Eye, label: 'Sightings & Matches', badge: true },
  ];

  return (
    <aside className="w-56 shrink-0 bg-slate-900 text-slate-100 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-700">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg">
          <div className="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center shrink-0">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="4" fill="white" />
              <path d="M12 4C7.6 4 4 7.6 4 12s3.6 8 8 8 8-3.6 8-8" stroke="white" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <div>TRACE</div>
            <div className="text-[10px] font-normal text-slate-400 -mt-0.5">AI SURVEILLANCE SUITE</div>
          </div>
        </Link>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {links.map(({ to, icon: Icon, label, badge }) => (
          <NavLink key={to} to={to} end={to === '/dashboard'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`
            }>
            <Icon size={16} />
            <span className="flex-1">{label}</span>
            {badge && <SightingsBadge />}
          </NavLink>
        ))}

        <div className="pt-3 border-t border-slate-700 mt-3">
          <NavLink to="/search"
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors">
            <Search size={16} />
            New Search
          </NavLink>
        </div>
      </nav>

      {/* Engine status */}
      <div className="px-5 py-4 border-t border-slate-700">
        <div className="flex items-center gap-2 mb-1 text-xs text-slate-400">
          <Cpu size={12} />
          InsightFace ArcFace
        </div>
        <StatusChip />
      </div>
    </aside>
  );
}

function SightingsBadge() {
  const { data } = useQuery({
    queryKey: ['sightings', 'POTENTIAL_MATCH'],
    queryFn: () => api.sightings.list({ match_status: 'POTENTIAL_MATCH' }),
    refetchInterval: 10_000,
  });
  if (!data?.length) return null;
  return (
    <span className="bg-red-500 text-white text-xs rounded-full px-1.5 py-0.5 font-bold">
      {data.length}
    </span>
  );
}

function AppShell() {
  const loc = useLocation();
  const isLanding = loc.pathname === '/';

  if (isLanding) {
    return (
      <Routes>
        <Route path="/" element={<Landing />} />
      </Routes>
    );
  }

  const isDashboard = !loc.pathname.startsWith('/search') && !loc.pathname.startsWith('/results');

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      {isDashboard && <Sidebar />}
      <div className="flex-1 overflow-y-auto bg-slate-50">
        {!isDashboard && (
          <header className="sticky top-0 z-50 glass border-b border-white/20 px-6 py-4 flex items-center justify-between">
            <Link to="/dashboard" className="flex items-center gap-2 font-bold text-xl tracking-tight">
              <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="4" fill="#3B82F6" />
                  <path d="M12 4C7.6 4 4 7.6 4 12s3.6 8 8 8 8-3.6 8-8" stroke="white" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
              </div>
              Trace
            </Link>
            <nav className="flex gap-6 items-center font-medium text-sm text-slate-600">
              <Link to="/search" className="hover:text-blue-600 transition-colors">New Search</Link>
              <Link to="/dashboard" className="hover:text-blue-600 transition-colors">Dashboard</Link>
            </nav>
          </header>
        )}
        <Routes>
          <Route path="/search" element={<SearchPage />} />
          <Route path="/results/:videoId/:personId" element={<Results />} />
          <Route path="/dashboard" element={<Dashboard tab="overview" />} />
          <Route path="/dashboard/cases" element={<Dashboard tab="cases" />} />
          <Route path="/dashboard/persons" element={<Dashboard tab="persons" />} />
          <Route path="/dashboard/footage" element={<Dashboard tab="footage" />} />
          <Route path="/dashboard/sightings" element={<Dashboard tab="sightings" />} />
          <Route path="*" element={<div className="p-12 text-center text-slate-500">Page not found</div>} />
        </Routes>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppShell />
      </Router>
    </QueryClientProvider>
  );
}
