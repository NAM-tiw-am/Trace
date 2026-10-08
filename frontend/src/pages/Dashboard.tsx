import { Link } from 'react-router-dom';

export default function Dashboard() {
  return (
    <div className="max-w-6xl mx-auto w-full px-6 py-12 relative z-10">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
          <p className="text-slate-600">Manage cases, persons, and footage.</p>
        </div>
        <Link to="/search" className="bg-slate-900 text-white px-4 py-2 rounded-lg font-medium hover:bg-slate-800 transition-colors">
          New Search
        </Link>
      </div>

      <div className="grid md:grid-cols-4 gap-6 mb-8">
        <div className="glass p-6 rounded-2xl">
          <div className="text-slate-500 text-sm font-medium uppercase mb-1">Active Cases</div>
          <div className="text-3xl font-bold">12</div>
        </div>
        <div className="glass p-6 rounded-2xl">
          <div className="text-slate-500 text-sm font-medium uppercase mb-1">Persons</div>
          <div className="text-3xl font-bold">45</div>
        </div>
        <div className="glass p-6 rounded-2xl">
          <div className="text-slate-500 text-sm font-medium uppercase mb-1">Videos Scanned</div>
          <div className="text-3xl font-bold">128</div>
        </div>
        <div className="glass p-6 rounded-2xl">
          <div className="text-slate-500 text-sm font-medium uppercase mb-1">Pending Review</div>
          <div className="text-3xl font-bold text-amber-600">7</div>
        </div>
      </div>

      <div className="glass p-6 rounded-3xl">
        <h2 className="text-xl font-bold mb-6">Recent Sightings</h2>
        <div className="text-center py-12 text-slate-500">
          Connect to backend API to view real data.
        </div>
      </div>
    </div>
  );
}
