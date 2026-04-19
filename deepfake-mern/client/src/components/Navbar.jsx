function Navbar({ activeTab, analysisComplete, setActiveTab }) {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0f172a]/95 backdrop-blur-[12px]">
      <nav className="mx-auto flex h-16 w-full max-w-7xl items-center px-6 relative">
        <div className="flex shrink-0 items-center gap-3 text-slate-100 transition-transform duration-200 hover:scale-105">
          <div className="relative">
            <span className="relative inline-flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800 p-1.5">
              <svg viewBox="0 0 32 32" fill="none" strokeWidth="1.6" className="h-7 w-7">
                <defs>
                  <linearGradient id="shieldGradient" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
                <path
                  d="M16 3.5 6.5 7.2v7.1c0 5 3.6 9.6 9.5 11.1 5.9-1.5 9.5-6.1 9.5-11.1V7.2L16 3.5Z"
                  stroke="url(#shieldGradient)"
                />
                <line x1="9" y1="12" x2="23" y2="12" stroke="url(#shieldGradient)" className="scan-line" />
              </svg>
            </span>
            <span className="pulse-dot absolute -bottom-0.5 -right-0.5 h-2 w-2 rounded-full bg-blue-500" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-[15px] font-semibold text-white">DeepFake</span>
            <span className="brand-text text-[15px] font-semibold">Detector</span>
          </div>
        </div>

        <div className="absolute left-1/2 -translate-x-1/2">
          <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-slate-800/80 p-1">
            <button
              type="button"
              onClick={() => setActiveTab('analyze')}
              className={`rounded-lg px-5 py-2 text-sm font-medium transition-all duration-200 ${
                activeTab === 'analyze'
                  ? 'bg-blue-600 text-white shadow-[0_4px_15px_rgba(59,130,246,0.3)]'
                  : 'text-slate-400 hover:bg-slate-700/50 hover:text-white'
              }`}
            >
              Analyze
            </button>
            <button
              type="button"
              onClick={() => {
                if (analysisComplete) {
                  setActiveTab('report');
                }
              }}
              className={`rounded-lg px-5 py-2 text-sm transition-all duration-200 ${
                activeTab === 'report'
                  ? 'bg-blue-600 text-white font-medium shadow-[0_4px_15px_rgba(59,130,246,0.3)]'
                  : analysisComplete
                    ? 'text-slate-400 hover:bg-slate-700/50 hover:text-white'
                    : 'cursor-not-allowed text-slate-600'
              }`}
              disabled={!analysisComplete}
            >
              Analysis Report
              {analysisComplete && (
                <span className="ml-2 inline-flex h-2 w-2 rounded-full bg-green-400 animate-pulse" />
              )}
            </button>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-slate-400 bg-slate-800/80 px-3 py-1 rounded-full border border-slate-700/60">
            EfficientNet-B0
          </span>
          <span className="group relative flex items-center">
            <span
              className={`h-1.5 w-1.5 rounded-full ${analysisComplete ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`}
            />
            <span className="pointer-events-none absolute right-0 top-full mt-2 w-max rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-xs text-slate-200 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
              {analysisComplete ? 'Model ready' : 'Awaiting analysis'}
            </span>
          </span>
        </div>
      </nav>
    </header>
  );
}

export default Navbar;
