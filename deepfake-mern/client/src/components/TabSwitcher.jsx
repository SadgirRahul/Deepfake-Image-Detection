function TabSwitcher({ activeTab, analysisComplete, setActiveTab }) {
  return (
    <div className="border-b border-slate-800 bg-[#0f172a]">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-center gap-6 px-4">
        <button
          type="button"
          onClick={() => setActiveTab('analyze')}
          className={`border-b-2 px-2 py-4 text-sm font-semibold transition-colors md:text-base ${
            activeTab === 'analyze'
              ? 'border-blue-400 text-white'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Analyze
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('report')}
          disabled={!analysisComplete}
          className={`border-b-2 px-2 py-4 text-sm font-semibold transition-colors md:text-base ${
            activeTab === 'report'
              ? 'border-blue-400 text-white'
              : 'border-transparent text-slate-500'
          } ${
            analysisComplete
              ? 'hover:text-slate-200'
              : 'cursor-not-allowed text-slate-600'
          }`}
        >
          Analysis Report
        </button>
      </div>
    </div>
  );
}

export default TabSwitcher;
