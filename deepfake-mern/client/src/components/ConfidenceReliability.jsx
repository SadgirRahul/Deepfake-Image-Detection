function ConfidenceReliability({ confidence, modelAccuracy, metricsLoading }) {
  const normalizedConfidence = Math.min(100, Math.max(0, Number(confidence) || 0));
  const normalizedAccuracy = Math.min(100, Math.max(0, Number(modelAccuracy) || 0));

  let label = 'Low Confidence — treat with caution';
  let color = 'bg-rose-500';
  let textColor = 'text-rose-200';
  let icon = (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v5" />
      <path d="M12 16h.01" />
    </svg>
  );

  if (normalizedConfidence >= 90) {
    label = 'Very High Confidence';
    color = 'bg-emerald-500';
    textColor = 'text-emerald-200';
    icon = (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M12 3 5 6v6.6c0 4.1 3 7.8 7 9 4-1.2 7-4.9 7-9V6l-7-3Z" />
        <path d="m9.5 12.5 1.8 1.8 3.6-3.6" />
      </svg>
    );
  } else if (normalizedConfidence >= 75) {
    label = 'High Confidence';
    color = 'bg-blue-500';
    textColor = 'text-blue-200';
    icon = (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="9" />
        <path d="m8.5 12.5 2.2 2.2 4.8-4.8" />
      </svg>
    );
  } else if (normalizedConfidence >= 60) {
    label = 'Moderate Confidence';
    color = 'bg-amber-500';
    textColor = 'text-amber-200';
    icon = (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M12 3 2.5 20h19L12 3Z" />
        <path d="M12 9v5" />
        <path d="M12 17h.01" />
      </svg>
    );
  }

  return (
    <div className="group relative inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/70 px-3 py-1 text-xs font-semibold text-slate-200">
      <span className={`h-2.5 w-2.5 rounded-full ${color}`} />
      <span className={textColor}>{label}</span>
      <span className="text-slate-400">{icon}</span>
      <span className="pointer-events-none absolute left-0 top-full mt-2 w-max rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-xs text-slate-200 opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100">
        {metricsLoading || !modelAccuracy
          ? 'Model metrics unavailable'
          : `Model overall accuracy is ${normalizedAccuracy}%. This prediction confidence is ${normalizedConfidence}%.`}
      </span>
    </div>
  );
}

export default ConfidenceReliability;
