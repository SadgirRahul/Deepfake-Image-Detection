import { useEffect, useMemo, useState } from 'react';

function clampPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.min(100, Math.max(0, numeric));
}

function AccuracyConfidencePanel({ confidence, label, modelMetrics, metricsLoading }) {
  const normalizedConfidence = clampPercent(confidence);
  const isFake = String(label || '').toUpperCase() === 'FAKE';
  const arcColor = isFake ? '#f87171' : '#34d399';

  const metrics = modelMetrics || {};
  const rows = useMemo(() => ([
    { key: 'overall_accuracy', label: 'Overall Accuracy', value: metrics.overall_accuracy ?? 0, color: 'bg-blue-500' },
    { key: 'real_accuracy', label: 'Real Detection', value: metrics.real_accuracy ?? 0, color: 'bg-emerald-500' },
    { key: 'fake_accuracy', label: 'Fake Detection', value: metrics.fake_accuracy ?? 0, color: 'bg-rose-500' },
    { key: 'auc_roc', label: 'AUC-ROC Score', value: metrics.auc_roc ?? 0, color: 'bg-purple-500' },
  ]), [metrics]);

  const [barValues, setBarValues] = useState(rows.map(() => 0));

  useEffect(() => {
    setBarValues(rows.map(() => 0));
    const id = setTimeout(() => {
      setBarValues(rows.map((row) => clampPercent(row.value)));
    }, 80);
    return () => clearTimeout(id);
  }, [rows]);

  const radius = 66;
  const circumference = 2 * Math.PI * radius;
  const progress = (normalizedConfidence / 100) * circumference;
  const dashOffset = circumference - progress;

  return (
    <section className="w-full rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-xl">
      <div className="grid gap-6 md:grid-cols-2">
        <div className="flex flex-col items-center text-center">
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
            Prediction Confidence
          </h3>
          <div className="relative mt-5 flex h-44 w-44 items-center justify-center">
            {metricsLoading ? (
              <div className="h-36 w-36 animate-pulse rounded-full border-4 border-slate-800" />
            ) : (
              <>
                <svg viewBox="0 0 200 200" className="h-full w-full">
                  <circle
                    cx="100"
                    cy="100"
                    r={radius}
                    fill="none"
                    stroke="#1f2937"
                    strokeWidth="12"
                  />
                  <circle
                    cx="100"
                    cy="100"
                    r={radius}
                    fill="none"
                    stroke={arcColor}
                    strokeWidth="12"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={dashOffset}
                    transform="rotate(90 100 100)"
                  />
                </svg>
                <div className="absolute flex flex-col items-center">
                  <span className="text-3xl font-semibold text-slate-100">
                    {normalizedConfidence}%
                  </span>
                  <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Confidence</span>
                </div>
              </>
            )}
          </div>
          <p className="mt-4 text-sm text-slate-200">
            This image is {normalizedConfidence}% likely to be {isFake ? 'FAKE' : 'REAL'}.
          </p>
        </div>

        <div>
          {metricsLoading ? (
            <div className="mt-4 space-y-4">
              {rows.map((row) => (
                <div key={row.key} className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span className="h-3 w-24 rounded-full bg-slate-800/80" />
                    <span className="h-3 w-10 rounded-full bg-slate-800/70" />
                  </div>
                  <div className="h-2 rounded-full bg-slate-800">
                    <div className="h-2 w-1/2 rounded-full bg-slate-700" />
                  </div>
                </div>
              ))}
            </div>
          ) : modelMetrics ? (
            <div className="mt-4 space-y-4">
              {rows.map((row, index) => (
                <div key={row.key} className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>{row.label}</span>
                    <span className="text-slate-200">{clampPercent(row.value)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800">
                    <div
                      className={`h-2 rounded-full ${row.color} transition-all duration-[1200ms] ease-out`}
                      style={{ width: `${barValues[index]}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default AccuracyConfidencePanel;
