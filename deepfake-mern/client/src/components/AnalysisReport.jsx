import ManipulationHeatmap from './ManipulationHeatmap';

const FEATURES = [
  {
    key: 'ela',
    name: 'Error Level Analysis',
    description: 'Highlights compression inconsistencies and edits.',
    accent: 'border-l-blue-500 text-blue-300',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 4h16v16H4z" />
        <path d="M8 8h8v8H8z" />
      </svg>
    ),
    insight: {
      real: 'Normal compression levels detected.',
      fake: 'High compression artifacts detected.',
    },
  },
  {
    key: 'fft',
    name: 'FFT Spectrum',
    description: 'Reveals unnatural frequency patterns.',
    accent: 'border-l-purple-500 text-purple-300',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 18h16" />
        <path d="M6 18V9" />
        <path d="M10 18V6" />
        <path d="M14 18v-4" />
        <path d="M18 18V8" />
      </svg>
    ),
    insight: {
      real: 'Frequency profile looks natural.',
      fake: 'Synthetic frequency spikes detected.',
    },
  },
  {
    key: 'edges',
    name: 'Edge Detection',
    description: 'Checks edge continuity around facial regions.',
    accent: 'border-l-emerald-500 text-emerald-300',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 12h8" />
        <path d="M12 4v8" />
        <path d="m12 12 8 8" />
      </svg>
    ),
    insight: {
      real: 'Edges appear smooth and consistent.',
      fake: 'Edge discontinuities detected.',
    },
  },
  {
    key: 'gradcam',
    name: 'Grad-CAM Attention',
    description: 'Shows model focus regions.',
    accent: 'border-l-orange-500 text-orange-300',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
    insight: {
      real: 'Attention concentrates on natural features.',
      fake: 'Attention clusters on manipulated areas.',
    },
  },
];

const normalizeImage = (src) => {
  if (!src) {
    return '';
  }

  if (src.startsWith('data:')) {
    return src;
  }

  return `data:image/png;base64,${src}`;
};

const clampPercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.min(100, Math.max(0, numeric));
};

const formatNumber = (value) => Number(value || 0).toLocaleString('en-US');

const MiniGauge = ({ label, value, color }) => {
  const radius = 32;
  const circumference = 2 * Math.PI * radius;
  const progress = (clampPercent(value) / 100) * circumference;
  const dashOffset = circumference - progress;

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 80 80" className="h-20 w-20">
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#1f2937" strokeWidth="6" />
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform="rotate(90 40 40)"
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-slate-100 text-[14px] font-semibold"
        >
          {clampPercent(value)}%
        </text>
      </svg>
      <span className="mt-2 text-xs text-slate-400">{label}</span>
    </div>
  );
};

const buildSmoothPath = (points) => {
  if (!points.length) {
    return '';
  }

  if (points.length === 1) {
    const [x, y] = points[0];
    return `M ${x} ${y}`;
  }

  const path = [`M ${points[0][0]} ${points[0][1]}`];

  for (let i = 0; i < points.length - 1; i += 1) {
    const [x0, y0] = points[i - 1] || points[i];
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    const [x3, y3] = points[i + 2] || points[i + 1];

    const cp1x = x1 + (x2 - x0) / 6;
    const cp1y = y1 + (y2 - y0) / 6;
    const cp2x = x2 - (x3 - x1) / 6;
    const cp2y = y2 - (y3 - y1) / 6;

    path.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`);
  }

  return path.join(' ');
};

const buildAreaPath = (points, lineY) => {
  if (!points.length) {
    return '';
  }

  const top = points.map(([x]) => [x, lineY]);
  const bottom = points.map(([x, y]) => [x, Math.max(y, lineY)]).reverse();

  const [startX, startY] = top[0];
  const path = [`M ${startX} ${startY}`];

  top.slice(1).forEach(([x, y]) => {
    path.push(`L ${x} ${y}`);
  });

  bottom.forEach(([x, y]) => {
    path.push(`L ${x} ${y}`);
  });

  path.push('Z');
  return path.join(' ');
};

const SkeletonCard = () => (
  <div className="rounded-2xl border border-slate-700 bg-slate-900/80 p-4 shadow-xl">
    <div className="flex items-start justify-between gap-3">
      <div className="flex items-center gap-3">
        <div className="h-9 w-9 rounded-xl bg-slate-800/80" />
        <div>
          <div className="h-4 w-36 rounded-full bg-slate-800/80" />
          <div className="mt-2 h-3 w-24 rounded-full bg-slate-800/70" />
        </div>
      </div>
      <div className="h-8 w-8 rounded-lg bg-slate-800/80" />
    </div>
    <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950">
      <div className="aspect-[4/3] w-full animate-pulse bg-slate-800/60" />
    </div>
    <div className="mt-4 flex items-center justify-between">
      <div className="h-4 w-40 rounded-full bg-slate-800/80" />
      <div className="h-3 w-20 rounded-full bg-slate-800/70" />
    </div>
  </div>
);

function AnalysisReport({ result, uploadedImage, loading, modelMetrics, metricsLoading, confidence, label }) {
  const isFake = String(result?.label || '').toLowerCase() === 'fake';
  const images = result?.visualizations || {};
  const metrics = modelMetrics || {};
  const confidenceValue = clampPercent(confidence ?? result?.confidence ?? 0);
  const accuracyValue = clampPercent(metrics.overall_accuracy ?? 0);
  const confidenceHigher = confidenceValue > accuracyValue;
  const epochData = [52, 61, 68, 74, 79, 83, 86, 89, 90, 91, 92, 93, 93.2];

  const onDownload = (key) => {
    const image = normalizeImage(images[key]);
    if (!image) {
      return;
    }

    const link = document.createElement('a');
    link.href = image;
    link.download = `${key}-analysis.png`;
    link.click();
  };

  return (
    <section className="mx-auto w-full max-w-6xl space-y-6 p-4">
      <div className={`grid gap-6 md:grid-cols-2 ${loading ? 'animate-pulse' : ''}`}>
        {loading ? (
          FEATURES.map((feature) => <SkeletonCard key={feature.key} />)
        ) : (
          FEATURES.map((feature) => {
            const imageSrc = normalizeImage(images[feature.key]);
            const insightText = isFake ? feature.insight.fake : feature.insight.real;

            return (
              <article
                key={feature.key}
                className={`relative rounded-2xl border border-slate-700 bg-slate-900/80 p-4 shadow-xl border-l-4 ${feature.accent}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className={`inline-flex h-9 w-9 items-center justify-center rounded-xl bg-slate-950/60 ${feature.accent}`}>
                      {feature.icon}
                    </span>
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100 md:text-base">{feature.name}</h3>
                      <div className="group relative mt-1 inline-flex items-center">
                        <button
                          type="button"
                          className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-slate-700 text-xs font-semibold text-slate-200"
                          aria-label={`${feature.name} info`}
                        >
                          ?
                        </button>
                        <span className="pointer-events-none absolute left-8 top-1/2 w-max -translate-y-1/2 rounded-full bg-slate-950 px-3 py-1 text-xs text-slate-200 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                          {feature.description}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onDownload(feature.key)}
                    disabled={!imageSrc}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-700 text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                    aria-label={`Download ${feature.name} image`}
                  >
                    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <path d="M12 3v12" />
                      <path d="m7 10 5 5 5-5" />
                      <path d="M5 21h14" />
                    </svg>
                  </button>
                </div>

                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950">
                  <div className="aspect-[4/3] w-full">
                    {imageSrc ? (
                      <img
                        src={imageSrc}
                        alt={`${feature.name} output`}
                        className="h-full w-full rounded-lg object-contain"
                      />
                    ) : (
                      <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
                        No image available
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <span className={`text-sm font-semibold ${isFake ? 'text-rose-300' : 'text-emerald-300'}`}>
                    {insightText}
                  </span>
                  <span className="text-xs text-slate-500">Confidence: {result?.confidence ?? '--'}%</span>
                </div>
                {feature.key === 'gradcam' && (
                  <p className="mt-2 text-center text-xs italic text-slate-500">
                    Full manipulation analysis available in the heatmap section below ↓
                  </p>
                )}
              </article>
            );
          })
        )}
      </div>

      <article className="rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-xl">
        <header className="mb-4">
          <h3 className="text-lg font-semibold text-slate-100">Manipulation Region Heatmap</h3>
        </header>
        {loading ? (
          <div className="animate-pulse rounded-xl border border-slate-800 bg-slate-950/60 p-6 text-center text-sm text-slate-400">
            Generating heatmap...
          </div>
        ) : (
          <ManipulationHeatmap
            originalImage={uploadedImage}
            gradcamImage={result?.visualizations?.gradcam}
            label={result?.label}
          />
        )}
      </article>
    </section>
  );
}

export default AnalysisReport;
