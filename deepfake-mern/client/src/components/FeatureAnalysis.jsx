import { useState } from 'react';

const FEATURES = [
  {
    key: 'ela',
    title: 'Error Level Analysis (ELA)',
    short: 'ELA',
    accent: 'text-blue-400',
    border: 'border-blue-500/40',
    bg: 'bg-blue-500/10',
    reveal: 'Compression inconsistencies that can indicate edited regions.',
    tooltip: 'Detects manipulation via JPEG recompression artifacts.',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 4h16v16H4z" />
        <path d="M8 8h8v8H8z" />
      </svg>
    ),
  },
  {
    key: 'fft',
    title: 'FFT Frequency Spectrum',
    short: 'FFT',
    accent: 'text-pink-400',
    border: 'border-pink-500/40',
    bg: 'bg-pink-500/10',
    reveal: 'Unnatural periodic patterns often left by generative pipelines.',
    tooltip: 'Highlights suspicious frequency-domain artifacts.',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 18h16" />
        <path d="M6 18V9" />
        <path d="M10 18V6" />
        <path d="M14 18v-4" />
        <path d="M18 18V8" />
      </svg>
    ),
  },
  {
    key: 'edges',
    title: 'Edge Detection Map',
    short: 'Edges',
    accent: 'text-purple-400',
    border: 'border-purple-500/40',
    bg: 'bg-purple-500/10',
    reveal: 'Boundary anomalies around facial contours and blended regions.',
    tooltip: 'Reveals discontinuities and edge inconsistencies.',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 12h8" />
        <path d="M12 4v8" />
        <path d="m12 12 8 8" />
      </svg>
    ),
  },
  {
    key: 'gradcam',
    title: 'Grad-CAM Attention',
    short: 'Grad-CAM',
    accent: 'text-orange-400',
    border: 'border-orange-500/40',
    bg: 'bg-orange-500/10',
    reveal: 'Where the model focuses most when deciding Real vs Fake.',
    tooltip: 'Visual explanation of model attention regions.',
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
];

function FeatureCard({ feature, src, alt }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <article className={`group rounded-2xl border bg-slate-900/80 p-4 shadow-xl ${feature.border}`}>
      <header className="mb-3 flex items-center gap-2">
        <span className={`inline-flex h-8 w-8 items-center justify-center rounded-lg ${feature.bg} ${feature.accent}`}>
          {feature.icon}
        </span>
        <h3 className="text-sm font-semibold text-slate-100 md:text-base">{feature.title}</h3>
      </header>

      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
        <span className="pointer-events-none absolute right-2 top-2 z-10 rounded-full bg-slate-950/90 px-2 py-1 text-xs font-medium text-slate-200 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
          {feature.tooltip}
        </span>

        <div className="aspect-[16/10] w-full">
          {src ? (
            <img
              src={src}
              alt={alt || feature.title}
              onLoad={() => setLoaded(true)}
              className={`h-full w-full bg-slate-950 object-contain transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
              No output yet
            </div>
          )}
        </div>
      </div>
    </article>
  );
}

function FeatureAnalysis({ elaImage, fftImage, edgeImage, gradcamImage }) {
  const imageMap = {
    ela: elaImage,
    fft: fftImage,
    edges: edgeImage,
    gradcam: gradcamImage,
  };

  return (
    <section className="mx-auto w-full max-w-6xl space-y-6 p-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {FEATURES.map((feature) => (
          <FeatureCard
            key={feature.key}
            feature={feature}
            src={imageMap[feature.key]}
            alt={`${feature.title} output`}
          />
        ))}
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-slate-900/80 shadow-xl">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-800/80 text-slate-200">
            <tr>
              <th className="px-4 py-3 font-semibold">Feature</th>
              <th className="px-4 py-3 font-semibold">Icon</th>
              <th className="px-4 py-3 font-semibold">What It Reveals</th>
            </tr>
          </thead>
          <tbody>
            {FEATURES.map((feature) => (
              <tr key={feature.key} className="border-t border-slate-800 text-slate-300">
                <td className="px-4 py-3 font-medium">{feature.short}</td>
                <td className="px-4 py-3">
                  <span className={`inline-flex h-8 w-8 items-center justify-center rounded-lg ${feature.bg} ${feature.accent}`}>
                    {feature.icon}
                  </span>
                </td>
                <td className="px-4 py-3">{feature.reveal}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default FeatureAnalysis;