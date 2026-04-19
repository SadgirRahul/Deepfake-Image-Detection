import { useEffect, useMemo, useRef } from 'react';

const normalizeOriginal = (src) => {
  if (!src) {
    return '';
  }

  if (
    src.startsWith('data:')
    || src.startsWith('http://')
    || src.startsWith('https://')
    || src.startsWith('blob:')
  ) {
    return src;
  }

  return `data:image/jpeg;base64,${src}`;
};

const normalizeGradcam = (src) => {
  if (!src) {
    return '';
  }

  if (
    src.startsWith('data:')
    || src.startsWith('http://')
    || src.startsWith('https://')
    || src.startsWith('blob:')
  ) {
    return src;
  }

  return `data:image/png;base64,${src}`;
};

function ManipulationHeatmap({ originalImage, gradcamImage, label }) {
  const canvasRef = useRef(null);
  const normalizedOriginal = useMemo(() => normalizeOriginal(originalImage), [originalImage]);
  const normalizedGradcam = useMemo(() => normalizeGradcam(gradcamImage), [gradcamImage]);
  const isFake = String(label || '').toLowerCase() === 'fake';

  useEffect(() => {
    if (!normalizedOriginal || !normalizedGradcam) {
      return undefined;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return undefined;
    }

    let cancelled = false;

    const img1 = new Image();
    const img2 = new Image();

    img1.crossOrigin = 'anonymous';
    img2.crossOrigin = 'anonymous';

    img1.onload = () => {
      if (cancelled) {
        return;
      }

      canvas.width = img1.naturalWidth || 224;
      canvas.height = img1.naturalHeight || 224;

      context.drawImage(img1, 0, 0, canvas.width, canvas.height);

      img2.onload = () => {
        if (cancelled) {
          return;
        }

        context.globalAlpha = 0.55;
        context.drawImage(img2, 0, 0, canvas.width, canvas.height);
        context.globalAlpha = 1.0;
      };

      img2.onerror = (err) => {
        // eslint-disable-next-line no-console
        console.error('gradcam load error', err);
      };

      img2.src = normalizedGradcam;
    };

    img1.onerror = (err) => {
      // eslint-disable-next-line no-console
      console.error('original load error', err);
    };

    img1.src = normalizedOriginal;

    return () => {
      cancelled = true;
    };
  }, [normalizedGradcam, normalizedOriginal]);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <div className="w-full overflow-hidden rounded-xl bg-[#0f172a]">
            <div className="aspect-square w-full">
              {normalizedOriginal ? (
                <img
                  src={normalizedOriginal}
                  alt="Original upload"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
                  No image available
                </div>
              )}
            </div>
          </div>
          <p className="mt-2 text-center text-sm text-slate-300">Original</p>
        </div>

        <div>
          <div className="w-full overflow-hidden rounded-xl bg-[#0f172a]">
            <div className="aspect-square w-full">
              {normalizedGradcam ? (
                <img
                  src={normalizedGradcam}
                  alt="Grad-CAM"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
                  No heatmap available
                </div>
              )}
            </div>
          </div>
          <p className="mt-2 text-center text-sm text-slate-300">Grad-CAM</p>
        </div>

        <div>
          <div className="w-full overflow-hidden rounded-xl bg-[#0f172a]">
            <div className="flex aspect-square w-full items-center justify-center">
              {normalizedOriginal && normalizedGradcam ? (
                <canvas
                  ref={canvasRef}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    borderRadius: '8px',
                    display: 'block',
                  }}
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
                  No overlay available
                </div>
              )}
            </div>
          </div>
          <p className="mt-2 text-center text-sm text-slate-300">Manipulation Map</p>
        </div>
      </div>

      <div>
        <div className="h-2 w-full rounded-full bg-[linear-gradient(90deg,#3b82f6_0%,#22c55e_35%,#facc15_65%,#ef4444_100%)]" />
        <div className="mt-2 flex items-center justify-between text-xs text-slate-300">
          <span>Low attention</span>
          <span>High attention / likely manipulated</span>
        </div>
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-950/70 px-4 py-3 text-sm text-slate-200">
        {isFake
          ? 'Red and yellow regions indicate areas the model identified as AI-generated or manipulated. Focus areas: nose bridge, skin texture, eye corners.'
          : 'No significant manipulation detected. Attention is distributed naturally across the image.'}
      </div>
    </div>
  );
}

export default ManipulationHeatmap;
