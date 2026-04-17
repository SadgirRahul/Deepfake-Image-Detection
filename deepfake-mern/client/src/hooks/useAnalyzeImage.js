import { useCallback, useState } from 'react';

function normalizePercent(value) {
  const numeric = Number(value);

  if (!Number.isFinite(numeric)) {
    return 0;
  }

  if (numeric <= 1) {
    return Math.round(numeric * 100);
  }

  return Math.round(numeric);
}

function normalizeImageOutput(value) {
  if (!value) {
    return '';
  }

  const trimmed = String(value).trim();

  if (!trimmed) {
    return '';
  }

  if (trimmed.startsWith('data:image') || trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    return trimmed;
  }

  return `data:image/png;base64,${trimmed}`;
}

export function useAnalyzeImage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const apiBaseUrl = (import.meta.env?.VITE_API_BASE_URL || '').toString().replace(/\/$/, '');
  const analyzeUrl = apiBaseUrl ? `${apiBaseUrl}/api/analyze` : '/api/analyze';

  const analyze = useCallback(async (file) => {
    if (!file) {
      setError('Please select an image first.');
      return null;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch(analyzeUrl, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.message || 'Image analysis failed.');
      }

      const label = String(payload?.label || 'UNKNOWN').toUpperCase();
      const confidence = normalizePercent(payload?.confidence);
      const inferenceTime = Number(payload?.inference_time ?? payload?.inferenceTime ?? 0);

      const realPct = payload?.real_pct != null
        ? normalizePercent(payload.real_pct)
        : label === 'REAL'
          ? confidence
          : Math.max(0, 100 - confidence);

      const fakePct = payload?.fake_pct != null
        ? normalizePercent(payload.fake_pct)
        : label === 'FAKE'
          ? confidence
          : Math.max(0, 100 - confidence);

      const normalizedResult = {
        label,
        confidence,
        inferenceTime: Number.isFinite(inferenceTime) ? inferenceTime : 0,
        real_pct: realPct,
        fake_pct: fakePct,
        ela: normalizeImageOutput(payload?.ela),
        fft: normalizeImageOutput(payload?.fft),
        edges: normalizeImageOutput(payload?.edges),
        gradcam: normalizeImageOutput(payload?.gradcam),
      };

      setResult(normalizedResult);
      return normalizedResult;
    } catch (err) {
      setError(err?.message || 'Something went wrong while analyzing this image.');
      setResult(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setLoading(false);
    setError('');
    setResult(null);
  }, []);

  return {
    analyze,
    loading,
    result,
    error,
    reset,
  };
}
