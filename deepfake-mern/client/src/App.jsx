import { useCallback, useEffect, useMemo, useState } from 'react';
import Navbar from './components/Navbar';
import AnalyzeSection from './components/AnalyzeSection';
import AnalysisReport from './components/AnalysisReport';

const emptyResult = {
  label: '',
  confidence: 0,
  real_pct: 0,
  fake_pct: 0,
  inference_time_ms: 0,
  visualizations: {
    ela: null,
    fft: null,
    edges: null,
    gradcam: null,
  },
};


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

function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [result, setResult] = useState(emptyResult);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelMetrics, setModelMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  const analyzeUrl = useMemo(() => {
    const apiBaseUrl = (import.meta.env?.VITE_API_BASE_URL || '').toString().replace(/\/$/, '');
    return apiBaseUrl ? `${apiBaseUrl}/api/analyze` : '/api/analyze';
  }, []);


  useEffect(() => {
    if (!analysisComplete && activeTab === 'report') {
      setActiveTab('analyze');
    }
  }, [activeTab, analysisComplete]);

  useEffect(() => {
    let isMounted = true;

    fetch('http://localhost:5000/api/model-metrics')
      .then((res) => res.json())
      .then((data) => {
        if (!isMounted) {
          return;
        }
        setModelMetrics(data);
        setMetricsLoading(false);
      })
      .catch(() => {
        if (!isMounted) {
          return;
        }
        setMetricsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleImageSelected = useCallback((base64Image) => {
    setUploadedImage(base64Image || null);
    setResult(emptyResult);
    setError('');
    setAnalysisComplete(false);
  }, []);

  const handleAnalyze = useCallback(async (file) => {
    if (!file) {
      setError('Please select an image first.');
      return;
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
      const inferenceTime = Number(payload?.inference_time_ms ?? 0);

      const rawRealPct = payload?.real_pct != null
        ? normalizePercent(payload.real_pct)
        : label === 'REAL'
          ? confidence
          : Math.max(0, 100 - confidence);

      const rawFakePct = payload?.fake_pct != null
        ? normalizePercent(payload.fake_pct)
        : label === 'FAKE'
          ? confidence
          : Math.max(0, 100 - confidence);

      const totalPct = rawRealPct + rawFakePct;
      const normalizedReal = totalPct > 0 ? (rawRealPct / totalPct) * 100 : 0;
      const normalizedFake = totalPct > 0 ? 100 - normalizedReal : 0;

      const realPct = Math.round(normalizedReal * 10) / 10;
      const fakePct = Math.round(normalizedFake * 10) / 10;

      setResult({
        label,
        confidence,
        real_pct: realPct,
        fake_pct: fakePct,
        inference_time_ms: Number.isFinite(inferenceTime) ? inferenceTime : 0,
        visualizations: {
          ela: payload?.visualizations?.ela ?? null,
          fft: payload?.visualizations?.fft ?? null,
          edges: payload?.visualizations?.edges ?? null,
          gradcam: payload?.visualizations?.gradcam ?? null,
        },
      });

      setAnalysisComplete(true);
    } catch (err) {
      setResult(emptyResult);
      setError(err?.message || 'Something went wrong while analyzing this image.');
    } finally {
      setLoading(false);
    }
  }, [analyzeUrl]);

  const handleReset = useCallback(() => {
    setActiveTab('analyze');
    setAnalysisComplete(false);
    setUploadedImage(null);
    setResult(emptyResult);
    setError('');
    setLoading(false);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <Navbar
        activeTab={activeTab}
        analysisComplete={analysisComplete}
        setActiveTab={setActiveTab}
      />
      <main className="pb-10">
        {activeTab === 'analyze' ? (
          <AnalyzeSection
            uploadedImage={uploadedImage}
            loading={loading}
            result={result}
            error={error}
            onImageSelected={handleImageSelected}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
            modelMetrics={modelMetrics}
            metricsLoading={metricsLoading}
            setActiveTab={setActiveTab}
          />
        ) : (
          <AnalysisReport
            result={result}
            uploadedImage={uploadedImage}
            loading={loading}
            modelMetrics={modelMetrics}
            metricsLoading={metricsLoading}
            confidence={result?.confidence}
            label={result?.label}
          />
        )}
      </main>
    </div>
  );
}

export default App;
