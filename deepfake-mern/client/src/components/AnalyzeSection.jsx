import { useCallback, useEffect, useRef, useState } from 'react';
import AccuracyConfidencePanel from './AccuracyConfidencePanel';
import ConfidenceReliability from './ConfidenceReliability';

function AnalyzeSection({
  uploadedImage,
  loading,
  result,
  error,
  onImageSelected,
  onAnalyze,
  onReset,
  modelMetrics,
  metricsLoading,
  setActiveTab,
}) {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [inputMode, setInputMode] = useState('upload');
  const [cameraError, setCameraError] = useState('');
  const [displayedRealPct, setDisplayedRealPct] = useState(0);
  const [displayedFakePct, setDisplayedFakePct] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    if (!result?.label) {
      setDisplayedRealPct(0);
      setDisplayedFakePct(0);
      return;
    }

    setDisplayedRealPct(0);
    setDisplayedFakePct(0);

    const nextReal = result?.real_pct ?? 0;
    const nextFake = result?.fake_pct ?? 0;

    const timeoutId = setTimeout(() => {
      setDisplayedRealPct(nextReal);
      setDisplayedFakePct(nextFake);
    }, 80);

    return () => clearTimeout(timeoutId);
  }, [result]);

  useEffect(() => {
    if (!uploadedImage) {
      setSelectedFile(null);
    }
  }, [uploadedImage]);

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    if (inputMode !== 'camera') {
      stopCamera();
      setCameraError('');
      return undefined;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Webcam is not supported in this browser.');
      return undefined;
    }

    let cancelled = false;

    const startCamera = async () => {
      try {
        setCameraError('');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      } catch (err) {
        stopCamera();
        setCameraError(err?.message || 'Unable to access webcam.');
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [inputMode, stopCamera]);

  const handleFileSelection = (file) => {
    if (!file || !file.type?.startsWith('image/')) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setSelectedFile(file);
      onImageSelected(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0];
    handleFileSelection(droppedFile);
  };

  const handleAnalyzeClick = async () => {
    if (!selectedFile) {
      return;
    }

    await onAnalyze(selectedFile);
  };

  const handleReset = () => {
    onReset();
    setCameraError('');
    setIsDragging(false);
    setDisplayedRealPct(0);
    setDisplayedFakePct(0);
    setSelectedFile(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const predictionLabel = result?.label || '--';
  const realPct = result?.real_pct ?? 0;
  const fakePct = result?.fake_pct ?? 0;
  const confidence = result?.confidence ?? 0;
  const inferenceTime = Number(result?.inference_time_ms ?? 0).toFixed(1);

  const onSelectUpload = () => {
    stopCamera();
    setCameraError('');
    setInputMode('upload');
  };

  const onSelectCamera = () => {
    handleReset();
    setCameraError('');
    setInputMode('camera');
  };

  const onDownloadReport = () => {
    // Placeholder until report export is implemented.
  };

  const onCapture = async () => {
    if (!videoRef.current) {
      return;
    }

    const video = videoRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      setCameraError('Unable to capture image.');
      return;
    }

    ctx.drawImage(video, 0, 0, width, height);

    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.92);
    });

    if (!blob) {
      setCameraError('Unable to capture image.');
      return;
    }

    const capturedFile = new File([blob], `webcam-${Date.now()}.jpg`, {
      type: blob.type || 'image/jpeg',
    });

    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    setSelectedFile(capturedFile);
    onImageSelected(dataUrl);
    onSelectUpload();
  };

  return (
    <>
      <section className="mx-auto grid w-full max-w-6xl gap-6 bg-[#0f172a] p-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-700/60 bg-slate-900/80 p-5 shadow-xl">
          <div className="mb-4 flex items-center rounded-full border border-slate-700/60 bg-slate-950/60 p-1">
            <button
              type="button"
              onClick={onSelectUpload}
              className={`flex-1 rounded-full px-4 py-2 text-sm font-semibold transition-colors ${
                inputMode === 'upload'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-slate-300 hover:text-slate-100'
              }`}
            >
              Upload Image
            </button>
            <button
              type="button"
              onClick={onSelectCamera}
              className={`flex-1 rounded-full px-4 py-2 text-sm font-semibold transition-colors ${
                inputMode === 'camera'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-slate-300 hover:text-slate-100'
              }`}
            >
              Use Webcam
            </button>
          </div>

          {inputMode === 'upload' ? (
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              className={`flex min-h-[320px] cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed p-4 transition-colors ${
                isDragging ? 'border-blue-400 bg-slate-800/60' : 'border-slate-600 hover:border-blue-500/70'
              }`}
            >
              {!uploadedImage ? (
                <span className="inline-flex h-14 w-14 items-center justify-center rounded-full bg-slate-800 text-blue-400">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    className="h-7 w-7"
                    aria-hidden="true"
                  >
                    <path d="M12 16V4" />
                    <path d="m7 9 5-5 5 5" />
                    <path d="M20 16.5a3.5 3.5 0 0 1-3.5 3.5h-9A3.5 3.5 0 0 1 4 16.5" />
                  </svg>
                </span>
              ) : (
                <img
                  src={uploadedImage}
                  alt="Selected preview"
                  className="max-h-[300px] w-full rounded-xl bg-slate-950 object-contain"
                />
              )}
            </div>
          ) : (
            <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-slate-700/60 bg-slate-950 p-4">
              {cameraError ? (
                <p className="text-center text-sm text-rose-300">{cameraError}</p>
              ) : (
                <video
                  ref={videoRef}
                  className="max-h-[300px] w-full rounded-xl bg-slate-950 object-contain"
                  playsInline
                  muted
                  autoPlay
                />
              )}
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => handleFileSelection(event.target.files?.[0])}
          />

          {inputMode === 'camera' && (
            <button
              type="button"
              onClick={onCapture}
              disabled={loading || !!cameraError}
              className="mt-4 w-full rounded-xl border border-slate-700 bg-slate-800/70 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Capture photo
            </button>
          )}

          <button
            type="button"
            onClick={handleAnalyzeClick}
            disabled={!selectedFile || loading}
            className="mt-4 w-full rounded-xl bg-gradient-to-r from-blue-500 to-cyan-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Analyze Image
          </button>

          <button
            type="button"
            onClick={handleReset}
            disabled={loading || (!uploadedImage && !result?.label && !error)}
            className="mt-3 w-full rounded-xl border border-slate-700 bg-transparent px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-slate-800/70 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Reset
          </button>

          {loading && (
            <div className="mt-4 flex items-center justify-center gap-2 text-sm text-blue-300">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-300 border-t-transparent" />
              <span>Analyzing image...</span>
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">
              {error}
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-slate-700/60 bg-slate-900/80 p-5 shadow-xl">
          {!result?.label ? (
            <div className="flex min-h-[360px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 p-6 text-center text-sm text-slate-300">
              Upload an image to begin
            </div>
          ) : (
            <div className="space-y-6 rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <p
                className={`text-center text-5xl font-extrabold tracking-wide ${
                  predictionLabel.toLowerCase() === 'real' ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {predictionLabel}
              </p>

              <div className="flex justify-center">
                <ConfidenceReliability
                  confidence={confidence}
                  modelAccuracy={modelMetrics?.overall_accuracy}
                  metricsLoading={metricsLoading}
                />
              </div>

              <div className="space-y-5">
                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-200">Real</span>
                    <span className="text-slate-300">{realPct}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-slate-800">
                    <div
                      className="h-3 rounded-full bg-emerald-500 transition-all duration-700"
                      style={{ width: `${displayedRealPct}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-slate-200">Fake</span>
                    <span className="text-slate-300">{fakePct}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-slate-800">
                    <div
                      className="h-3 rounded-full bg-rose-500 transition-all duration-700"
                      style={{ width: `${displayedFakePct}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3 text-sm font-medium text-slate-100">
                Confidence: <span className="text-emerald-300">{confidence}%</span>
                <span className="mx-2 text-slate-500">|</span>
                Inference: <span className="text-cyan-300">{inferenceTime} ms</span>
              </div>

                <AccuracyConfidencePanel
                  confidence={confidence}
                  label={predictionLabel}
                  modelMetrics={modelMetrics}
                  metricsLoading={metricsLoading}
                />

              <div className="grid gap-3 md:grid-cols-3">
                <button
                  type="button"
                  onClick={() => setActiveTab('report')}
                  className="w-full rounded-xl bg-blue-500/20 px-3 py-2 text-sm font-semibold text-blue-200 transition hover:bg-blue-500/30"
                >
                  View Analysis Report
                </button>
                <button
                  type="button"
                  onClick={onDownloadReport}
                  className="w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-800"
                >
                  Download Report
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-semibold text-slate-200 transition hover:bg-slate-800"
                >
                  Reset
                </button>
              </div>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

export default AnalyzeSection;