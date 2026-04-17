import { useCallback, useEffect, useRef, useState } from 'react';

import { useAnalysis } from '../context/analysisContext';
import FeatureAnalysis from './FeatureAnalysis';

function AnalyzeSection() {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [inputMode, setInputMode] = useState('upload');
  const [cameraError, setCameraError] = useState('');
  const {
    imageFile,
    previewUrl,
    selectFile,
    analyzeSelected,
    loading,
    result,
    error,
    clearAll,
  } = useAnalysis();

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

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0];
    selectFile(droppedFile);
  };

  const onAnalyze = async () => {
    if (!imageFile) {
      return;
    }

    await analyzeSelected();
  };

  const onReset = () => {
    clearAll();
    setCameraError('');
    setIsDragging(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const predictionLabel = result?.label || '--';
  const realPct = result?.real_pct ?? 0;
  const fakePct = result?.fake_pct ?? 0;
  const confidence = result?.confidence ?? 0;
  const inferenceTime = Number(result?.inferenceTime ?? 0).toFixed(1);

  const onSelectUpload = () => {
    stopCamera();
    setCameraError('');
    setInputMode('upload');
  };

  const onSelectCamera = () => {
    clearAll();
    setCameraError('');
    setInputMode('camera');
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

    selectFile(capturedFile);
    onSelectUpload();
  };

  return (
    <>
      <section className="mx-auto grid w-full max-w-6xl gap-6 p-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-700/60 bg-slate-900/80 p-5 shadow-xl">
          <div className="mb-4 grid grid-cols-2 gap-2 rounded-xl border border-slate-700/60 bg-slate-950/40 p-1">
            <button
              type="button"
              onClick={onSelectUpload}
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${
                inputMode === 'upload'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-300 hover:bg-slate-800/70 hover:text-slate-100'
              }`}
            >
              Upload image
            </button>
            <button
              type="button"
              onClick={onSelectCamera}
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${
                inputMode === 'camera'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-300 hover:bg-slate-800/70 hover:text-slate-100'
              }`}
            >
              Use webcam
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
              {!previewUrl ? (
                <div className="flex flex-col items-center gap-3 text-center">
                  <span className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 text-blue-400">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      className="h-6 w-6"
                      aria-hidden="true"
                    >
                      <path d="M12 16V4" />
                      <path d="m7 9 5-5 5 5" />
                      <path d="M20 16.5a3.5 3.5 0 0 1-3.5 3.5h-9A3.5 3.5 0 0 1 4 16.5" />
                    </svg>
                  </span>
                  <p className="text-sm font-medium text-slate-200">Drop image here or click to browse</p>
                </div>
              ) : (
                <img
                  src={previewUrl}
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
            onChange={(event) => selectFile(event.target.files?.[0])}
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
            onClick={onAnalyze}
            disabled={!imageFile || loading}
            className="mt-4 w-full rounded-xl bg-gradient-to-r from-blue-500 to-cyan-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Analyze Image
          </button>

          <button
            type="button"
            onClick={onReset}
            disabled={loading || (!imageFile && !result && !error)}
            className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-800/50 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-slate-800/80 disabled:cursor-not-allowed disabled:opacity-50"
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
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-center text-5xl font-extrabold tracking-wide text-white">{predictionLabel}</p>

            <div className="mt-8 space-y-5">
              <div>
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-slate-200">Real</span>
                  <span className="text-slate-300">{realPct}%</span>
                </div>
                <div className="h-3 rounded-full bg-slate-800">
                  <div
                    className="h-3 rounded-full bg-emerald-500 transition-all duration-700"
                    style={{ width: `${realPct}%` }}
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
                    style={{ width: `${fakePct}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3">
            <p className="text-sm font-medium text-slate-100">
              Confidence: <span className="text-blue-300">{confidence}%</span>
              <span className="mx-2 text-slate-500">|</span>
              Inference: <span className="text-cyan-300">{inferenceTime} ms</span>
            </p>
          </div>
        </div>
      </section>

      <FeatureAnalysis
        elaImage={result?.ela}
        fftImage={result?.fft}
        edgeImage={result?.edges}
        gradcamImage={result?.gradcam}
      />
    </>
  );
}

export default AnalyzeSection;