import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { useAnalyzeImage } from '../hooks/useAnalyzeImage';

const AnalysisContext = createContext(null);

export function AnalysisProvider({ children }) {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const {
    analyze,
    loading,
    result,
    error,
    reset: resetAnalysis,
  } = useAnalyzeImage();

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl(null);
      return undefined;
    }

    const nextUrl = URL.createObjectURL(imageFile);
    setPreviewUrl(nextUrl);

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [imageFile]);

  const selectFile = useCallback((file) => {
    if (!file || !file.type?.startsWith('image/')) {
      return;
    }

    setImageFile(file);
    resetAnalysis();
  }, [resetAnalysis]);

  const analyzeSelected = useCallback(async () => {
    if (!imageFile) {
      return null;
    }

    return analyze(imageFile);
  }, [analyze, imageFile]);

  const clearAll = useCallback(() => {
    setImageFile(null);
    resetAnalysis();
  }, [resetAnalysis]);

  const value = useMemo(() => (
    {
      imageFile,
      previewUrl,
      selectFile,
      analyzeSelected,
      loading,
      result,
      error,
      clearAll,
    }
  ), [analyzeSelected, clearAll, error, imageFile, loading, previewUrl, result, selectFile]);

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}
