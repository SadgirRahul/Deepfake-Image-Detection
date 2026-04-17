export function errorHandler(err, req, res, next) {
  const statusCode = err.statusCode || 500;
  const isMulterLimit = err?.code === 'LIMIT_FILE_SIZE';

  if (isMulterLimit) {
    res.status(400).json({
      message: 'Uploaded file is too large. Max allowed size is 10MB.',
    });
    return;
  }

  res.status(statusCode).json({
    message: err.message || 'Internal server error',
    ...(process.env.NODE_ENV !== 'production' ? { details: err.stack } : {}),
  });
}
