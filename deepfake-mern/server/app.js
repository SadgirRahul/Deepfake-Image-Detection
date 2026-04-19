import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { readFile } from 'fs/promises';
import path from 'path';

import { loadEnv } from './config/env.js';
import detectionRoutes from './routes/detection.routes.js';
import { notFound } from './middleware/notFound.js';
import { errorHandler } from './middleware/errorHandler.js';

const app = express();
const env = loadEnv();

const corsOptions = {
  origin(origin, callback) {
    if (!origin) {
      callback(null, true);
      return;
    }

    const allowedExact = new Set([env.CLIENT_URL]);

    if (allowedExact.has(origin)) {
      callback(null, true);
      return;
    }

    if (env.NODE_ENV === 'development') {
      if (origin.startsWith('http://localhost:') || origin.startsWith('http://127.0.0.1:')) {
        callback(null, true);
        return;
      }
    }

    callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
};

app.use(cors({
  ...corsOptions,
}));
app.use(helmet());
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

app.get('/api/model-metrics', async (req, res, next) => {
  try {
    const metricsPath = path.join(process.cwd(), 'server', 'data', 'model_metrics.json');
    const raw = await readFile(metricsPath, 'utf-8');
    const payload = JSON.parse(raw);
    res.status(200).json(payload);
  } catch (err) {
    err.statusCode = err.statusCode || 500;
    next(err);
  }
});

app.use('/api', detectionRoutes);

app.use(notFound);
app.use(errorHandler);

export default app;
