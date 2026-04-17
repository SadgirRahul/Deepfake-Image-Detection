import { spawn } from 'child_process';
import { unlink } from 'fs/promises';

import { loadEnv } from '../config/env.js';

const env = loadEnv();

async function safeDelete(filePath) {
  if (!filePath) {
    return;
  }

  try {
    await unlink(filePath);
  } catch {
    // no-op
  }
}

function runPredictScript(imagePath) {
  return new Promise((resolve, reject) => {
    const processArgs = [env.PYTHON_SCRIPT, imagePath];
    const pythonProcess = spawn(env.PYTHON_EXECUTABLE, processArgs, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        // Reduce BLAS thread usage to avoid memory allocation failures on constrained hosts.
        OPENBLAS_NUM_THREADS: '1',
        OMP_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
        VECLIB_MAXIMUM_THREADS: '1',
      },
    });

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    pythonProcess.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        const scriptError = new Error(`Prediction script failed with code ${code}: ${stderrData || 'Unknown error'}`);
        scriptError.statusCode = 500;
        reject(scriptError);
        return;
      }

      try {
        const parsed = JSON.parse(stdoutData.trim());
        resolve(parsed);
      } catch {
        const parseError = new Error('Prediction script returned invalid JSON');
        parseError.statusCode = 500;
        reject(parseError);
      }
    });
  });
}

export async function analyzeImage(req, res, next) {
  const uploadedFilePath = req.file?.path;

  if (!uploadedFilePath) {
    const missingFileError = new Error('Image file is required under field name "image"');
    missingFileError.statusCode = 400;
    next(missingFileError);
    return;
  }

  try {
    const response = await runPredictScript(uploadedFilePath);
    res.status(200).json(response);
  } catch (err) {
    next(err);
  } finally {
    await safeDelete(uploadedFilePath);
  }
}
