import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const SERVER_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REPO_ROOT = path.resolve(SERVER_ROOT, '..', '..');

dotenv.config({ path: path.resolve(SERVER_ROOT, '.env') });

function resolvePythonExecutable() {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }

  const venvCandidate = process.platform === 'win32'
    ? path.resolve(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.resolve(REPO_ROOT, '.venv', 'bin', 'python');

  if (fs.existsSync(venvCandidate)) {
    return venvCandidate;
  }

  return 'python';
}

export function loadEnv() {
  return {
    PORT: Number(process.env.PORT || 5000),
    NODE_ENV: process.env.NODE_ENV || 'development',
    CLIENT_URL: process.env.CLIENT_URL || 'http://localhost:5173',
    UPLOAD_DIR: path.resolve(SERVER_ROOT, process.env.UPLOAD_DIR || 'uploads'),
    PYTHON_EXECUTABLE: resolvePythonExecutable(),
    PYTHON_SCRIPT: path.resolve(SERVER_ROOT, process.env.PYTHON_SCRIPT || 'model/predict.py'),
  };
}
