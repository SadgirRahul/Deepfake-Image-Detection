import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

export function loadEnv() {
  return {
    PORT: Number(process.env.PORT || 5000),
    NODE_ENV: process.env.NODE_ENV || 'development',
    CLIENT_URL: process.env.CLIENT_URL || 'http://localhost:5173',
    UPLOAD_DIR: path.resolve(process.cwd(), process.env.UPLOAD_DIR || 'uploads'),
    PYTHON_EXECUTABLE: process.env.PYTHON_EXECUTABLE || 'python',
    PYTHON_SCRIPT: path.resolve(process.cwd(), process.env.PYTHON_SCRIPT || 'model/predict.py'),
  };
}
