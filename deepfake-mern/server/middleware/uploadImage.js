import multer from 'multer';
import path from 'path';
import crypto from 'crypto';

import { loadEnv } from '../config/env.js';

const env = loadEnv();

const storage = multer.diskStorage({
  destination: (_, __, cb) => {
    cb(null, env.UPLOAD_DIR);
  },
  filename: (_, file, cb) => {
    const extension = path.extname(file.originalname) || '.jpg';
    const uniqueName = `${Date.now()}-${crypto.randomUUID()}${extension}`;
    cb(null, uniqueName);
  },
});

function fileFilter(_, file, cb) {
  if (file.mimetype?.startsWith('image/')) {
    cb(null, true);
    return;
  }

  const invalidTypeError = new Error('Only image uploads are supported');
  invalidTypeError.statusCode = 400;
  cb(invalidTypeError);
}

export const uploadImage = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
});