import app from './app.js';
import { loadEnv } from './config/env.js';
import { mkdirSync } from 'fs';

const env = loadEnv();

mkdirSync(env.UPLOAD_DIR, { recursive: true });

app.listen(env.PORT, () => {
  console.log(`Server running on http://localhost:${env.PORT}`);
});
