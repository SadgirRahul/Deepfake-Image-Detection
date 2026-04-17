import app from './app.js';
import { loadEnv } from './config/env.js';
import { mkdirSync } from 'fs';
import http from 'http';

const env = loadEnv();

mkdirSync(env.UPLOAD_DIR, { recursive: true });

const server = http.createServer(app);

function probeHealth(port) {
  return new Promise((resolve) => {
    const req = http.request(
      {
        hostname: 'localhost',
        port,
        path: '/health',
        method: 'GET',
        headers: { accept: 'application/json' },
        timeout: 800,
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk.toString();
        });
        res.on('end', () => {
          try {
            const payload = JSON.parse(data);
            resolve(res.statusCode === 200 && payload?.status === 'ok');
          } catch {
            resolve(false);
          }
        });
      },
    );

    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });

    req.on('error', () => {
      resolve(false);
    });

    req.end();
  });
}

server.on('error', async (err) => {
  if (err?.code === 'EADDRINUSE') {
    const isOurServer = await probeHealth(env.PORT);
    if (isOurServer) {
      console.log(`\nℹ️  Port ${env.PORT} is already in use (server already running).`);
      console.log('   Reuse the existing backend terminal, or stop it before starting a new one.\n');
      process.exitCode = 0;
      server.unref();
      return;
    }

    console.error(`\n❌ Port ${env.PORT} is already in use by another process.`);
    console.error('   Close the process using it, or set a different PORT.');
    console.error('   PowerShell example: $env:PORT=5002; npm run dev\n');
    process.exitCode = 1;
    server.unref();
    return;
  }

  console.error(err);
  process.exitCode = 1;
  server.unref();
});

server.listen(env.PORT, () => {
  console.log(`Server running on http://localhost:${env.PORT}`);
});
