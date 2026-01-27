import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';
import { log } from './logger';
import { spawn, type ChildProcess } from 'node:child_process';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, writeFileSync, unlinkSync, mkdirSync } from 'node:fs';

// Silence transformers.js internal console warnings
const origWarn = console.warn;
console.warn = (...args: unknown[]) => {
  const msg = String(args[0] || '');
  if (msg.includes('Unable to determine') || msg.includes('context size')) return;
  origWarn.apply(console, args);
};

// Allow both local cache and remote
env.allowLocalModels = true;
env.allowRemoteModels = true;

let extractor: FeatureExtractionPipeline | null = null;
let loading: Promise<FeatureExtractionPipeline> | null = null;
let failed = false;

// Xenova's public fork of Jina code embeddings (no auth required)
const MODELS = [
  { name: 'Xenova/jina-embeddings-v2-small-en', dims: 512 },
  { name: 'Xenova/all-MiniLM-L6-v2', dims: 384 },
];

let activeDims: number | null = null;
let providerLogged = false;

// SINGLE CONFIG VARIABLE
const isGpuEnabled = process.env.OPENCODE_CODESEARCH_GPU === 'true';

const gpuUrl = 'http://127.0.0.1:17373';
const gpuTimeoutMs = 30000; // Allow time for model loading
const gpuHealthCacheMs = 15000;

let gpuChild: ChildProcess | null = null;
let gpuSpawned = false;
let gpuStarting = false;
let gpuLastCheck = 0;
let gpuDevice: string | null = null;
let gpuDepsChecked = false;
let gpuDepsReady = false;
let venvPythonPath: string | null = null;

const home = process.env.HOME || process.env.USERPROFILE || process.cwd();
const VENV_DIR = path.resolve(home, '.local', 'share', 'opencode', 'storage', 'plugin', 'codesearch', 'gpu', 'venv');

export async function getExtractor(): Promise<FeatureExtractionPipeline | null> {
  if (failed) return null;
  if (extractor) return extractor;
  if (loading) return loading;

  for (const model of MODELS) {
    if (activeDims !== null && model.dims !== activeDims) {
      log.debug(`Skipping local model ${model.name} due to dimension mismatch (expected ${activeDims}, got ${model.dims})`);
      continue;
    }
    log.info(`Trying model: ${model.name}`);
    try {
      loading = pipeline('feature-extraction', model.name, {
        quantized: true,
        progress_callback: (progress: { status: string; progress?: number }) => {
          if (progress.status === 'downloading' && progress.progress) {
            log.debug(`Downloading: ${Math.round(progress.progress)}%`);
          }
        },
      });
      extractor = await loading;
      if (activeDims === null) {
        activeDims = model.dims;
      }
      log.info(`Model loaded: ${model.name} (${model.dims} dims)`);
      loading = null;
      return extractor;
    } catch (err) {
      log.warn(`Failed to load ${model.name}: ${(err as Error).message}`);
      loading = null;
    }
  }

  failed = true;
  log.error('All models failed to load');
  return null;
}

export async function embed(text: string): Promise<number[] | null> {
  const vectors = await embedMany([text]);
  return vectors[0] ?? null;
}

export async function embedMany(texts: string[]): Promise<Array<number[] | null>> {
  if (texts.length === 0) return [];

  if (await shouldUseGpu()) {
    const remote = await embedGpu(texts);
    if (remote) return remote;
    log.warn('GPU embeddings failed, falling back to local model');
  }

  return embedLocal(texts);
}

async function embedLocal(texts: string[]): Promise<Array<number[] | null>> {
  const ext = await getExtractor();
  if (!ext) return texts.map(() => null);

  try {
    const output = await ext(texts, { pooling: 'mean', normalize: true });
    const data = output.data as Float32Array;
    const results: Array<number[] | null> = [];
    const dims = data.length / texts.length;

    for (let i = 0; i < texts.length; i++) {
      const start = i * dims;
      const slice = data.slice(start, start + dims);
      results.push(Array.from(slice));
    }

    return results;
  } catch (err) {
    log.error('Embedding failed', (err as Error).message);
    return texts.map(() => null);
  }
}

async function embedGpu(texts: string[]): Promise<Array<number[] | null> | null> {
  const response = await fetchJson(`${gpuUrl}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: texts }),
  });

  if (!response) return null;
  const embeddings = Array.isArray(response.embeddings) ? response.embeddings : [];
  if (embeddings.length === 0) return null;

  const dims = Number.isFinite(response.dims) ? response.dims : embeddings[0]?.length;
  if (activeDims === null && dims) {
    activeDims = dims;
  } else if (activeDims !== null && dims && dims !== activeDims) {
    log.warn(`GPU embedding dimension mismatch: expected ${activeDims}, got ${dims}`);
    return null;
  }

  if (!providerLogged) {
    log.info(`Using local GPU embeddings: ${response.device || 'unknown'} (${gpuUrl})`);
    providerLogged = true;
  }

  return embeddings;
}

async function shouldUseGpu(): Promise<boolean> {
  if (!isGpuEnabled) return false;
  const status = await getGpuStatus();
  if (!status) return false;
  return true;
}

async function getGpuStatus(): Promise<{ device?: string; dims?: number } | null> {
  const now = Date.now();
  if (gpuDevice && now - gpuLastCheck < gpuHealthCacheMs) {
    return { device: gpuDevice };
  }

  await maybeStartGpuServer();

  if (gpuStarting) {
    // Wait for server to become healthy (up to 60s for model loading)
    const start = Date.now();
    while (Date.now() - start < 60000) {
      const response = await fetchJson(`${gpuUrl}/health`, { method: 'GET' }, true);
      if (response) {
        gpuStarting = false;
        gpuDevice = response.device || 'unknown';
        gpuLastCheck = Date.now();
        log.info(`GPU server ready: ${gpuDevice} (${response.model})`);
        return response;
      }
      if (!gpuChild) {
        gpuStarting = false;
        return null;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    gpuStarting = false;
    log.warn('GPU server failed to become healthy within 60s');
    return null;
  }

  const response = await fetchJson(`${gpuUrl}/health`, { method: 'GET' });
  if (!response) {
    // If we can't reach a "ready" server, back off for a bit
    gpuLastCheck = now;
    return null;
  }

  if (typeof response.device === 'string') {
    gpuDevice = response.device;
  }
  gpuLastCheck = now;
  return response;
}

async function maybeStartGpuServer(): Promise<void> {
  if (!isGpuEnabled || gpuSpawned || gpuStarting) return;

  // Check if someone is already listening on this port
  const existing = await fetchJson(`${gpuUrl}/health`, { method: 'GET' }, true);
  if (existing) {
    log.info(`GPU server already running: ${existing.device} (${existing.model})`);
    gpuSpawned = true;
    gpuDevice = existing.device;
    if (activeDims === null && existing.dims) activeDims = existing.dims;
    return;
  }

  const scriptPath = resolveGpuServerPath();
  if (!scriptPath) return;

  const python = await ensureGpuDeps(scriptPath);
  if (!python) return;

  let host = '127.0.0.1';
  let port = '17373';
  try {
    const parsed = new URL(gpuUrl);
    if (parsed.hostname) host = parsed.hostname;
    if (parsed.port) port = parsed.port;
  } catch {
    // ignore invalid URL
  }

  gpuStarting = true;
  gpuChild = spawn(python, [scriptPath, '--host', host, '--port', port], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  gpuChild.stdout?.on('data', (chunk) => {
    const lines = chunk.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) log.info(`[gpu-server] ${line.trim()}`);
    }
  });

  gpuChild.stderr?.on('data', (chunk) => {
    const lines = chunk.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) log.info(`[gpu-server-err] ${line.trim()}`);
    }
  });

  gpuChild.on('error', (err) => {
    gpuStarting = false;
    gpuSpawned = false;
    gpuChild = null;
    log.warn(`GPU server failed to start: ${err.message}`);
  });

  gpuChild.on('exit', (code) => {
    gpuStarting = false;
    if (gpuSpawned && code !== 0 && code !== null) {
      log.warn(`GPU server exited unexpectedly with code ${code}`);
    }
    gpuSpawned = false;
    gpuChild = null;
  });

  gpuSpawned = true;
  // Keep gpuStarting = true, getGpuStatus will flip it after successful health check or timeout
}

async function ensureGpuDeps(scriptPath: string): Promise<string | null> {
  if (gpuDepsChecked) return gpuDepsReady ? (venvPythonPath || 'python3') : null;
  gpuDepsChecked = true;

  const systemCheck = await runPython('python3', ['-c', 'import fastapi,uvicorn,sentence_transformers,torch']);
  if (systemCheck.code === 0) {
    gpuDepsReady = true;
    return 'python3';
  }

  if (!isGpuEnabled) {
    log.warn('GPU deps missing; set OPENCODE_CODESEARCH_GPU=true to enable GPU support');
    gpuDepsReady = false;
    return null;
  }

  const pythonExec = process.platform === 'win32' ? path.join(VENV_DIR, 'Scripts', 'python.exe') : path.join(VENV_DIR, 'bin', 'python3');
  
  if (!existsSync(VENV_DIR)) {
    log.info(`Creating central virtual environment in ${VENV_DIR}...`);
    const parent = path.dirname(VENV_DIR);
    if (!existsSync(parent)) mkdirSync(parent, { recursive: true });
    
    const venvResult = await runPython('python3', ['-m', 'venv', VENV_DIR, '--without-pip']);
    if (venvResult.code !== 0) {
      log.warn(`Failed to create venv: ${venvResult.output}`);
      return null;
    }
  }
  
  venvPythonPath = pythonExec;

  const pipCheck = await runPython(pythonExec, ['-m', 'pip', '--version']);
  if (pipCheck.code !== 0) {
    log.info('Bootstrapping pip in virtual environment...');
    const bootstrapped = await bootstrapPip(pythonExec);
    if (!bootstrapped) return null;
  }

  const requirements = path.resolve(path.dirname(scriptPath), 'requirements-gpu.txt');
  log.info('Installing GPU embedding dependencies in venv (this can take 5-10 minutes)...');
  const install = await runPython(
    pythonExec,
    ['-m', 'pip', 'install', '-r', requirements],
    true,
    { PYO3_USE_ABI3_FORWARD_COMPATIBILITY: '1' }
  );
  
  if (install.code !== 0) {
    log.warn(`GPU deps install failed (exit ${install.code}): ${install.output.slice(0, 500)}`);
    return null;
  }

  const finalCheck = await runPython(pythonExec, ['-c', 'import fastapi,uvicorn,sentence_transformers,torch']);
  gpuDepsReady = finalCheck.code === 0;
  return gpuDepsReady ? pythonExec : null;
}

async function bootstrapPip(pythonExec: string): Promise<boolean> {
  const getPipUrl = 'https://bootstrap.pypa.io/get-pip.py';
  const tempPath = path.resolve(VENV_DIR, 'get-pip-temp.py');

  try {
    if (!existsSync(VENV_DIR)) mkdirSync(VENV_DIR, { recursive: true });
    log.info('Downloading get-pip.py...');
    const response = await fetch(getPipUrl);
    if (!response.ok) throw new Error(`Download failed: ${response.statusText}`);
    const content = await response.text();
    writeFileSync(tempPath, content);

    log.info('Running get-pip.py in venv...');
    const result = await runPython(pythonExec, [tempPath]);
    if (existsSync(tempPath)) unlinkSync(tempPath);

    if (result.code !== 0) {
      log.warn(`get-pip.py failed (exit ${result.code}): ${result.output.slice(0, 500)}`);
      return false;
    }

    return true;
  } catch (err) {
    log.warn(`Failed to bootstrap pip: ${(err as Error).message}`);
    if (existsSync(tempPath)) unlinkSync(tempPath);
    return false;
  }
}

async function runPython(
  pythonExec: string,
  args: string[],
  streamToLog = false,
  envVars: Record<string, string> = {}
): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn(pythonExec, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, ...envVars },
    });
    let output = '';

    child.stdout.on('data', (chunk) => {
      const str = chunk.toString();
      output += str;
      if (streamToLog) {
        const lines = str.split('\n');
        for (const line of lines) {
          if (line.trim()) log.info(`[pip] ${line.trim()}`);
        }
      }
    });

    child.stderr.on('data', (chunk) => {
      const str = chunk.toString();
      output += str;
      if (streamToLog) {
        const lines = str.split('\n');
        for (const line of lines) {
          if (line.trim()) log.info(`[pip-err] ${line.trim()}`);
        }
      }
    });

    child.on('error', (err) => {
      resolve({ code: 127, output: err.message });
    });

    child.on('exit', (code) => {
      resolve({ code: code ?? 1, output: output.trim() });
    });
  });
}

function resolveGpuServerPath(): string | null {
  try {
    const here = path.dirname(fileURLToPath(import.meta.url));
    return path.resolve(here, '..', 'scripts', 'gpu_embeddings_server.py');
  } catch (err) {
    log.warn('Unable to resolve GPU server script path', (err as Error).message);
    return null;
  }
}

async function fetchJson(
  url: string,
  options: RequestInit,
  silent = false
): Promise<Record<string, any> | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), gpuTimeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) return null;
    return (await response.json()) as Record<string, any>;
  } catch (err) {
    if (isGpuEnabled && !silent) {
      log.warn('GPU embeddings request failed', (err as Error).message);
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

// Cleanup: ensure the child process is killed when the main process exits
const cleanup = () => {
  if (gpuChild) {
    log.info('Shutting down GPU embedding server...');
    gpuChild.kill();
    gpuChild = null;
  }
};

process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(); });
process.on('SIGTERM', () => { cleanup(); process.exit(); });

export async function prepare(): Promise<void> {
  if (isGpuEnabled) {
    await getGpuStatus();
  } else {
    await getExtractor();
  }
}

export function getDims(): number {
  return activeDims ?? 512;
}
