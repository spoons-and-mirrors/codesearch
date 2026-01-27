import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';
import { log } from './logger';
import { spawn } from 'node:child_process';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

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
// Falls back to MiniLM if this fails
const MODELS = [
  { name: 'Xenova/jina-embeddings-v2-small-en', dims: 512 },
  { name: 'Xenova/all-MiniLM-L6-v2', dims: 384 },
];

let activeDims = 512;
let providerLogged = false;

type ProviderPreference = 'auto' | 'local' | 'local-gpu';

const providerPreference = (process.env.CODESEARCH_EMBEDDINGS_PROVIDER || 'auto')
  .toLowerCase()
  .trim() as ProviderPreference;
const gpuUrl = process.env.CODESEARCH_EMBEDDINGS_GPU_URL || 'http://127.0.0.1:17373';
const gpuStartEnabled = process.env.CODESEARCH_EMBEDDINGS_GPU_START !== 'false';
const gpuTimeoutMsRaw = Number.parseInt(
  process.env.CODESEARCH_EMBEDDINGS_GPU_TIMEOUT_MS || '8000',
  10
);
const gpuTimeoutMs = Number.isFinite(gpuTimeoutMsRaw) ? gpuTimeoutMsRaw : 8000;
const gpuAutoInstall = process.env.CODESEARCH_EMBEDDINGS_GPU_AUTO_INSTALL !== 'false';
const gpuHealthCacheMsRaw = Number.parseInt(
  process.env.CODESEARCH_EMBEDDINGS_GPU_HEALTH_CACHE_MS || '15000',
  10
);
const gpuHealthCacheMs = Number.isFinite(gpuHealthCacheMsRaw) ? gpuHealthCacheMsRaw : 15000;

let gpuSpawned = false;
let gpuStarting = false;
let gpuLastCheck = 0;
let gpuDevice: string | null = null;
let gpuDepsChecked = false;
let gpuDepsReady = false;

export async function getExtractor(): Promise<FeatureExtractionPipeline | null> {
  if (failed) return null;
  if (extractor) return extractor;
  if (loading) return loading;

  for (const model of MODELS) {
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
      activeDims = model.dims;
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
    if (providerPreference === 'local-gpu') {
      log.warn('GPU embeddings failed, falling back to local model');
    }
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

    for (let i = 0; i < texts.length; i++) {
      const start = i * activeDims;
      const slice = data.slice(start, start + activeDims);
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

  if (!providerLogged) {
    log.info(`Using local GPU embeddings: ${response.device || 'unknown'} (${gpuUrl})`);
    providerLogged = true;
  }

  const dims = Number.isFinite(response.dims) ? response.dims : embeddings[0]?.length;
  if (dims) activeDims = dims;
  return embeddings;
}

async function shouldUseGpu(): Promise<boolean> {
  if (providerPreference === 'local') return false;
  const status = await getGpuStatus();
  if (!status) return false;
  if (providerPreference === 'local-gpu') return true;
  const device = (status.device || '').toLowerCase();
  return device !== '' && device !== 'cpu';
}

async function getGpuStatus(): Promise<{ device?: string; dims?: number } | null> {
  const now = Date.now();
  if (gpuDevice && now - gpuLastCheck < gpuHealthCacheMs) {
    return { device: gpuDevice };
  }

  await maybeStartGpuServer();

  const response = await fetchJson(`${gpuUrl}/health`, { method: 'GET' });
  if (!response) return null;

  if (typeof response.device === 'string') {
    gpuDevice = response.device;
  }
  gpuLastCheck = now;
  return response;
}

async function maybeStartGpuServer(): Promise<void> {
  if (!gpuStartEnabled || gpuSpawned || gpuStarting) return;
  if (providerPreference === 'local') return;

  const scriptPath = resolveGpuServerPath();
  if (!scriptPath) return;

  const depsReady = await ensureGpuDeps(scriptPath);
  if (!depsReady) return;

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
  const child = spawn('python3', [scriptPath, '--host', host, '--port', port], {
    stdio: 'ignore',
  });

  child.on('error', (err) => {
    gpuStarting = false;
    gpuSpawned = false;
    log.warn(`GPU server failed to start: ${err.message}`);
  });

  child.on('exit', (code) => {
    gpuStarting = false;
    if (!gpuSpawned) {
      log.warn(`GPU server exited early with code ${code}`);
    }
    gpuSpawned = false;
  });

  gpuSpawned = true;
  gpuStarting = false;
}

async function ensureGpuDeps(scriptPath: string): Promise<boolean> {
  if (gpuDepsChecked) return gpuDepsReady;

  gpuDepsChecked = true;

  const importCheck = await runPython(['-c', 'import fastapi,uvicorn,sentence_transformers,torch']);
  if (importCheck === 0) {
    gpuDepsReady = true;
    return true;
  }

  if (!gpuAutoInstall) {
    log.warn('GPU deps missing; set CODESEARCH_EMBEDDINGS_GPU_AUTO_INSTALL=true to install');
    gpuDepsReady = false;
    return false;
  }

  const requirements = path.resolve(path.dirname(scriptPath), 'requirements-gpu.txt');
  log.info('Installing GPU embedding dependencies...');
  const install = await runPython(['-m', 'pip', 'install', '-r', requirements]);
  if (install !== 0) {
    log.warn('GPU deps install failed; falling back to local model');
    gpuDepsReady = false;
    return false;
  }

  const recheck = await runPython(['-c', 'import fastapi,uvicorn,sentence_transformers,torch']);
  gpuDepsReady = recheck === 0;
  if (!gpuDepsReady) {
    log.warn('GPU deps still missing after install; falling back to local model');
  }

  return gpuDepsReady;
}

async function runPython(args: string[]): Promise<number> {
  return new Promise((resolve) => {
    const child = spawn('python3', args, { stdio: 'ignore' });
    child.on('error', () => resolve(1));
    child.on('exit', (code) => resolve(code ?? 1));
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
  options: RequestInit
): Promise<Record<string, any> | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), gpuTimeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) return null;
    return (await response.json()) as Record<string, any>;
  } catch (err) {
    if (providerPreference === 'local-gpu') {
      log.warn('GPU embeddings request failed', (err as Error).message);
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export function getDims(): number {
  return activeDims;
}
