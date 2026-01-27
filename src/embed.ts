import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';
import { log } from './logger';
import { spawn } from 'node:child_process';
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
// Falls back to MiniLM if this fails
const MODELS = [
  { name: 'Xenova/jina-embeddings-v2-small-en', dims: 512 },
  { name: 'Xenova/all-MiniLM-L6-v2', dims: 384 },
];

let activeDims = 512;
let providerLogged = false;

// SINGLE CONFIG VARIABLE
const isGpuEnabled = process.env.OPENCODE_CODESEARCH_GPU === 'true';

const gpuUrl = 'http://127.0.0.1:17373';
const gpuTimeoutMs = 30000; // Increased to 30s to allow for model loading
const gpuHealthCacheMs = 15000;

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

  // If we just spawned it, wait a few seconds before the first health check
  // as model loading can be heavy
  if (gpuStarting) {
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  const response = await fetchJson(`${gpuUrl}/health`, { method: 'GET' });
  if (!response) return null;

  if (typeof response.device === 'string') {
    gpuDevice = response.device;
  }
  gpuLastCheck = now;
  return response;
}

async function maybeStartGpuServer(): Promise<void> {
  if (!isGpuEnabled || gpuSpawned || gpuStarting) return;

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
  const child = spawn(python, [scriptPath, '--host', host, '--port', port], {
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

async function ensureGpuDeps(scriptPath: string): Promise<string | null> {
  if (gpuDepsChecked) return gpuDepsReady ? (venvPythonPath || 'python3') : null;
  gpuDepsChecked = true;

  // 1. Try system python first
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

  // 2. Setup venv in a central location to avoid redownloading per-project
  const pythonExec = process.platform === 'win32' ? path.join(VENV_DIR, 'Scripts', 'python.exe') : path.join(VENV_DIR, 'bin', 'python3');
  
  if (!existsSync(VENV_DIR)) {
    log.info(`Creating central virtual environment in ${VENV_DIR}...`);
    // Ensure parent dir exists
    const parent = path.dirname(VENV_DIR);
    if (!existsSync(parent)) mkdirSync(parent, { recursive: true });
    
    const venvResult = await runPython('python3', ['-m', 'venv', VENV_DIR, '--without-pip']);
    if (venvResult.code !== 0) {
      log.warn(`Failed to create venv: ${venvResult.output}`);
      return null;
    }
  }
  
  venvPythonPath = pythonExec;

  // 3. Ensure pip in venv
  const pipCheck = await runPython(pythonExec, ['-m', 'pip', '--version']);
  if (pipCheck.code !== 0) {
    log.info('Bootstrapping pip in virtual environment...');
    const bootstrapped = await bootstrapPip(pythonExec);
    if (!bootstrapped) return null;
  }

  // 4. Install requirements
  const requirements = path.resolve(path.dirname(scriptPath), 'requirements-gpu.txt');
  log.info('Installing GPU embedding dependencies in venv (this can take 5-10 minutes)...');
  const install = await runPython(pythonExec, ['-m', 'pip', 'install', '-r', requirements], true);
  
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
  streamToLog = false
): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn(pythonExec, args, { stdio: ['ignore', 'pipe', 'pipe'] });
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
  options: RequestInit
): Promise<Record<string, any> | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), gpuTimeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) return null;
    return (await response.json()) as Record<string, any>;
  } catch (err) {
    if (isGpuEnabled) {
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
