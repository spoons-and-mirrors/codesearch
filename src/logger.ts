import * as fs from 'node:fs';
import * as path from 'node:path';

const LOG_DIR = path.join(process.cwd(), '.logs');
const LOG_FILE = path.join(LOG_DIR, 'codesearch.log');
const WRITE_INTERVAL_MS = 100;

let buffer: string[] = [];
let scheduled = false;
let initialized = false;

function ensureInit(): boolean {
  if (initialized) return true;
  try {
    if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.writeFileSync(LOG_FILE, '');
    initialized = true;
    return true;
  } catch {
    return false;
  }
}

async function flush(): Promise<void> {
  if (buffer.length === 0) {
    scheduled = false;
    return;
  }
  const data = buffer.join('');
  buffer = [];
  scheduled = false;
  try {
    await fs.promises.appendFile(LOG_FILE, data);
  } catch {}
}

function schedule(): void {
  if (!scheduled) {
    scheduled = true;
    setTimeout(flush, WRITE_INTERVAL_MS);
  }
}

type Level = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

function write(level: Level, msg: string, data?: unknown): void {
  if (!ensureInit()) return;
  const ts = new Date().toISOString();
  const extra = data !== undefined ? ` | ${JSON.stringify(data)}` : '';
  buffer.push(`[${ts}] [${level}] [codesearch] ${msg}${extra}\n`);
  schedule();
}

export const log = {
  debug: (msg: string, data?: unknown) => write('DEBUG', msg, data),
  info: (msg: string, data?: unknown) => write('INFO', msg, data),
  warn: (msg: string, data?: unknown) => write('WARN', msg, data),
  error: (msg: string, data?: unknown) => write('ERROR', msg, data),
  flush,
};
