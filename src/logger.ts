import * as fs from 'node:fs';
import * as path from 'node:path';

const WRITE_INTERVAL_MS = 100;
const MAX_LOG_SIZE = 1024 * 1024; // 1MB max, then rotate

let logFile: string | null = null;
let buffer: string[] = [];
let scheduled = false;

export function initLogger(projectDir: string): void {
  const logDir = path.join(projectDir, '.opencode', 'plugins', 'codesearch');
  fs.mkdirSync(logDir, { recursive: true });
  logFile = path.join(logDir, 'codesearch.log');

  // Rotate if too large
  if (fs.existsSync(logFile)) {
    const stats = fs.statSync(logFile);
    if (stats.size > MAX_LOG_SIZE) {
      const old = path.join(logDir, 'codesearch.log.old');
      if (fs.existsSync(old)) fs.unlinkSync(old);
      fs.renameSync(logFile, old);
    }
  }
}

async function flush(): Promise<void> {
  if (buffer.length === 0 || !logFile) {
    scheduled = false;
    return;
  }
  const data = buffer.join('');
  buffer = [];
  scheduled = false;
  try {
    await fs.promises.appendFile(logFile, data);
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
  if (!logFile) return;
  const ts = new Date().toISOString();
  const extra = data !== undefined ? ` | ${JSON.stringify(data)}` : '';
  buffer.push(`[${ts}] [${level}] ${msg}${extra}\n`);
  schedule();
}

export const log = {
  debug: (msg: string, data?: unknown) => write('DEBUG', msg, data),
  info: (msg: string, data?: unknown) => write('INFO', msg, data),
  warn: (msg: string, data?: unknown) => write('WARN', msg, data),
  error: (msg: string, data?: unknown) => write('ERROR', msg, data),
  flush,
};
