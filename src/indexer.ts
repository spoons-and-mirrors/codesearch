import { LocalIndex } from 'vectra';
import { embed, embedMany } from './embed';
import { log } from './logger';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';

//.. testdffsdsdsd

const EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.jsx',
  '.py',
  '.go',
  '.rs',
  '.java',
  '.cpp',
  '.c',
  '.h',
  '.md',
]);
const EXCLUDE_DIRS = new Set([
  'node_modules',
  '.git',
  'dist',
  'build',
  '.opencode',
  '.next',
  'vendor',
  '.venv-gpu',
  '.venv',
  'venv',
  'env',
  '.env',
]);
const CHUNK_SIZE = 2000;
const OVERLAP = 200;
const EMBED_BATCH_SIZE = Math.max(
  1,
  Number.parseInt(process.env.CODESEARCH_EMBEDDINGS_BATCH_SIZE || '16', 10) || 16
);

interface ChunkMeta {
  filePath: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  fileHash: string;
  content: string;
  [key: string]: string | number;
}

interface StateFile {
  hashes: Record<string, string>;
  lastIndexed: number;
}

export class CodeIndexer {
  private index: LocalIndex;
  private indexDir: string;
  private stateFile: string;
  private state: StateFile = { hashes: {}, lastIndexed: 0 };
  private projectRoot: string;

  constructor(projectRoot: string) {
    this.projectRoot = projectRoot;
    this.indexDir = path.join(projectRoot, '.opencode', 'plugin', 'codesearch');
    this.stateFile = path.join(this.indexDir, 'state.json');
    this.index = new LocalIndex(this.indexDir);
  }

  async init(): Promise<void> {
    fs.mkdirSync(this.indexDir, { recursive: true });
    if (!(await this.index.isIndexCreated())) {
      await this.index.createIndex();
    }
    if (fs.existsSync(this.stateFile)) {
      this.state = JSON.parse(fs.readFileSync(this.stateFile, 'utf-8'));
    }
  }

  async indexProject(): Promise<{ indexed: number; skipped: number; deleted: number }> {
    const files = this.collectFiles(this.projectRoot);
    const allKnown = new Set(files.map(f => path.relative(this.projectRoot, f)));
    
    // List all files in current project for debug
    const entries = fs.readdirSync(this.projectRoot, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile()) {
        const rel = entry.name;
        if (!allKnown.has(rel) && !EXCLUDE_DIRS.has(rel) && !rel.startsWith('.')) {
          log.debug(`Skipping file (unsupported extension): ${rel}`);
        }
      }
    }

    const relFiles = new Set(files.map((f) => path.relative(this.projectRoot, f)));
    let indexed = 0;
    let skipped = 0;
    let deleted = 0;

    // Handle deletions
    for (const rel of Object.keys(this.state.hashes)) {
      if (!relFiles.has(rel)) {
        log.info(`Removing deleted file: ${rel}`);
        await this.removeFile(rel);
        delete this.state.hashes[rel];
        deleted++;
      }
    }

    // Handle additions/modifications
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf-8');
      const hash = crypto.createHash('sha256').update(content).digest('hex');
      const rel = path.relative(this.projectRoot, file);

      if (this.state.hashes[rel] === hash) {
        skipped++;
        continue;
      }

      await this.indexFile(rel, content, hash);
      this.state.hashes[rel] = hash;
      indexed++;
    }

    this.state.lastIndexed = Date.now();
    fs.writeFileSync(this.stateFile, JSON.stringify(this.state, null, 2));
    log.info(`Indexed ${indexed} files, skipped ${skipped}, deleted ${deleted}`);
    return { indexed, skipped, deleted };
  }

  async removeFile(filePath: string): Promise<void> {
    const existing = await this.index.listItems();
    for (const item of existing) {
      if (item.metadata.filePath === filePath) {
        await this.index.deleteItem(item.id);
      }
    }
  }

  async indexFile(filePath: string, content: string, hash: string): Promise<void> {
    await this.removeFile(filePath);

    const chunks = this.chunkContent(content, filePath, hash);
    let indexed = 0;
    for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
      const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
      const vectors = await embedMany(batch.map((chunk) => chunk.content));

      for (let j = 0; j < batch.length; j++) {
        const vector = vectors[j];
        if (!vector) continue;
        await this.index.insertItem({ vector, metadata: batch[j] });
        indexed++;
      }
    }
    log.debug(`Indexed ${filePath} (${indexed}/${chunks.length} chunks)`);
  }

  private chunkContent(content: string, filePath: string, hash: string): ChunkMeta[] {
    const lines = content.split('\n');
    const chunks: ChunkMeta[] = [];
    let start = 0;

    while (start < lines.length) {
      let end = start;
      let len = 0;
      while (end < lines.length && len < CHUNK_SIZE) {
        len += lines[end].length + 1;
        end++;
      }

      const text = lines.slice(start, end).join('\n');
      chunks.push({
        filePath,
        chunkIndex: chunks.length,
        startLine: start + 1,
        endLine: end,
        fileHash: hash,
        content: text,
      });

      const overlap = Math.floor(OVERLAP / 40);
      start = Math.max(start + 1, end - overlap);
    }

    return chunks;
  }

  private collectFiles(dir: string): string[] {
    const files: string[] = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      if (EXCLUDE_DIRS.has(entry.name)) continue;
      if (entry.name.startsWith('.') && entry.name !== '.') continue;

      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...this.collectFiles(full));
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name);
        if (
          EXTENSIONS.has(ext) &&
          !entry.name.includes('.test.') &&
          !entry.name.includes('.spec.')
        ) {
          files.push(full);
        }
      }
    }
    return files;
  }

  async search(
    query: string,
    limit = 5
  ): Promise<Array<{ filePath: string; content: string; score: number; startLine: number }>> {
    const vector = await embed(query);
    if (!vector) {
      log.error('Failed to embed query');
      return [];
    }
    const results = await this.index.queryItems(vector, limit);
    log.debug(`Search: "${query}" -> ${results.length} results`);
    return results.map((r) => ({
      filePath: r.item.metadata.filePath as string,
      content: r.item.metadata.content as string,
      score: r.score,
      startLine: r.item.metadata.startLine as number,
    }));
  }

  isIndexed(): boolean {
    return this.state.lastIndexed > 0;
  }

  watch(): void {
    log.info(`Watching for changes in ${this.projectRoot}...`);
    fs.watch(this.projectRoot, { recursive: true }, async (event, filename) => {
      try {
        if (!filename) return;

        // Skip internal plugin state changes to avoid infinite loops
        if (filename.startsWith('.opencode')) return;

        // Skip excluded paths
        const parts = filename.split(path.sep);
        if (parts.some((p) => EXCLUDE_DIRS.has(p))) return;

        const ext = path.extname(filename);
        if (!EXTENSIONS.has(ext)) return;
        if (filename.includes('.test.') || filename.includes('.spec.')) return;

        const fullPath = path.join(this.projectRoot, filename);

        if (!fs.existsSync(fullPath)) {
          // Deletion
          log.info(`File deleted: ${filename}`);
          await this.removeFile(filename);
          delete this.state.hashes[filename];
        } else {
          // Addition or modification
          const content = fs.readFileSync(fullPath, 'utf-8');
          const hash = crypto.createHash('sha256').update(content).digest('hex');

          if (this.state.hashes[filename] === hash) return;

          log.info(`File ${event === 'rename' ? 'added' : 'changed'}: ${filename}`);
          await this.indexFile(filename, content, hash);
          this.state.hashes[filename] = hash;
        }

        this.state.lastIndexed = Date.now();
        fs.writeFileSync(this.stateFile, JSON.stringify(this.state, null, 2));
      } catch (err: any) {
        log.error(`Watch error for ${filename}`, err.message);
      }
    });
  }
}
