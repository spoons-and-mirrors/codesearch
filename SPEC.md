# CodeSearch Plugin Specification

## Overview

A semantic codebase search plugin for OpenCode that indexes code using local embeddings and provides fast, contextually-aware search through a single `codebase_search` tool.

**Location:** `~/.config/opencode/plugins/codesearch/`

## Core Capabilities

### Single Tool: `codebase_search`

**Purpose:** Semantic search across the codebase using natural language queries.

**Input:**

- `query` (string): Natural language search query (e.g., "authentication functions", "error handling for API calls")
- `limit` (number, optional): Max results to return (default: 5)

**Output:**

- Array of search results, each containing:
  - `filePath`: Relative path to file
  - `content`: The code chunk that matched
  - `score`: Similarity score (0-1)
  - `startLine`: Approximate starting line number
  - `context`: Surrounding context if available

**Example Usage:**

```json
{
  "query": "user authentication logic",
  "limit": 3
}
```

## Technical Architecture

### 1. Embedding Model

**Selected Model:** `Xenova/jina-embeddings-v2-base-code`

**Rationale:**

- Native ONNX support in transformers.js (easy integration)
- Code-specific pretraining (better than general-purpose models)
- 768 dimensions (truncate to 512 for speed/storage balance)
- ~280MB model size, ~500MB RAM usage
- ~200ms inference time per chunk on CPU
- Long context window (8192 tokens) handles large functions
- MAP≈72 on CodeSearchNet benchmark

**Fallback Option:** `Xenova/all-MiniLM-L6-v2` (if jina proves too heavy)

- 384 dimensions
- 21MB model, ~200MB RAM
- ~50ms inference time
- Lower quality (MAP≈55) but universally deployable

### 2. Vector Storage

**Selected Solution:** `vectra`

**Rationale:**

- Pure JavaScript, zero native dependencies
- Works in Bun/Node without compilation
- JSON-based storage (easy to inspect/debug)
- Built-in metadata support
- Simple API, minimal learning curve
- Suitable for codebases up to ~10k chunks

**Storage Location:** `/.opencode/plugins/codesearch/index/`

**Index Structure:**

```
/.opencode/plugins/codesearch/
├── index/
│   ├── items.json          # Vector embeddings + metadata
│   ├── manifest.json       # Index metadata
│   └── metadata.json       # Schema info
└── state.json              # Indexing state (file hashes, timestamps)
```

**Future Consideration:** Upgrade to `hnswlib-node` if performance becomes critical (requires native compilation).

### 3. Code Chunking Strategy

**Approach:** Semantic chunking with overlap

**Parameters:**

- **Max chunk size:** 512 tokens (~2048 characters)
- **Overlap:** 50 tokens (~200 characters)
- **Chunking method:**
  - Split on function/class boundaries where possible
  - Fall back to paragraph-level splits (double newlines)
  - Hard split at max length if necessary

**Metadata per chunk:**

- `filePath`: Relative path from project root
- `chunkIndex`: Zero-based chunk index within file
- `startLine`: Approximate line number
- `endLine`: Approximate ending line number
- `fileHash`: SHA256 of file content (for invalidation)
- `lastIndexed`: Unix timestamp

### 4. File Watching & Incremental Updates

**Strategy:** Watch filesystem, re-index on changes

**Implementation:**

- Use Bun's `watch()` API or `fs.watch()` with recursive option
- On file change:
  1. Delete all chunks for that file from index
  2. Re-chunk and re-embed the entire file
  3. Insert new chunks into index
- Track file hashes in `state.json` to detect actual content changes (skip if hash unchanged)

**Watched Patterns:**

- Include: `**/*.{ts,tsx,js,jsx,py,go,rs,java,cpp,c,h,hpp}`
- Exclude: `node_modules/`, `.git/`, `dist/`, `build/`, `*.test.*`, `*.spec.*`

**Initial Indexing:**

- Run on plugin initialization if index doesn't exist
- Run in background, don't block OpenCode startup
- Emit progress events (e.g., "Indexed 50/200 files")

### 5. Background Process Lifecycle

**Initialization (Plugin Load):**

1. Load embedding model (cache in memory)
2. Check if index exists:
   - If yes: Load index, start filesystem watcher
   - If no: Trigger initial indexing
3. Start background indexing queue

**During Operation:**

- Filesystem watcher enqueues changed files
- Background worker processes queue (rate-limited to avoid CPU spikes)
- Tool calls query the current index state (always available, even during indexing)

**Shutdown:**

- Gracefully stop watcher
- Flush any pending index updates to disk
- Unload model from memory

## Implementation Considerations

### Performance Targets

- **Initial indexing:** < 5 minutes for 1000 files
- **Search latency:** < 500ms for typical queries
- **Memory footprint:** < 1GB total (model + index)
- **Incremental update:** < 2 seconds per changed file

### Error Handling

- If model fails to load: Disable plugin, log error, don't crash OpenCode
- If index is corrupted: Delete and rebuild automatically
- If file watcher fails: Fall back to polling (warn user)
- If embedding fails for a chunk: Skip chunk, log warning, continue

### Configuration (Future)

Potential `opencode.json` configuration:

```json
{
  "plugins": {
    "codesearch": {
      "model": "Xenova/jina-embeddings-v2-base-code",
      "embeddingDims": 512,
      "chunkSize": 512,
      "maxFiles": 5000,
      "excludePatterns": ["*.test.*", "dist/**"]
    }
  }
}
```

## Dependencies

Required npm packages:

- `@xenova/transformers` - Embedding model inference
- `vectra` - Vector storage and search
- `hasha` or native `crypto` - File hashing

## Success Criteria

The plugin is successful if:

1. Users can ask natural language questions and get relevant code results
2. Index updates automatically without manual intervention
3. Works on typical developer machines (no GPU required)
4. Doesn't noticeably slow down OpenCode startup
5. Handles codebases up to 5000 files without performance degradation

## Non-Goals (v1)

- Multi-language support (focus on common languages first)
- Cross-repository search
- Historical code search (git blame integration)
- Syntax-aware search (AST-based)
- GPU acceleration (nice-to-have for v2)

## Open Questions

1. **Should test files be indexed?**
   - Leaning no (exclude `*.test.*`, `*.spec.*`) to reduce noise
2. **How to handle very large files (>10k lines)?**
   - Cap at first 5000 lines? Split into multiple logical units?
3. **Should the index be per-project or global?**
   - Per-project (in `<project>/.opencode/plugins/codesearch/`)
   - Keeps indices isolated, easier to debug
4. **What if the project doesn't have `.opencode/`?**
   - Create it automatically on first run
   - Suggest user adds to `.gitignore`

## Future Enhancements (v2+)

### Configuration File Support

Add `codesearch.config.json` in project root or `~/.config/opencode/plugin/codesearch/`:

```json
{
  "model": "jina-small" | "jina-base" | "minilm" | "custom",
  "customModel": "Xenova/your-model-here",
  "dimensions": 512,
  "chunkSize": 2000,
  "chunkOverlap": 200,
  "maxResults": 10,
  "exclude": [
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/fixtures/**",
    "**/mocks/**",
    "vendor/**"
  ],
  "include": ["src/**", "lib/**"],
  "extensions": [".ts", ".tsx", ".js", ".py", ".go", ".rs"],
  "indexTestFiles": false,
  "maxFileSize": 100000,
  "watchMode": true,
  "logLevel": "info" | "debug" | "error"
}
```

### Reindex Command

Expose a `/reindex` slash command or tool to force full reindex:

- `codebase_reindex()` - wipe and rebuild entire index
- `codebase_reindex({ path: "src/auth" })` - reindex specific directory
- Show progress: "Reindexing... 150/300 files"

### Index Statistics Tool

`codebase_stats()` returns:

```json
{
  "totalFiles": 342,
  "totalChunks": 1847,
  "indexSize": "2.3MB",
  "lastFullIndex": "2026-01-27T19:00:00Z",
  "modelLoaded": "jina-small",
  "dimensions": 512,
  "topDirectories": [
    { "path": "src/", "chunks": 1200 },
    { "path": "lib/", "chunks": 400 }
  ]
}
```

### File Watching (Live Index)

Use `chokidar` or Bun's native `watch()` for real-time updates:

- Watch for file changes in included paths
- Debounce rapid changes (100ms)
- Queue updates, process in background
- Log: "Updated index: src/auth/login.ts (3 chunks)"

### AST-Aware Chunking

Instead of naive line-based chunking, parse code structure:

- Chunk by function/method boundaries
- Chunk by class definitions
- Preserve import statements with each chunk
- Store function signatures as metadata for better matching

Example metadata:

```json
{
  "filePath": "src/auth.ts",
  "type": "function",
  "name": "validateToken",
  "signature": "async validateToken(token: string): Promise<User | null>",
  "startLine": 45,
  "endLine": 72
}
```

### Hybrid Search (Semantic + Keyword)

Combine vector similarity with keyword matching:

1. Run semantic search (top 20 candidates)
2. Re-rank with BM25 keyword score
3. Boost exact matches (function names, variable names)
4. Final score = `0.7 * semantic + 0.3 * keyword`

### Sharded Index for Large Repos

For repos with >5000 files, shard by top-level directory:

```
.opencode/plugins/codesearch/
├── shards/
│   ├── src.json
│   ├── lib.json
│   ├── packages.json
│   └── apps.json
├── manifest.json      # shard metadata
└── state.json
```

Query all shards in parallel, merge results.

### External Embedding APIs

Support cloud embedding APIs as alternative to local models:

```json
{
  "embeddings": {
    "provider": "openai" | "cohere" | "voyage" | "local",
    "model": "text-embedding-3-small",
    "apiKey": "${OPENAI_API_KEY}",
    "batchSize": 100
  }
}
```

Benefits: faster, higher quality, but requires API key and has cost.

### Query Understanding

Pre-process queries to improve results:

- Expand abbreviations: "auth" → "authentication"
- Detect intent: "how does X work" → search for X implementation
- Code pattern detection: "useState hook" → boost React files
- Language hints: "python logging" → boost .py files

### Result Caching

Cache recent queries to avoid re-embedding:

```json
{
  "queryCache": {
    "maxSize": 100,
    "ttl": 3600
  }
}
```

LRU cache with configurable size and TTL.

### Multi-Project Index

Global index mode for monorepos or multiple related projects:

```
~/.cache/codesearch/
├── projects.json
├── project-a/
│   └── index/
├── project-b/
│   └── index/
└── global-search.json  # cross-project metadata
```

### Search Filters

Extend tool with filter options:

```json
{
  "query": "error handling",
  "limit": 5,
  "filters": {
    "path": "src/api/**",
    "extension": [".ts", ".tsx"],
    "exclude": ["*.test.ts"],
    "minScore": 0.5,
    "after": "2026-01-01"
  }
}
```

### Explanation Mode

Show why a result matched:

```
## src/auth/login.ts:45-72 (score: 0.87)
Match reasons:
- High similarity to "authentication"
- Contains "validateToken" function
- References "user" and "session"
```

### Git-Aware Indexing

- Respect `.gitignore` patterns
- Option to only index tracked files
- Diff-based reindex: only re-embed changed lines after git pull
- Blame integration: "who wrote this code?"

### Performance Optimizations

- **Model preloading**: Load model during OpenCode startup, not first query
- **Batch embedding**: Embed multiple chunks in one model call
- **Index compression**: gzip the JSON index files
- **Memory mapping**: mmap large indices instead of loading to RAM
- **HNSW upgrade**: Switch to `hnswlib-node` for O(log n) search

### Plugin Hooks

Emit events for other plugins to consume:

- `codesearch:indexed` - file was indexed
- `codesearch:query` - search was performed
- `codesearch:result` - results were returned

### Security Considerations

- Never index `.env`, `credentials.json`, `*.pem`, etc.
- Configurable secrets detection
- Option to encrypt index at rest
- Audit log for searches (optional)
