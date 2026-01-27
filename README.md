# CodeSearch Plugin

Semantic codebase search for OpenCode. This plugin builds a local vector index of your project, enabling natural language search through a single tool: `codebase_search`.

## Features

- **Semantic Search**: Find code by meaning, not just keywords (e.g., "how do we handle auth?").
- **Automatic Indexing**: Background indexing on startup with incremental updates via filesystem watching.
- **Hybrid Embedding Engine**: Uses fast, local CPU embeddings by default (Transformers.js) with optional automatic local GPU acceleration (Python/Torch).
- **Zero-Config Storage**: All index data and logs are stored within the project's `.opencode/` directory.

## Usage

Once installed, the `codebase_search` tool becomes available in OpenCode.

```json
{
  "query": "database connection logic",
  "limit": 5
}
```

### Supported Languages
`.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.go`, `.rs`, `.java`, `.cpp`, `.c`, `.h`, `.md`

## Configuration

### Local GPU Acceleration (Optional)
To enable hardware-accelerated embeddings, set the following environment variable:

`OPENCODE_CODESEARCH_GPU=true`

**Note:** On first run with GPU enabled, the plugin will bootstrap a dedicated Python virtual environment and download `torch` and `sentence-transformers` (~1.5GB). This process happens in the background.

## Technical Details

- **Embedding Model**: `jina-embeddings-v2-base-code` (via Xenova/Transformers.js or Python).
- **Vector DB**: `vectra` (Local JSON-based storage).
- **Storage Path**: `<project_root>/.opencode/plugin/codesearch/`
- **Chunking**: Semantic chunking with overlap (~2000 chars per chunk).

## Troubleshooting

- **Logs**: Detailed logs are available at `.opencode/plugin/codesearch/codesearch.log`.
- **Reindexing**: If the index becomes out of sync, delete the `.opencode/plugin/codesearch/` directory and restart OpenCode.
- **Exclusions**: By default, `node_modules`, `.git`, `dist`, `build`, and test files are ignored.
