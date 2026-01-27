# CodeSearch Plugin

Semantic codebase search for OpenCode. It builds a local vector index and exposes a single tool: `codebase_search`.

## What it does
- **Automatic Indexing**: Indexes your project on startup and keeps it in sync using a background watcher.
- **Hybrid Embeddings**: Uses fast CPU embeddings by default (Transformers.js) and can automatically scale to a local GPU if available.
- **Zero Configuration**: Automatically handles its own Python virtual environment and dependencies.

## Usage
Start OpenCode as usual. The plugin indexes in the background. You can then use the `codebase_search` tool:

```json
codebase_search({"query": "where is the embedding logic?", "limit": 5})
```

---

## Local GPU Embeddings (Opt-in)
By default, this plugin uses **CPU-only embeddings** (Transformers.js) which are lightweight and require no extra setup.

To enable **hardware acceleration (GPU)**:
- Set `OPENCODE_CODESEARCH_GPU=true`

### 🚀 GPU Auto-Setup
When opted-in, the plugin will:
1.  **Global Virtual Environment**: Create a central folder at `~/.local/share/opencode/storage/plugin/codesearch/gpu/venv`.
2.  **Auto-Bootstrap**: Install `pip`, `fastapi`, `uvicorn`, `sentence-transformers`, and `torch`.

> [!IMPORTANT]
> **First run takes 5-10 minutes.**
> The GPU dependencies (specifically `torch`) are large (~1GB+). The plugin will stream the installation progress to your logs with the `[pip]` prefix. **Do not close the application during this process.**

---

## Troubleshooting
- **Stuck on "Installing..."**: Check your internet connection and disk space. The download is ~1.5GB total.
- **CPU Fallback**: If GPU setup fails, the plugin will log the error and fallback to CPU embeddings automatically so you can still search.
- **Indexer Noise**: The plugin ignores `node_modules`, `.git`, and its own `.venv-gpu` directory automatically.

## Files
- `src/index.ts`: Plugin entry point and tool definition.
- `src/indexer.ts`: File watching and vector database sync.
- `src/embed.ts`: Provider logic (Transformers.js vs Python Sidecar).
- `scripts/gpu_embeddings_server.py`: The Python sidecar for GPU acceleration.
