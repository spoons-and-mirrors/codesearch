# CodeSearch Plugin

Semantic codebase search for OpenCode. It builds a local vector index and exposes a single tool: `codebase_search`.

## What it does
- Indexes your project on startup and keeps it in sync
- Uses local embeddings by default (CPU)
- Optional local GPU embeddings with automatic CPU fallback

## Usage
Start OpenCode as usual. The plugin indexes in the background and provides:

```
codebase_search({"query":"where is auth handled?","limit":5})
```

## Local GPU embeddings (optional)
This plugin can run embeddings on your **local GPU** via a small Python sidecar server. If the GPU path fails, it falls back to CPU embeddings automatically.

Python requirement: **Python 3.10+** (recommended by sentence-transformers).

Install GPU server deps (auto-install is on by default):

```
pip install -r scripts/requirements-gpu.txt
```

Note: `torch` often installs a CPU-only build by default. If you want CUDA/MPS acceleration, install the GPU-enabled torch build for your system.

### GPU env vars
- `CODESEARCH_EMBEDDINGS_PROVIDER=auto|local-gpu|local`
- `CODESEARCH_EMBEDDINGS_GPU_URL=http://127.0.0.1:17373`
- `CODESEARCH_EMBEDDINGS_GPU_START=false` (disable auto-start)
- `CODESEARCH_EMBEDDINGS_GPU_AUTO_INSTALL=false` (disable auto-install)
- `CODESEARCH_EMBEDDINGS_GPU_MODEL=jinaai/jina-embeddings-v2-base-code`
- `CODESEARCH_EMBEDDINGS_GPU_DEVICE=auto|cuda|mps|cpu`
- `CODESEARCH_EMBEDDINGS_GPU_BATCH_SIZE=16`

## Files
- `src/index.ts` plugin entry
- `src/indexer.ts` indexing pipeline
- `src/embed.ts` embedding provider selection
- `scripts/gpu_embeddings_server.py` local GPU server
