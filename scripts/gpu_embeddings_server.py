#!/usr/bin/env python3
import argparse
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

DEFAULT_MODEL = os.getenv(
    'CODESEARCH_EMBEDDINGS_GPU_MODEL',
    'jinaai/jina-embeddings-v2-base-code',
)
DEVICE_PREF = os.getenv('CODESEARCH_EMBEDDINGS_GPU_DEVICE', 'auto').lower()
BATCH_SIZE = int(os.getenv('CODESEARCH_EMBEDDINGS_GPU_BATCH_SIZE', '16'))


def pick_device() -> str:
    if DEVICE_PREF != 'auto':
        return DEVICE_PREF
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = pick_device()
MODEL = SentenceTransformer(DEFAULT_MODEL, device=DEVICE)
DIMS = MODEL.get_sentence_embedding_dimension()

app = FastAPI()


class EmbedRequest(BaseModel):
    input: list[str]


@app.get('/health')
def health():
    return {
        'device': DEVICE,
        'model': DEFAULT_MODEL,
        'dims': DIMS,
    }


@app.post('/embed')
def embed(req: EmbedRequest):
    embeddings = MODEL.encode(
        req.input,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {
        'embeddings': embeddings.tolist(),
        'device': DEVICE,
        'dims': DIMS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=17373, type=int)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
