import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';
import { log } from './logger';

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
  const ext = await getExtractor();
  if (!ext) return null;

  try {
    const output = await ext(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data as Float32Array).slice(0, activeDims);
  } catch (err) {
    log.error('Embedding failed', (err as Error).message);
    return null;
  }
}

export function getDims(): number {
  return activeDims;
}
