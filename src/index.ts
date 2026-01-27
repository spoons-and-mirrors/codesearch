import { z } from 'zod';
import { tool, type Plugin } from '@opencode-ai/plugin';
import { CodeIndexer } from './indexer';
import { log, initLogger } from './logger';

let indexer: CodeIndexer | null = null;
let indexing = false;

const plugin: Plugin = async (ctx) => {
  initLogger(ctx.directory);
  log.info(`CodeSearch starting in: ${ctx.directory}`);
  indexer = new CodeIndexer(ctx.directory);

  indexer
    .init()
    .then(async () => {
      indexing = true;
      log.info('Syncing index...');
      const result = await indexer!.indexProject();
      log.info(`Sync complete: ${result.indexed} indexed, ${result.deleted} deleted`);
      indexing = false;

      // Start live watcher
      indexer!.watch();
    })
    .catch((err) => {
      log.error('Init failed', err.message);
    });

  return {
    tool: {
      codebase_search: tool({
        description:
          'Semantic search across the codebase using natural language queries. Use this tool as a first step to find relevant code snippets.',
        args: {
          query: tool.schema.string().describe('Natural language search query'),
          limit: tool.schema.number().optional().describe('Max results to return (default: 5)'),
        },
        async execute(args) {
          if (!indexer) return 'Error: Indexer not initialized';
          if (indexing) log.debug('Search while indexing in progress');

          const limit = args.limit ?? 5;
          const results = await indexer.search(args.query, limit);

          if (results.length === 0) return 'No results found';

          return results
            .map((r) => {
              const endLine = r.startLine + r.content.split('\n').length;
              const snippet = r.content.slice(0, 500) + (r.content.length > 500 ? '...' : '');
              return `## ${r.filePath}:${r.startLine}-${endLine} (score: ${Math.round(r.score * 100) / 100})\n\`\`\`\n${snippet}\n\`\`\``;
            })
            .join('\n\n');
        },
      }),
    },
  };
};

export default plugin;
