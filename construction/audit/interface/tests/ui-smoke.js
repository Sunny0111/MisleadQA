// Optional manual browser check. Never loads or writes formal audit responses.
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { syntheticFixture } from './fixtures.js';
import { ReviewStore } from '../server/json-store.js';
import { createReviewServer } from '../server/local-http.js';

const directory = await mkdtemp(resolve(tmpdir(), 'misleadqa-ui-test-'));
const store = new ReviewStore({ frozen: syntheticFixture(), directory });
const { draft } = await store.get('A1');
Object.values(draft.reviews).slice(0, 49).forEach(review => Object.assign(review, { q1: 'yes', q2: 'yes', q3: 'yes', note: 'Synthetic test only.' }));
draft.cursor = 49;
await store.save('A1', draft);
const server = createReviewServer({ store, dist: fileURLToPath(new URL('../dist/', import.meta.url)) });
server.listen(3002, '127.0.0.1', () => console.log(`Synthetic UI check: http://127.0.0.1:3002\nTemporary files: ${directory}`));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close(async () => {
  const target = resolve(directory);
  if (dirname(target) !== resolve(tmpdir()) || !basename(target).startsWith('misleadqa-ui-test-')) throw new Error('Unsafe cleanup target.');
  await rm(target, { recursive: true, force: true });
  process.exit(0);
}));
