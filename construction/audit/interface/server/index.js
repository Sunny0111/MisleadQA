import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { loadFrozen } from '../scripts/freeze-sample.js';
import { ReviewStore } from './json-store.js';
import { createReviewServer } from './local-http.js';

const frozen = await loadFrozen();
const directory = process.env.REVIEW_DATA_DIR || fileURLToPath(new URL('../../responses/', import.meta.url));
const store = new ReviewStore({ frozen, directory });
const server = createReviewServer({ store, dist: fileURLToPath(new URL('../dist/', import.meta.url)) });
const port = Number(process.env.PORT || 3001);
server.listen(port, '127.0.0.1', () => {
  console.log(`MisleadQA review: http://127.0.0.1:${port}`);
  console.log(`Local JSON files: ${resolve(directory, frozen.sample.version)}`);
});
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => server.close(() => process.exit(0)));
