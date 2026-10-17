import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { ReviewError } from './json-store.js';

async function body(request) {
  if (!request.headers['content-type']?.startsWith('application/json')) throw new ReviewError('Expected JSON.', 415);
  let text = '';
  for await (const chunk of request) {
    text += chunk;
    if (Buffer.byteLength(text) > 1024 * 1024) throw new ReviewError('Request is too large.', 413);
  }
  try { return JSON.parse(text); } catch { throw new ReviewError('Invalid JSON.'); }
}
export function createReviewServer({ store, dist }) {
  return createServer(async (request, response) => {
    const json = (status, value) => { response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' }); response.end(JSON.stringify(value)); };
    try {
      const host = new URL(`http://${request.headers.host || ''}`);
      if (!['127.0.0.1', 'localhost', '[::1]'].includes(host.hostname)) throw new ReviewError('This service is local only.', 403);
      if (request.headers.origin && ![`http://${host.host}`, 'http://127.0.0.1:5173', 'http://localhost:5173'].includes(request.headers.origin)) throw new ReviewError('Origin is not allowed.', 403);
      const url = new URL(request.url, host);
      if (url.pathname === '/api/config' && request.method === 'GET') return json(200, { version: store.frozen.sample.version });
      const route = url.pathname.match(/^\/api\/reviewers\/(A[1-6])(?:\/(draft|submit|export))?$/);
      if (route) {
        const [, id, action] = route;
        if (!action && request.method === 'GET') return json(200, await store.get(id));
        if (action === 'draft' && request.method === 'PUT') return json(200, await store.save(id, await body(request)));
        if (action === 'submit' && request.method === 'POST') return json(200, await store.submit(id, (await body(request)).revision));
        if (action === 'export' && request.method === 'GET') {
          const session = await store.get(id);
          const path = store.file(id, session.submission ? 'submitted' : 'draft');
          const contents = session.submission || session.draft.revision > 0 ? await readFile(path) : Buffer.from(JSON.stringify(session.draft, null, 2));
          response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Content-Disposition': `attachment; filename="${id}.${session.submission ? 'submitted' : 'draft'}.json"` });
          return response.end(contents);
        }
        throw new ReviewError('Method not allowed.', 405);
      }
      if (url.pathname.startsWith('/api/')) throw new ReviewError('Not found.', 404);
      if (request.method !== 'GET') throw new ReviewError('Method not allowed.', 405);
      const root = resolve(dist), file = resolve(root, '.' + decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
      if (!file.startsWith(root + sep)) throw new ReviewError('Not found.', 404);
      const contentTypes = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg' };
      let bytes;
      try { bytes = await readFile(file); } catch (error) { if (error.code === 'ENOENT') throw new ReviewError('Page not found. Run npm run build before npm start.', 404); throw error; }
      response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff' }); response.end(bytes);
    } catch (error) {
      if (!error.status) console.error(error);
      json(error.status || 500, { error: error.status ? error.message : 'Could not read or save the local files. Your changes have not been confirmed as saved.' });
    }
  });
}
