import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, dirname, basename } from 'node:path';
import { STRATA, REVIEWERS, selectAndAssign, validateAllocation, loadFrozen, sourcePath, hash } from '../scripts/freeze-sample.js';
import { ReviewStore } from '../server/json-store.js';
import { createReviewServer } from '../server/local-http.js';
import { DraftWriter } from '../src/local-api.js';
import { syntheticFixture } from './fixtures.js';

async function fixture(t) {
  const directory = await mkdtemp(resolve(tmpdir(), 'misleadqa-review-test-'));
  t.after(async () => {
    const target = resolve(directory);
    if (dirname(target) !== resolve(tmpdir()) || !basename(target).startsWith('misleadqa-review-test-')) throw new Error('Unsafe cleanup target.');
    await rm(target, { recursive: true, force: true });
  });
  const frozen = syntheticFixture();
  return { frozen, directory, store: new ReviewStore({ frozen, directory }) };
}
function complete(draft) {
  return { ...draft, reviews: Object.fromEntries(Object.keys(draft.reviews).map(id => [id, { q1: 'yes', q2: 'yes', q3: 'yes', no_reason: '', note: 'Synthetic automated test only.' }])) };
}

test('frozen sample contains unchanged main records with exact quotas and balanced independent assignments', async () => {
  const frozen = await loadFrozen();
  const sourceBytes = await readFile(sourcePath);
  assert.equal(hash(sourceBytes), frozen.manifest.source_sha256);
  const source = sourceBytes.toString().trim().split(/\r?\n/).map(line => JSON.parse(line));
  const expected = selectAndAssign(source);
  assert.deepEqual(frozen.sample, expected.sample);
  assert.deepEqual(frozen.assignments, expected.assignments);
  validateAllocation(frozen.sample, frozen.assignments);
  assert.equal(frozen.sample.items.length, 150);
  const byId = new Map(source.map(item => [item.id, item]));
  frozen.sample.items.forEach(item => assert.deepEqual(item, byId.get(item.id)));
  const distribution = REVIEWERS.map(id => STRATA.map(s => frozen.assignments.reviewers[id].filter(itemId => frozen.assignments.stratum_by_item[itemId] === s.key).length));
  assert.deepEqual(distribution, [[5,5,11,12,17], [5,5,11,12,17], [5,5,12,11,17], [5,5,12,11,17], [5,5,12,12,16], [5,5,12,12,16]]);
  assert.deepEqual(selectAndAssign([...source].reverse()), expected, 'Source line order must not change the frozen sample.');
});

test('changed frozen files are rejected', async t => {
  const { frozen, directory } = await fixture(t);
  await Promise.all(Object.entries({ 'sample.json': frozen.sample, 'assignments.json': frozen.assignments, 'manifest.json': frozen.manifest }).map(([name, value]) => writeFile(resolve(directory, name), JSON.stringify(value))));
  await loadFrozen(directory);
  frozen.sample.items[0].question = 'Tampered';
  await writeFile(resolve(directory, 'sample.json'), JSON.stringify(frozen.sample));
  await assert.rejects(loadFrozen(directory), /changed/);
});

test('local JSON survives restart, final submission is complete, immutable and idempotent', async t => {
  const { store, frozen, directory } = await fixture(t);
  const initial = await store.get('A1');
  assert.equal(initial.items.length, 50);
  await assert.rejects(store.submit('A1', 0), /every item/);
  const draft = await store.save('A1', complete(initial.draft));
  assert.equal(draft.revision, 1);
  const restarted = new ReviewStore({ frozen, directory });
  assert.deepEqual((await restarted.get('A1')).draft, draft);
  const receipt = await restarted.submit('A1', 1);
  const snapshot = JSON.parse(await readFile(receipt.path));
  assert.equal(snapshot.records.length, 50);
  assert.deepEqual(snapshot.records[0].displayed_input, initial.items[0]);
  const { sha256, ...body } = snapshot;
  assert.equal(hash(JSON.stringify(body)), sha256);
  const original = await readFile(receipt.path, 'utf8');
  assert.deepEqual(await restarted.submit('A1', 1), receipt);
  await assert.rejects(restarted.save('A1', draft), /locked/);
  assert.equal(await readFile(receipt.path, 'utf8'), original);
  assert.equal((await restarted.get('A2')).draft.revision, 0);
  assert.deepEqual((await readdir(store.directory)).sort(), ['A1.draft.json', 'A1.submitted.json']);
});

test('No needs a reason; invalid IDs, foreign items and unknown response values are rejected', async t => {
  const { store } = await fixture(t);
  const { draft } = await store.get('A2');
  const ready = complete(draft), id = Object.keys(ready.reviews)[0];
  ready.reviews[id].q2 = 'no'; ready.reviews[id].note = '';
  const saved = await store.save('A2', ready);
  await assert.rejects(store.submit('A2', saved.revision), /Explain each No/);
  saved.reviews[id].note = 'Q2: unsure whether the entity is clear.';
  saved.reviews[id].no_reason = 'uncertain';
  const fixed = await store.save('A2', saved);
  await store.submit('A2', fixed.revision);
  await assert.rejects(store.get('../A1'), /reviewer ID/);
  const invalid = (await store.get('A3')).draft;
  invalid.reviews.foreign = invalid.reviews[Object.keys(invalid.reviews)[0]];
  await assert.rejects(store.save('A3', invalid), /assignment/);
  delete invalid.reviews.foreign;
  invalid.reviews[Object.keys(invalid.reviews)[0]].q1 = 'maybe';
  await assert.rejects(store.save('A3', invalid), /Invalid response/);
});

test('concurrent revisions cannot overwrite edits; a lost write response can be retried', async t => {
  const { store } = await fixture(t);
  const initial = (await store.get('A1')).draft;
  const first = complete(initial);
  const second = { ...first, cursor: 1 };
  const results = await Promise.allSettled([store.save('A1', first), store.save('A1', second)]);
  assert.equal(results[0].status, 'fulfilled');
  assert.equal(results[1].status, 'rejected');
  assert.equal(results[1].reason.status, 409);
  assert.equal((await store.save('A1', first)).revision, 1);
});

test('a write failure never yields a saved or submitted record', async t => {
  const { store } = await fixture(t);
  const draft = complete((await store.get('A1')).draft);
  await writeFile(store.directory, 'This file prevents directory creation.');
  await assert.rejects(store.save('A1', draft));
  assert.equal(await readFile(store.directory, 'utf8'), 'This file prevents directory creation.');
});

test('HTTP serves assigned items, saves and exports local JSON, and rejects external origins', async t => {
  const { store, directory } = await fixture(t);
  const server = createReviewServer({ store, dist: resolve(directory, 'web') });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  const config = await (await fetch(base + '/api/config')).json();
  assert.deepEqual(config, { version: 'main150-v1' });
  assert.equal((await fetch(base + '/api/reviewers')).status, 404, 'The landing page must not expose a reviewer roster or progress.');
  const session = await (await fetch(base + '/api/reviewers/A4')).json();
  assert.equal(session.items.length, 50);
  assert.equal(session.items[0].passages.G, '2020: one.');
  const headers = { 'Content-Type': 'application/json' };
  const saved = await (await fetch(base + '/api/reviewers/A4/draft', { method: 'PUT', headers, body: JSON.stringify(complete(session.draft)) })).json();
  const blocked = await fetch(base + '/api/reviewers/A4/submit', { method: 'POST', headers: { ...headers, Origin: 'https://unrelated.example' }, body: JSON.stringify({ revision: saved.revision }) });
  assert.equal(blocked.status, 403);
  const submit = await fetch(base + '/api/reviewers/A4/submit', { method: 'POST', headers, body: JSON.stringify({ revision: saved.revision }) });
  assert.equal(submit.status, 200);
  const exported = await (await fetch(base + '/api/reviewers/A4/export')).json();
  assert.equal(exported.records.length, 50);
  assert.equal((await fetch(base + '/api/reviewers/A7')).status, 404);
  assert.equal((await fetch(base + '/main150-v1/A4.submitted.json')).status, 404);
});

test('autosave recovers a lost response and later edits across a refresh', async t => {
  const { store } = await fixture(t);
  const session = await store.get('A5');
  const memory = new Map();
  const storage = { getItem: key => memory.get(key), setItem: (key, value) => memory.set(key, value), removeItem: key => memory.delete(key) };
  let loseResponse = true;
  const send = async (path, options) => {
    const result = await store.save('A5', JSON.parse(options.body));
    if (loseResponse) { loseResponse = false; throw new Error('Connection lost after save'); }
    return result;
  };
  const writer = new DraftWriter(session, { storage, send });
  const edited = complete(session.draft);
  writer.change(edited);
  await assert.rejects(writer.flush(), /Connection lost/);
  assert.equal(writer.dirty, true);
  writer.change({ ...edited, cursor: 4 });
  const restored = new DraftWriter(await store.get('A5'), { storage, send });
  assert.equal(restored.initial.cursor, 4);
  await restored.flush();
  assert.equal(restored.dirty, false);
  assert.equal(memory.size, 0);
  assert.equal((await store.get('A5')).draft.cursor, 4);
  assert.equal(restored.revision, 2);
});
