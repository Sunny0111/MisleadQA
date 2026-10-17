import { mkdir, readFile, open, rename, link, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { REVIEWERS, hash } from '../scripts/freeze-sample.js';
import { blankReview, displayItem, isComplete } from '../src/review.js';

export class ReviewError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}
async function readJSON(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
async function atomicJSON(path, value, exclusive = false) {
  const temporary = `${path}.${randomUUID()}.tmp`;
  const handle = await open(temporary, 'wx');
  try { await handle.writeFile(JSON.stringify(value, null, 2) + '\n'); await handle.sync(); }
  finally { await handle.close(); }
  try {
    // Publish a final file exclusively: never replace an existing submission.
    if (exclusive) await link(temporary, path);
    else await rename(temporary, path);
  } finally { await unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; }); }
}

export class ReviewStore {
  constructor({ frozen, directory }) {
    this.frozen = frozen;
    this.directory = resolve(directory, frozen.sample.version);
    this.items = new Map(frozen.sample.items.map(item => [item.id, item]));
    this.queue = Promise.resolve();
  }
  assertReviewer(id) { if (!REVIEWERS.includes(id)) throw new ReviewError('Enter your assigned reviewer ID, from A1 to A6.'); }
  file(id, kind) { this.assertReviewer(id); return resolve(this.directory, `${id}.${kind}.json`); }
  ids(id) { this.assertReviewer(id); return this.frozen.assignments.reviewers[id]; }
  blank(id) {
    return { schema_version: 1, sample_version: this.frozen.sample.version, sample_sha256: this.frozen.manifest.sample_sha256, assignments_sha256: this.frozen.manifest.assignments_sha256,
      reviewer_id: id, revision: 0, cursor: 0, reviews: Object.fromEntries(this.ids(id).map(itemId => [itemId, blankReview()])), updated_at: null };
  }
  verifySnapshot(value, id) {
    if (value && (value.sample_sha256 !== this.frozen.manifest.sample_sha256 || value.assignments_sha256 !== this.frozen.manifest.assignments_sha256 || value.reviewer_id !== id)) throw new ReviewError('Saved data belongs to another frozen assignment. Keep the files and contact the organizer.', 409);
    return value;
  }
  async get(id) {
    this.assertReviewer(id);
    const submission = this.verifySnapshot(await readJSON(this.file(id, 'submitted')), id);
    const draft = submission?.draft || this.verifySnapshot(await readJSON(this.file(id, 'draft')), id) || this.blank(id);
    return { reviewer_id: id, items: this.ids(id).map(itemId => displayItem(this.items.get(itemId))), draft, submission: submission ? { submitted_at: submission.submitted_at, sha256: submission.sha256, path: this.file(id, 'submitted') } : null, directory: this.directory };
  }
  serialize(operation) {
    const next = this.queue.then(operation);
    this.queue = next.catch(() => {});
    return next;
  }
  validateDraft(id, input) {
    if (!input || !Number.isInteger(input.revision) || input.revision < 0 || !Number.isInteger(input.cursor) || input.cursor < 0 || input.cursor >= this.ids(id).length) throw new ReviewError('Invalid draft or revision.');
    if (!input.reviews || Object.keys(input.reviews).sort().join() !== [...this.ids(id)].sort().join()) throw new ReviewError('Draft does not match this reviewer’s assignment.');
    const reviews = {};
    for (const itemId of this.ids(id)) {
      const value = input.reviews[itemId];
      if (!value || !['q1', 'q2', 'q3'].every(key => [null, 'yes', 'no'].includes(value[key])) || typeof value.note !== 'string' || value.note.length > 5000 || !['', 'issue', 'uncertain'].includes(value.no_reason)) throw new ReviewError(`Invalid response for ${itemId}.`);
      reviews[itemId] = { q1: value.q1, q2: value.q2, q3: value.q3, note: value.note, no_reason: value.no_reason };
    }
    return reviews;
  }
  async save(id, input) {
    return this.serialize(async () => {
      const current = await this.get(id);
      if (current.submission) throw new ReviewError('This review has already been submitted and is locked.', 409);
      const reviews = this.validateDraft(id, input);
      const previousReviews = Object.fromEntries(Object.entries(current.draft.reviews).map(([key, value]) => [key, { q1: value.q1, q2: value.q2, q3: value.q3, note: value.note, no_reason: value.no_reason }]));
      // A lost response can be retried without creating a revision conflict.
      if (input.revision !== current.draft.revision) {
        if (input.revision + 1 === current.draft.revision && input.cursor === current.draft.cursor && JSON.stringify(reviews) === JSON.stringify(previousReviews)) return current.draft;
        throw new ReviewError('This review changed in another tab. Download your recovery copy, then reload before continuing.', 409);
      }
      const now = new Date().toISOString();
      for (const itemId of this.ids(id)) reviews[itemId].updated_at = JSON.stringify(reviews[itemId]) === JSON.stringify(previousReviews[itemId]) ? current.draft.reviews[itemId].updated_at || null : now;
      const draft = { ...current.draft, cursor: input.cursor, revision: input.revision + 1, reviews, updated_at: now };
      await mkdir(this.directory, { recursive: true });
      await atomicJSON(this.file(id, 'draft'), draft);
      return draft;
    });
  }
  async submit(id, revision) {
    return this.serialize(async () => {
      const current = await this.get(id);
      if (current.submission) return current.submission;
      if (revision !== current.draft.revision) throw new ReviewError('Save your latest changes before submitting.', 409);
      if (!Object.values(current.draft.reviews).every(isComplete)) throw new ReviewError('Answer Q1–Q3 for every item. Explain each No and indicate whether it is an issue or uncertainty.');
      const snapshot = { schema_version: 1, sample_version: this.frozen.sample.version, sample_sha256: this.frozen.manifest.sample_sha256, assignments_sha256: this.frozen.manifest.assignments_sha256, reviewer_id: id, submitted_at: new Date().toISOString(),
        draft: current.draft, records: current.items.map(item => ({ item_id: item.id, reviewer_id: id, ...current.draft.reviews[item.id], displayed_input: item })) };
      const sha256 = hash(JSON.stringify(snapshot));
      await mkdir(this.directory, { recursive: true });
      await atomicJSON(this.file(id, 'submitted'), { ...snapshot, sha256 }, true);
      return { submitted_at: snapshot.submitted_at, sha256, path: this.file(id, 'submitted') };
    });
  }
}
