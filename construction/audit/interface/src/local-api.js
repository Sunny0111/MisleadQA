export async function request(path, options = {}) {
  const response = await fetch('/api' + path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers }, signal: AbortSignal.timeout(15000) });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Could not save the local file.');
  return value;
}

// Writes are serialized. A failed request is retried unchanged before later edits,
// including after a refresh, because the server may have saved a lost response.
export class DraftWriter {
  constructor(session, { storage = localStorage, send = request, onStatus = () => {} } = {}) {
    this.id = session.reviewer_id;
    this.key = `misleadqa:${session.draft.sample_sha256}:${this.id}`;
    this.storage = storage; this.send = send; this.onStatus = onStatus;
    this.revision = session.draft.revision;
    this.pending = null; this.inFlight = null; this.running = null; this.error = null;
    let cached;
    try { cached = JSON.parse(storage.getItem(this.key)); } catch { /* No recovery copy available. */ }
    if (cached && !session.submission) {
      this.revision = cached.revision;
      this.pending = cached.pending; this.inFlight = cached.inFlight;
    }
    this.initial = this.pending || this.inFlight?.draft || session.draft;
  }
  get dirty() { return Boolean(this.pending || this.inFlight); }
  cache() {
    try {
      if (this.dirty) this.storage.setItem(this.key, JSON.stringify({ revision: this.revision, pending: this.pending, inFlight: this.inFlight }));
      else this.storage.removeItem(this.key);
    } catch { /* Server saving still works if browser storage is disabled. */ }
  }
  change(draft) {
    this.pending = structuredClone(draft);
    this.cache();
    if (!this.error) this.flush().catch(() => {});
  }
  flush() {
    if (this.running) return this.running;
    this.error = null;
    this.running = this.drain().finally(() => { this.running = null; });
    return this.running;
  }
  async drain() {
    try {
      while (this.dirty) {
        this.onStatus('saving');
        if (!this.inFlight) {
          this.inFlight = { revision: this.revision, draft: this.pending };
          this.pending = null;
        }
        this.cache();
        const result = await this.send(`/reviewers/${this.id}/draft`, { method: 'PUT', body: JSON.stringify({ ...this.inFlight.draft, revision: this.inFlight.revision }) });
        this.revision = result.revision;
        this.inFlight = null;
        this.cache();
      }
      this.onStatus('saved');
    } catch (error) { this.error = error; this.onStatus('error', error.message); throw error; }
  }
}
