// Reproduce with: node construction/audit/analysis/analyze.mjs
// Read-only inputs: formal submissions, frozen sample, and original source data.
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, relative } from 'node:path';
import assert from 'node:assert/strict';
import { hash, loadFrozen, selectAndAssign } from '../interface/scripts/freeze-sample.js';
import { displayItem, isComplete } from '../interface/src/review.js';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const out = fileURLToPath(new URL('./', import.meta.url));
const responseDir = resolve(root, 'construction/human audit/main150-v1');
const { sample, assignments, manifest } = await loadFrozen();
const sourceBytes = await readFile(resolve(root, manifest.source));
assert.equal(hash(sourceBytes), manifest.source_sha256, 'Main split hash');
const population = sourceBytes.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
const regenerated = selectAndAssign(population);
assert.deepEqual(regenerated.sample, sample, 'Deterministic frozen sample');
assert.deepEqual(regenerated.assignments, assignments, 'Deterministic assignments');

const checks = ['q1', 'q2', 'q3'];
const reviewKeys = [...checks, 'note', 'no_reason', 'updated_at'];
const expectedFiles = manifest.reviewers.map(id => `${id}.submitted.json`).sort();
assert.deepEqual((await readdir(responseDir)).filter(f => f.endsWith('.submitted.json')).sort(), expectedFiles);
const items = new Map(sample.items.map(item => [item.id, item]));
const byItem = new Map(sample.items.map(item => [item.id, []]));
const submissions = [];
for (const id of manifest.reviewers) {
  const file = resolve(responseDir, `${id}.submitted.json`);
  const bytes = await readFile(file);
  const submission = JSON.parse(bytes);
  const { sha256, ...body } = submission;
  assert.equal(hash(JSON.stringify(body)), sha256, `${id}: receipt hash`);
  for (const snapshot of [submission, submission.draft]) {
    assert.equal(snapshot.schema_version, 1);
    assert.equal(snapshot.sample_version, manifest.version);
    assert.equal(snapshot.sample_sha256, manifest.sample_sha256);
    assert.equal(snapshot.assignments_sha256, manifest.assignments_sha256);
    assert.equal(snapshot.reviewer_id, id);
  }
  assert.deepEqual(submission.records.map(r => r.item_id), assignments.reviewers[id], `${id}: ordered assignment`);
  assert.deepEqual(Object.keys(submission.draft.reviews).sort(), [...assignments.reviewers[id]].sort());
  assert.equal(submission.records.length, 50);
  for (const r of submission.records) {
    assert.equal(r.reviewer_id, id);
    assert.ok(isComplete(r), `${id}/${r.item_id}: complete response`);
    assert.ok(['', 'issue', 'uncertain'].includes(r.no_reason));
    assert.ok(Number.isFinite(Date.parse(r.updated_at)));
    assert.ok(Date.parse(r.updated_at) <= Date.parse(submission.submitted_at));
    for (const key of reviewKeys) assert.deepEqual(r[key], submission.draft.reviews[r.item_id][key]);
    assert.deepEqual(r.displayed_input, displayItem(items.get(r.item_id)), `${id}/${r.item_id}: displayed input`);
    byItem.get(r.item_id).push(r);
  }
  submissions.push({
    reviewer_id: id, file: relative(root, file).replaceAll('\\', '/'),
    file_sha256: hash(bytes), receipt_sha256: sha256,
    submitted_at: submission.submitted_at, records: submission.records,
  });
}
for (const [id, records] of byItem) {
  assert.equal(records.length, 2);
  assert.deepEqual(records.map(r => r.reviewer_id).sort(), [...assignments.by_item[id]].sort());
}

const records = submissions.flatMap(s => s.records);
const isFlagged = r => checks.some(q => r[q] === 'no');
const ratio = (a, b) => b ? a / b : null;
function agreement(pairs) {
  const n = pairs.length;
  const bothYes = pairs.filter(p => p.every(v => v === 'yes')).length;
  const bothNo = pairs.filter(p => p.every(v => v === 'no')).length;
  const mixed = n - bothYes - bothNo;
  const yes = 2 * bothYes + mixed;
  const pYes = yes / (2 * n);
  const observed = (bothYes + bothNo) / n;
  const expected = pYes ** 2 + (1 - pYes) ** 2;
  return {
    paired_units: n, yes_judgments: yes, no_judgments: 2 * n - yes,
    both_yes: bothYes, mixed, both_no: bothNo,
    yes_rate: pYes, observed_agreement: observed,
    fleiss_kappa: expected === 1 ? null : (observed - expected) / (1 - expected),
    positive_specific_agreement: ratio(2 * bothYes, 2 * bothYes + mixed),
    negative_specific_agreement: ratio(2 * bothNo, 2 * bothNo + mixed),
  };
}
const itemResults = [...byItem].map(([id, rs]) => ({
  item_id: id, stratum: assignments.stratum_by_item[id],
  reviewers: rs.map(r => r.reviewer_id),
  flagged: rs.some(isFlagged), both_flagged: rs.every(isFlagged),
  any_check_disagreement: checks.some(q => rs[0][q] !== rs[1][q]),
  judgments: rs.map(r => Object.fromEntries(['reviewer_id', ...reviewKeys].map(k => [k, r[k]]))),
}));
const questionStats = Object.fromEntries(checks.map(q => [q, agreement([...byItem.values()].map(rs => rs.map(r => r[q])))]));
const itemAgreement = agreement([...byItem.values()].map(rs => rs.map(r => isFlagged(r) ? 'no' : 'yes')));
const pooled = agreement([...byItem.values()].flatMap(rs => checks.map(q => rs.map(r => r[q]))));
const strata = manifest.strata.map(s => {
  const selected = itemResults.filter(i => i.stratum === s.key);
  const flagged = selected.filter(i => i.flagged).length;
  return { ...s, flagged_items: flagged, unanimous_all_yes: selected.length - flagged,
    flagged_rate: flagged / selected.length, population_weight: s.population / population.length,
    check_no_counts: Object.fromEntries(checks.map(q => [q, selected.flatMap(i => byItem.get(i.item_id)).filter(r => r[q] === 'no').length])) };
});
const pairs = [];
for (let i = 0; i < manifest.reviewers.length; i++) {
  for (const b of manifest.reviewers.slice(i + 1)) {
    const a = manifest.reviewers[i];
    const shared = itemResults.filter(x => x.reviewers.includes(a) && x.reviewers.includes(b));
    assert.equal(shared.length, 10);
    pairs.push({ reviewers: [a, b], shared_items: shared.length,
      items_with_disagreement: shared.filter(x => x.any_check_disagreement).length });
  }
}
const reviewerStats = submissions.map(s => ({
  reviewer_id: s.reviewer_id, records: s.records.length, submitted_at: s.submitted_at,
  flagged_reviews: s.records.filter(isFlagged).length,
  clear_issue_reviews: s.records.filter(r => isFlagged(r) && r.no_reason === 'issue').length,
  uncertain_reviews: s.records.filter(r => isFlagged(r) && r.no_reason === 'uncertain').length,
  no_counts: Object.fromEntries(checks.map(q => [q, s.records.filter(r => r[q] === 'no').length])),
}));
const flags = itemResults.filter(x => x.flagged).map(x => ({
  ...x, displayed_input: displayItem(items.get(x.item_id)), adjudication_status: 'not_recorded',
}));
const weightedFlagRate = strata.reduce((sum, s) => sum + s.population_weight * s.flagged_rate, 0);

// The source link is evaluated only for the observed named-entity dispute.
// It supports a provisional interpretation, never a new human adjudication.
const caseId = 'ambigdocs_entity_f96e8711d9ee121a_v8';
let caseEvidence = null;
if (flags.some(f => f.item_id === caseId)) {
  const item = items.get(caseId);
  const ref = item.evidence.correct.source_refs[0];
  const locator = Object.fromEntries(ref.source_locator.map(x => [x.name, x.value]));
  const sourcePath = resolve(root, 'construction/sources/ambigdocs/test.json');
  const rawBytes = await readFile(sourcePath).catch(error => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
  // Raw source caches are optional and are not distributed with the code.
  if (rawBytes !== null) {
    assert.equal(hash(rawBytes), ref.archive_file_sha256, 'AmbigDocs archived source hash');
    const sourceRecord = JSON.parse(rawBytes)[Number(locator.index)];
    const document = sourceRecord.documents[Number(locator.document_index)];
    assert.equal(document.pid, locator.pid);
    assert.equal(document.answer, item.gold_answer);
    assert.equal(item.evidence.correct.text.replace(item.gold_answer, item.lure_answer), item.evidence.false.text);
    caseEvidence = {
      item_id: caseId, source_file: relative(root, sourcePath).replaceAll('\\', '/'),
      source_file_sha256: hash(rawBytes), record_index: Number(locator.index),
      document_index: Number(locator.document_index), qid: sourceRecord.qid, pid: document.pid,
      title: document.title, source_text: document.text,
      source_answer: document.answer, f_is_exact_answer_replacement: true,
      analyst_interpretation: 'The archived title and body link Jamie Elliott to James Harry Elliott. The displayed G/F passages omit an explicit alias link. This is a possible referential-clarity issue, not an established wrong-answer error. Organizer adjudication is still required.',
    };
  }
}
const result = {
  schema_version: 1, analysis_date: '2026-10-09', sample_version: manifest.version,
  provenance: {
    response_directory: relative(root, responseDir).replaceAll('\\', '/'),
    source_sha256: manifest.source_sha256, sample_sha256: manifest.sample_sha256,
    assignments_sha256: manifest.assignments_sha256,
    submissions: submissions.map(({ records: ignored, ...s }) => s),
  },
  validation: { passed: true, submitted_files: submissions.length, unique_items: byItem.size,
    item_reviews: records.length, binary_judgments: records.length * checks.length,
    verified: ['Frozen byte hashes', 'Source hash', 'Deterministic sampling and assignments',
      'All six submission receipts', 'Metadata and completeness', 'Two assigned reviewers per item',
      'Embedded draft versus final judgments', 'Displayed input versus frozen item', '15 reviewer pairs, 10 shared items each'],
    limitation: 'File integrity does not independently establish reviewer identity or independent working conditions.' },
  counts: { unanimous_all_yes_items: itemResults.filter(x => !x.flagged).length,
    any_flag_items: flags.length, both_flagged_items: itemResults.filter(x => x.both_flagged).length,
    all_yes_reviews: records.filter(r => !isFlagged(r)).length,
    clear_issue_reviews: reviewerStats.reduce((n, r) => n + r.clear_issue_reviews, 0),
    uncertain_reviews: reviewerStats.reduce((n, r) => n + r.uncertain_reviews, 0) },
  question_stats: questionStats, item_pass_agreement: itemAgreement, pooled_check_agreement: pooled,
  strata, reviewers: reviewerStats, reviewer_pairs: pairs,
  weighted_any_flag_rate: weightedFlagRate,
  adjudication: { status: 'No completed adjudication located in the repository',
    confirmed_error_rate: null,
    flagged_case_scenarios_only: { unweighted: [0, flags.length / byItem.size], weighted: [0, weightedFlagRate] },
    caveat: 'Scenario endpoints assume unflagged sampled items are non-errors and vary only the resolution of observed flags. They are not confidence intervals or bounds on the true population error rate.' },
  items: itemResults, flagged_items: flags, case_evidence: caseEvidence,
};

await mkdir(out, { recursive: true });
await writeFile(resolve(out, 'summary.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ validation: result.validation, counts: result.counts,
  question_stats: questionStats, item_pass_agreement: itemAgreement,
  pooled_check_agreement: pooled, strata, weighted_any_flag_rate: weightedFlagRate,
  output: resolve(out, 'summary.json') }, null, 2));
