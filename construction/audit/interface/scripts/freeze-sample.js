import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const VERSION = 'main150-v1';
export const SEED = '20261007';
export const REVIEWERS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'];
export const STRATA = [
  { key: 'fred_time', source: 'FRED', dimension: 'time', population: 200, sample: 15, label: 'FRED · Time' },
  { key: 'fred_geography', source: 'FRED', dimension: 'geography', population: 200, sample: 15, label: 'FRED · Location' },
  { key: 'situatedqa_time', source: 'SituatedQA', dimension: 'time', population: 183, sample: 35, label: 'SituatedQA · Time' },
  { key: 'situatedqa_geography', source: 'SituatedQA', dimension: 'geography', population: 184, sample: 35, label: 'SituatedQA · Location' },
  { key: 'ambigdocs_entity', source: 'AmbigDocs', dimension: 'entity', population: 192, sample: 50, label: 'AmbigDocs · Entity' },
];
export const PAIRS = REVIEWERS.flatMap((a, i) => REVIEWERS.slice(i + 1).map(b => [a, b]));
const extraPairs = [[], [], ['A2-A3', 'A3-A4', 'A4-A5', 'A5-A6', 'A1-A6'], ['A1-A3', 'A1-A5', 'A2-A5', 'A2-A6', 'A4-A6'], ['A1-A2', 'A1-A4', 'A2-A4', 'A3-A5', 'A3-A6']];
export const hash = data => createHash('sha256').update(data).digest('hex');
export const frozenDir = fileURLToPath(new URL('../../frozen/main150-v1/', import.meta.url));
export const sourcePath = fileURLToPath(new URL('../../../data/main_split.jsonl', import.meta.url));
const rank = (domain, id) => hash(`${SEED}\0${domain}\0${id}`);
const ranked = (values, domain, id = v => v.id) => [...values].sort((a, b) => {
  const x = rank(domain, id(a)), y = rank(domain, id(b));
  return x < y ? -1 : x > y ? 1 : id(a).localeCompare(id(b));
});

export function selectAndAssign(sourceItems) {
  if (sourceItems.length !== 959 || new Set(sourceItems.map(i => i.id)).size !== 959) throw new Error('Expected 959 unique main items.');
  const selected = [], byItem = {}, stratumByItem = {};
  STRATA.forEach((stratum, index) => {
    const population = sourceItems.filter(i => i.source === stratum.source && i.dimension === stratum.dimension);
    if (population.length !== stratum.population) throw new Error(`Unexpected population: ${stratum.key}`);
    const items = ranked(population, `sample/${stratum.key}`).slice(0, stratum.sample);
    const slots = PAIRS.flatMap(pair => Array.from({ length: Math.floor(stratum.sample / 15) + Number(extraPairs[index].includes(pair.join('-'))) }, (_, n) => ({ id: `${pair.join('-')}/${n}`, pair })));
    const shuffledSlots = ranked(slots, `pair/${stratum.key}`);
    items.forEach((item, n) => {
      selected.push(item);
      byItem[item.id] = shuffledSlots[n].pair;
      stratumByItem[item.id] = stratum.key;
    });
  });
  const reviewers = Object.fromEntries(REVIEWERS.map(id => [id, ranked(selected.filter(item => byItem[item.id].includes(id)), `order/${id}`).map(item => item.id)]));
  return {
    sample: { version: VERSION, seed: SEED, items: selected },
    assignments: { version: VERSION, reviewers, by_item: byItem, stratum_by_item: stratumByItem },
  };
}

export function validateAllocation(sample, assignments) {
  const ids = sample.items.map(i => i.id);
  if (sample.version !== VERSION || assignments.version !== VERSION || ids.length !== 150 || new Set(ids).size !== 150) throw new Error('Invalid frozen sample.');
  if (Object.keys(assignments.reviewers).sort().join() !== REVIEWERS.join()) throw new Error('Expected A1–A6.');
  for (const reviewer of REVIEWERS) {
    const assigned = assignments.reviewers[reviewer];
    if (assigned.length !== 50 || new Set(assigned).size !== 50 || assigned.some(id => !ids.includes(id))) throw new Error(`Invalid assignment: ${reviewer}`);
  }
  const counts = Object.fromEntries(PAIRS.map(p => [p.join('-'), 0]));
  for (const id of ids) {
    const owners = REVIEWERS.filter(r => assignments.reviewers[r].includes(id));
    if (owners.length !== 2 || owners.join() !== assignments.by_item[id]?.join()) throw new Error(`Expected two reviewers: ${id}`);
    counts[owners.join('-')]++;
  }
  if (Object.values(counts).some(n => n !== 10)) throw new Error('Each pair must share 10 items.');
  for (const s of STRATA) {
    if (sample.items.filter(i => i.source === s.source && i.dimension === s.dimension).length !== s.sample) throw new Error(`Wrong quota: ${s.key}`);
  }
}

export async function loadFrozen(directory = frozenDir) {
  const [sampleBytes, assignmentBytes, manifestBytes] = await Promise.all(['sample.json', 'assignments.json', 'manifest.json'].map(name => readFile(resolve(directory, name))));
  const manifest = JSON.parse(manifestBytes);
  if (hash(sampleBytes) !== manifest.sample_sha256 || hash(assignmentBytes) !== manifest.assignments_sha256) throw new Error('Frozen audit files have changed. Restore the original files.');
  const sample = JSON.parse(sampleBytes), assignments = JSON.parse(assignmentBytes);
  validateAllocation(sample, assignments);
  return { sample, assignments, manifest };
}

async function main() {
  const sourceBytes = await readFile(sourcePath);
  try {
    const existing = await loadFrozen();
    if (existing.manifest.source_sha256 !== hash(sourceBytes)) throw new Error('Main release hash changed.');
    console.log(`Verified existing ${VERSION}: 150 items, 6 reviewers × 50, 2 reviewers per item. No files overwritten.`);
    return;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const { sample, assignments } = selectAndAssign(sourceBytes.toString('utf8').trim().split(/\r?\n/).map(line => JSON.parse(line)));
  validateAllocation(sample, assignments);
  const sampleText = JSON.stringify(sample, null, 2) + '\n';
  const assignmentText = JSON.stringify(assignments, null, 2) + '\n';
  const manifest = {
    version: VERSION, frozen_at: new Date().toISOString(), seed: SEED,
    method: 'SHA-256 ranking: seed + NUL + domain + NUL + item ID. Separate sample, pair-slot and reviewer-order domains.',
    source: 'construction/data/main_split.jsonl', source_sha256: hash(sourceBytes),
    sample_sha256: hash(sampleText), assignments_sha256: hash(assignmentText),
    strata: STRATA, total_items: 150, total_reviews: 300, reviewers: REVIEWERS,
    reviews_per_reviewer: 50, reviewers_per_item: 2, items_per_reviewer_pair: 10,
    excluded: ['condition_supplement', 'answer-card stage', 'external G controls'],
  };
  await mkdir(frozenDir, { recursive: true });
  // Exclusive creation protects an existing freeze, including a partially written one.
  for (const [name, text] of [['sample.json', sampleText], ['assignments.json', assignmentText], ['manifest.json', JSON.stringify(manifest, null, 2) + '\n']]) await writeFile(resolve(frozenDir, name), text, { flag: 'wx' });
  console.log(`Frozen ${VERSION} at ${frozenDir}`);
  console.table(Object.fromEntries(REVIEWERS.map(r => [r, Object.fromEntries(STRATA.map(s => [s.key, assignments.reviewers[r].filter(id => assignments.stratum_by_item[id] === s.key).length]))])));
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(error.message); process.exitCode = 1; });
