import { STRATA, selectAndAssign, hash } from '../scripts/freeze-sample.js';

export function syntheticFixture() {
  const scope = { attributes: [{ name: 'time', value: '2020' }] };
  const source = STRATA.flatMap(s => Array.from({ length: s.population }, (_, n) => ({
    id: `SYNTHETIC-${s.key}-${n}`, source: s.source, dimension: s.dimension,
    question: 'Synthetic UI test: how many lanterns were recorded in Example Town in 2020?',
    options: [{ label: 'A', text: 'One' }, { label: 'B', text: 'Two' }, { label: 'C', text: 'Three' }, { label: 'D', text: 'Four' }, { label: 'E', text: 'Insufficient information' }],
    gold_answer: 'One', lure_answer: 'Two', answer_key: { gold: 'A', lure: 'B' }, target_scope: scope,
    alternate_scope: { attributes: [{ name: 'time', value: '2021' }] },
    evidence: { correct: { text: '2020: one.' }, mis_scoped: { text: '2021: two.' }, false: { text: '2020: two.' } },
  })));
  const data = selectAndAssign(source);
  return { ...data, manifest: { sample_sha256: hash(JSON.stringify(data.sample)), assignments_sha256: hash(JSON.stringify(data.assignments)) } };
}
