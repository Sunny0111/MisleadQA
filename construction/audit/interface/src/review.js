export const CHECKS = [
  { key: 'q1', passage: 'G', title: 'Passage G', question: 'Does G clearly support the gold answer in the target scope?', hint: 'Check the answer and the entity, time, location, or condition being asked about.' },
  { key: 'q2', passage: 'S', title: 'Passage S', question: 'Does S clearly support the lure answer in a different scope from G?', hint: 'Only one scope element should differ: entity, time, location, or condition.' },
  { key: 'q3', passage: 'F', title: 'Passage F', question: 'Does F clearly support the lure answer in the same scope as G?', hint: 'F should match G except for replacing the gold answer with the lure answer.' },
];
export const blankReview = () => ({ q1: null, q2: null, q3: null, note: '', no_reason: '' });
export const isComplete = review => Boolean(review && CHECKS.every(c => ['yes', 'no'].includes(review[c.key])) && (!CHECKS.some(c => review[c.key] === 'no') || (['issue', 'uncertain'].includes(review.no_reason) && review.note.trim())));
export const completedCount = reviews => Object.values(reviews).filter(isComplete).length;
export function displayItem(item) {
  return { id: item.id, question: item.question, options: item.options, gold_answer: item.gold_answer, lure_answer: item.lure_answer, answer_key: item.answer_key, target_scope: item.target_scope, alternate_scope: item.alternate_scope,
    passages: { G: item.evidence.correct.text, S: item.evidence.mis_scoped.text, F: item.evidence.false.text } };
}
