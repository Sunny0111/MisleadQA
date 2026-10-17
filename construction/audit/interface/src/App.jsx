import { useEffect, useMemo, useState } from 'react';
import { CHECKS, completedCount, isComplete } from './review.js';
import { request, DraftWriter } from './local-api.js';
import { download } from './storage.js';
import Icon from './Icon.jsx';
import Modal from './Modal.jsx';

function Brand() {
  return <div className="brand"><span className="brand-icon"><Icon name="leaf" size={23}/></span><span>MisleadQA<span className="brand-sub">Human review workspace</span></span></div>;
}

function Guide() {
  return <div className="guide">
    <p>Review each item independently. Read the question, options, and all three passages before answering Q1–Q3.</p>
    <ol>
      <li><strong>Check what the passage says.</strong> Treat the underlying source facts as given; do not search online or use outside materials to verify them. F may deliberately state a false answer.</li>
      <li><strong>Check the scope.</strong> Scope means the entity, time, location, or condition the question or passage refers to. Judge the text itself, even when a scope label is provided.</li>
      <li><strong>Choose Yes only when the whole check is met.</strong> For S, a missing answer or vague wording is not a clear change of scope. For F, look for other edits, a leftover gold answer, or broken grammar.</li>
      <li><strong>If unsure, choose No.</strong> Explain which check is affected and mark whether you found a clear issue, are unsure, or both (choose Unsure and explain).</li>
    </ol>
    <p>Your assignment contains 50 items. You may pause and resume using the same reviewer ID on this computer. Review your answers before final submission; submitted reviews are locked.</p>
  </div>;
}

function Scope({ scope, label }) {
  const attributes = scope?.attributes || [];
  return <div className="scope"><span className="eyebrow">{label}</span><div className="scope-values">{attributes.map(attribute => <span key={attribute.name}><b>{attribute.name.replaceAll('_', ' ')}</b> {String(attribute.value)}</span>)}</div></div>;
}

function ReviewItem({ item, review, update, locked }) {
  const hasNo = CHECKS.some(check => review[check.key] === 'no');
  return <>
    <section className="question-card" aria-labelledby="question-title">
      <span className="eyebrow">Question</span>
      <h2 id="question-title">{item.question}</h2>
      <div className="options">{item.options.map(option => <div key={option.label} className="option"><span>{option.label}</span><p>{option.text}</p></div>)}</div>
      <div className="answer-key"><div><span className="eyebrow">Gold answer</span><strong className="gold-value">{item.answer_key.gold} · {item.gold_answer}</strong></div><div><span className="eyebrow">Lure answer</span><strong className="lure-value">{item.answer_key.lure} · {item.lure_answer}</strong></div></div>
      <Scope scope={item.target_scope} label="Target scope"/>
    </section>
    <div className="section-caption"><h2>Read the evidence. Check each passage.</h2><span>3 checks per item</span></div>
    {CHECKS.map((check, index) => <section className={`passage-card passage-${check.passage}`} key={check.key} aria-labelledby={`${check.key}-title`}>
      <div className="passage-text">
        <div className="passage-heading"><span className={`passage-letter letter-${check.passage}`}>{check.passage}</span><div><h3 id={`${check.key}-title`}>{check.title}</h3><span className="muted">{index === 0 ? 'Gold answer · target scope' : index === 1 ? 'Lure answer · alternate scope' : 'Lure answer · target scope'}</span></div></div>
        <p className="evidence">{item.passages[check.passage]}</p>
        {index === 1 && <Scope scope={item.alternate_scope} label="Alternate scope"/>}
      </div>
      <div className="check-panel">
        <span className="eyebrow">Q{index + 1} · {check.title}</span>
        <fieldset disabled={locked}>
          <legend>{check.question}</legend>
          <p className="check-hint">{check.hint}</p>
          <div className="choices">{['yes', 'no'].map(value => <label className={`choice choice-${value}${review[check.key] === value ? ' selected' : ''}`} key={value}><input type="radio" name={`${item.id}-${check.key}`} value={value} checked={review[check.key] === value} onChange={() => update({ [check.key]: value })}/><span>{value === 'yes' ? 'Yes' : 'No'}</span></label>)}</div>
        </fieldset>
      </div>
    </section>)}
    <section className="note-card">
      <div className="note-heading"><label htmlFor="review-note">Review note <span className="muted">{hasNo ? 'Required for No' : 'Optional'}</span></label><Icon name="file"/></div>
      {hasNo && <fieldset className="reason" disabled={locked}><legend>Why did you choose No?</legend>{[['issue', 'Clear issue'], ['uncertain', 'Unsure']].map(([value, label]) => <label key={value}><input type="radio" name="no-reason" value={value} checked={review.no_reason === value} onChange={() => update({ no_reason: value })}/>{label}</label>)}</fieldset>}
      <textarea id="review-note" rows={3} maxLength={5000} disabled={locked} value={review.note} onChange={event => update({ note: event.target.value })} placeholder={hasNo ? 'Mention Q1, Q2, or Q3 and briefly explain the issue or uncertainty.' : 'Anything the organizer should know about this item?'} />
    </section>
  </>;
}

function Workspace({ session, onLeave }) {
  const [status, setStatus] = useState('saved');
  const [error, setError] = useState('');
  const [modal, setModal] = useState(null);
  const [busy, setBusy] = useState(false);
  const [submission, setSubmission] = useState(session.submission);
  const writer = useMemo(() => new DraftWriter(session, { onStatus: (next, message) => { setStatus(next); setError(message || ''); } }), [session]);
  const [draft, setDraft] = useState(writer.initial);
  const count = completedCount(draft.reviews), total = session.items.length;
  const cursor = draft.cursor, item = session.items[cursor], review = draft.reviews[item.id];
  useEffect(() => { if (!session.submission) writer.flush().catch(() => {}); }, [writer, session]);
  useEffect(() => {
    const protect = event => { if (writer.dirty) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', protect);
    return () => window.removeEventListener('beforeunload', protect);
  }, [writer]);
  function change(next) { setDraft(next); if (!submission) writer.change(next); }
  function update(patch) {
    const next = { ...review, ...patch };
    if (CHECKS.every(check => next[check.key] !== 'no')) next.no_reason = '';
    change({ ...draft, reviews: { ...draft.reviews, [item.id]: next } });
  }
  function navigate(index) { change({ ...draft, cursor: index }); window.scrollTo({ top: 0, behavior: 'instant' }); }
  async function leave() { try { await writer.flush(); onLeave(); } catch { /* Keep unsaved edits visible. */ } }
  async function submit() {
    setBusy(true);
    try { await writer.flush(); const receipt = await request(`/reviewers/${session.reviewer_id}/submit`, { method: 'POST', body: JSON.stringify({ revision: writer.revision }) }); setSubmission(receipt); setModal(null); }
    catch (err) { setError(err.message); setModal(null); }
    finally { setBusy(false); }
  }
  return <div className="workspace">
    <aside className="sidebar">
      <Brand/>
      <div className="reviewer-label"><span className="avatar">{session.reviewer_id}</span><div><strong>Reviewer {session.reviewer_id}</strong><span className="muted">Main dataset review</span></div></div>
      <div className="progress-label"><strong>{count} of {total}</strong><span>items complete</span></div><progress value={count} max={total}/>
      <div className="item-grid" aria-label="Assigned items">{session.items.map((entry, index) => <button key={entry.id} className={`${index === cursor ? 'current ' : ''}${isComplete(draft.reviews[entry.id]) ? 'complete' : ''}`} aria-label={`Item ${index + 1}${isComplete(draft.reviews[entry.id]) ? ', complete' : ', incomplete'}`} aria-current={index === cursor ? 'step' : undefined} onClick={() => navigate(index)}>{index + 1}</button>)}</div>
      <div className="grid-legend"><span><i/> Complete</span><span><i className="outline"/> Current</span></div>
      <div className="sidebar-bottom"><button className="text-button" onClick={() => setModal('guide')}><Icon name="book"/> Review instructions</button><button className="text-button" onClick={leave}><Icon name="back"/> Exit review</button><div className="local-label"><Icon name="shield" size={15}/> Saved on this computer</div></div>
    </aside>
    <main className="review-main">
      <header className="topbar"><div><span className="eyebrow">Item review</span><h1>Item {String(cursor + 1).padStart(2, '0')} <span>/ {total}</span></h1></div><div className={`save-state ${status}`} role="status"><Icon name={submission ? 'lock' : status === 'saved' ? 'check' : 'clock'}/>{submission ? 'Submitted · locked' : status === 'saved' ? (session.draft.revision || writer.revision ? 'Saved locally' : 'Ready to review') : status === 'saving' ? 'Saving to this computer…' : 'Changes not saved'}</div></header>
      {error && <div className="alert" role="alert"><strong>{error}</strong><div><button onClick={() => writer.flush().catch(() => {})}>Retry save</button><button onClick={() => download(draft, `${session.reviewer_id}.recovery.json`)}>Download recovery copy</button></div></div>}
      {submission && <div className="receipt" role="status"><Icon name="check" size={24}/><div><strong>Your review has been submitted.</strong><p>All {total} items are saved as JSON. You can view your answers below.</p><code>{submission.path}</code><a href={`/api/reviewers/${session.reviewer_id}/export`} download>Download submitted JSON <Icon name="download" size={15}/></a></div></div>}
      <ReviewItem item={item} review={review} update={update} locked={Boolean(submission)}/>
      <div className="item-id">Item ID <code>{item.id}</code></div>
      <footer className="navigation"><button className="button secondary" disabled={cursor === 0} onClick={() => navigate(cursor - 1)}><Icon name="back"/> Previous</button><span>{isComplete(review) ? <><Icon name="check"/> Item complete</> : 'Answer all three checks'}</span>{cursor < total - 1 ? <button className="button primary" onClick={() => navigate(cursor + 1)}>Next item <Icon name="arrow"/></button> : !submission ? <button className="button primary" disabled={count !== total || status !== 'saved'} onClick={() => setModal('submit')}>Submit {total} items <Icon name="check"/></button> : <button className="button primary" onClick={leave}>Done <Icon name="check"/></button>}</footer>
      {cursor === total - 1 && count < total && <button className="text-button incomplete-link" onClick={() => navigate(session.items.findIndex(entry => !isComplete(draft.reviews[entry.id])))}>Go to the first incomplete item <Icon name="arrow"/></button>}
      <div className="file-location"><Icon name="file" size={14}/><span>Local JSON folder: <code>{session.directory}</code></span></div>
    </main>
    {modal === 'guide' && <Modal title="Review instructions" onClose={() => setModal(null)}><Guide/></Modal>}
    {modal === 'submit' && <Modal title="Submit your review" onClose={busy ? undefined : () => setModal(null)}><p>All {total} items are complete. Submit as Reviewer {session.reviewer_id}?</p><p>Your final JSON file will be saved on this computer. After submission, these answers will be locked.</p><div className="modal-actions"><button className="button secondary" disabled={busy} onClick={() => setModal(null)}>Keep reviewing</button><button className="button primary" disabled={busy} onClick={submit}>{busy ? 'Saving…' : 'Submit review'}</button></div></Modal>}
  </div>;
}

export default function App() {
  const [config, setConfig] = useState(null), [session, setSession] = useState(null);
  const [reviewerId, setReviewerId] = useState(''), [error, setError] = useState(''), [idError, setIdError] = useState(''), [busy, setBusy] = useState(false);
  const [guide, setGuide] = useState(false);
  async function refresh() { try { setConfig(await request('/config')); setError(''); } catch (err) { setError(err.message); } }
  useEffect(() => { refresh(); }, []);
  async function open(event) {
    event.preventDefault();
    const id = reviewerId.trim().toUpperCase();
    if (!/^A[1-6]$/.test(id)) { setIdError('Enter your assigned reviewer ID, from A1 to A6.'); return; }
    setIdError(''); setReviewerId(id);
    setBusy(true); setError('');
    try { setSession(await request(`/reviewers/${id}`)); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  if (session) return <Workspace key={session.reviewer_id} session={session} onLeave={() => { setSession(null); setReviewerId(''); setIdError(''); setError(''); }}/ >;
  return <div className="welcome"><header><Brand/><span className="local-label"><Icon name="shield" size={15}/> Local workspace</span></header><main className="welcome-main"><div className="welcome-copy"><span className="eyebrow">MisleadQA · Human audit</span><h1>A careful read.<br/>Three clear checks.</h1><p>Review the question and its three passages. Check the answers and scope, then record your judgment.</p><div className="welcome-facts"><div><strong>50</strong><span>items per reviewer</span></div><div><strong>3</strong><span>checks per item</span></div><div><strong>JSON</strong><span>saved locally</span></div></div><button className="text-button" onClick={() => setGuide(true)}><Icon name="book"/> Read the review instructions <Icon name="arrow" size={16}/></button></div>
    <section className="start-card">
      <span className="eyebrow">Your assignment</span>
      <h2>Enter your reviewer ID</h2>
      <p>Use only the ID assigned to you by the organizer. Your progress will be saved on this computer.</p>
      <form onSubmit={open} noValidate>
        <div className="reviewer-field">
          <label htmlFor="reviewer-id">Reviewer ID</label>
          <input id="reviewer-id" name="reviewerId" type="text" value={reviewerId} onChange={event => { setReviewerId(event.target.value); setIdError(''); }} placeholder="Enter your ID" autoComplete="off" autoCapitalize="characters" spellCheck={false} disabled={busy} aria-invalid={Boolean(idError)} aria-describedby={idError ? 'reviewer-id-help reviewer-id-error' : 'reviewer-id-help'}/>
          <p id="reviewer-id-help">Enter your assigned ID (A1–A6) to open or resume your review.</p>
          {idError && <p id="reviewer-id-error" className="field-error" role="alert">{idError}</p>}
        </div>
        {error && <div className="alert" role="alert">{error}<button type="button" onClick={refresh}>Reconnect</button></div>}
        <button type="submit" className="button primary start-button" disabled={!reviewerId.trim() || busy || !config}>{busy ? 'Opening…' : 'Open my assignment'}<Icon name="arrow"/></button>
      </form>
      <div className="start-note"><Icon name="info" size={16}/><span>Work independently. If you are unsure, choose No and leave a brief note.</span></div>
    </section></main><footer className="welcome-footer"><span>150 main-dataset items · 6 reviewers · 2 reviews per item</span><span>Frozen assignment · {config?.version || 'main150-v1'}</span></footer>{guide && <Modal title="Review instructions" onClose={() => setGuide(false)}><Guide/></Modal>}</div>;
}
