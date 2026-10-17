# Main dataset audit: 150 items

This is a single-stage, independent item review of the main split. The sample
and assignments are frozen as `main150-v1`. All six formal submissions have
been collected; the one disputed item remains pending organizer adjudication.

## Completed reviews

The collected originals are in
[`../human audit/main150-v1/`](../human%20audit/main150-v1/):
`A1.submitted.json` through `A6.submitted.json`, submitted on 2026-10-07 and
2026-10-08. They contain 300 complete item reviews and 900 Q1–Q3 judgments.
Draft files are not additional observations.

| Check | Yes judgments | Paired agreement |
| --- | ---: | ---: |
| Q1 · G | 299/300 | 149/150 (99.33%) |
| Q2 · S | 300/300 | 150/150 (100%) |
| Q3 · F | 299/300 | 149/150 (99.33%) |

Both reviewers answered Yes to all three checks on 149/150 items. A2 marked
Q1 and Q3 No on `ambigdocs_entity_f96e8711d9ee121a_v8`, citing the different
names Jamie Elliott and James Harry Elliott; A1 answered Yes throughout.
The source archive associates both names with the same document, but the
displayed passages omit an explicit alias relation. This remains an unresolved
referential-clarity concern, not a confirmed wrong-answer error. No Unsure
responses or jointly flagged items were recorded.

The sample's any-reviewer flag rate is 1/150 (0.67%); the main-population-weighted
flag rate is (192/959) × (1/50) = 0.40%. Neither is a confirmed-error rate.
Q1/Q3 pooled-marginal Fleiss kappa is approximately −0.0033 under the extreme
Yes prevalence; Q2 kappa is undefined because all judgments are Yes. Report
raw agreement and class counts alongside kappa.

Run `node construction/audit/analysis/analyze.mjs` from the repository root to
verify frozen/source hashes, all six submission receipts, exact assignments,
complete judgments, and displayed-input equality. It regenerates
`analysis/summary.json` (generated locally and ignored by Git).
File integrity does not independently establish reviewer identities or working
conditions. Participant preparation, consent, and compensation require their
own records.

The analyzer can run without local raw source caches. If the AmbigDocs archive
is available, it checks its hash and adds source context for the disputed item;
otherwise `case_evidence` is null. Submission and frozen-sample validation still
run in full. Generated reports do not replace organizer adjudication.

## Frozen sample

| Stratum | Main items | Sampled items |
| --- | ---: | ---: |
| FRED · Time | 200 | 15 |
| FRED · Location | 200 | 15 |
| SituatedQA · Time | 183 | 35 |
| SituatedQA · Location | 184 | 35 |
| AmbigDocs · Entity | 192 | 50 |
| **Total** | **959** | **150** |

- [sample.json](frozen/main150-v1/sample.json): all 150 original item objects, including question, options, answers, scopes, and G/S/F evidence. Source fields are preserved without rewriting.
- [assignments.json](frozen/main150-v1/assignments.json): reviewer-specific display order, two owners for every item, and strata.
- [manifest.json](frozen/main150-v1/manifest.json): seed, algorithm, source SHA-256, sample and assignment hashes, quotas, and freeze time.

Sampling uses seed `20261007` and SHA-256 ranking within each stratum. It does
not depend on model responses, observed errors, or human judgments. Separate
hash domains randomize pair slots and each reviewer's display order. The
generator in `interface/scripts/freeze-sample.js` documents the exact algorithm.
Run `npm run freeze` from `interface` to verify the existing freeze; it will not
overwrite or silently resample it. Keep these three files together and unchanged.

## Assignments

Every item is assigned to **two distinct reviewers**. Each reviewer receives
**50 items**; there are **300 item reviews**. Every pair of reviewers shares
exactly 10 items. The rows below count review assignments, so column totals
are twice the sampled-item counts.

| Reviewer | FRED time | FRED location | SituatedQA time | SituatedQA location | AmbigDocs entity | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 5 | 5 | 11 | 12 | 17 | 50 |
| A2 | 5 | 5 | 11 | 12 | 17 | 50 |
| A3 | 5 | 5 | 12 | 11 | 17 | 50 |
| A4 | 5 | 5 | 12 | 11 | 17 | 50 |
| A5 | 5 | 5 | 12 | 12 | 16 | 50 |
| A6 | 5 | 5 | 12 | 12 | 16 | 50 |
| **Total reviews** | **30** | **30** | **70** | **70** | **100** | **300** |

Assign one ID to each person. Reviewers work independently and do not discuss
items or consult outside materials. They see only their assigned questions
and their own responses in the review screen. Budget approximately 100 minutes
per person at two minutes per item, plus breaks and instruction time.

## Review workflow

1. Give reviewers the [reviewer guide](verification_task.md), their ID, and the local app.
2. Each reviewer types their assigned ID (A1–A6) and sees the question, options, gold/lure answers, target/alternate scopes, all three passages, and Q1–Q3. The entry page does not display a reviewer list or anyone else's progress.
3. Reviewers answer Yes or No for each check. A No requires a note and either Clear issue or Unsure. Notes should identify the affected question; mixed issue/uncertainty cases use Unsure and explain both.
4. Drafts save automatically as local JSON. Reviewers may pause, resume, and edit before submitting.
5. Once all 50 items are complete, the reviewer submits. The app writes an immutable final JSON file, displays its path, and offers a download. Existing submissions open read-only.
6. Collect A1–A6 final files. Verify matching sample and assignment hashes, 50 records per reviewer, and exactly two distinct reviewer records for each sampled item. Do not count a missing submission as a judgment.

See [local app setup and response storage](interface/README.md).
If reviewers use different computers, distribute the same frozen files and
collect their submitted JSON files manually. Local files are saved on the
computer running the Node server; they are not synchronized across computers.

## Q1–Q3

**Q1 · Passage G**  
Does G clearly support the gold answer in the target scope?

**Q2 · Passage S**  
Does S clearly support the lure answer in a different scope from G?  
Only one scope element should differ: entity, time, location, or condition.

**Q3 · Passage F**  
Does F clearly support the lure answer in the same scope as G?  
F should match G except for replacing the gold answer with the lure answer.

The word “condition” remains in the general scope definition requested for the
interface; condition-supplement items are not in this sample. These are textual
construction checks. They are not a blind human susceptibility test or an
independent factual verification of the gold/lure labels. The new Q1 is also
not an explicit check that every other answer option is false.

## Adjudication and reporting

- Record disagreements, clear issues, and uncertainties separately. The organizer resolves contested items using Q1–Q3 and keeps the two original judgments.
- Count item-level construction issues after adjudication, not one error per reviewer response. Report denominators and per-stratum outcomes. To estimate a main-dataset rate, use population weights 200/959, 200/959, 183/959, 184/959, and 192/959.
- Report unresolved cases separately or as a lower/upper error-rate bound. Do not turn uncertainty directly into a confirmed error.
- Confirmed issues go to `construction/data/errata.md`. “Revise” and “reject” are labels for this release: do not modify frozen v1.0.0, source records, or existing model outputs. Exclusions apply only to separately reported sensitivity analyses; content fixes belong to a later release.
- This audit covers 150 main items in a single review stage. The condition supplement is outside its scope. Keep review completion separate from the pending organizer adjudication.
- Before recruitment, communicate voluntary participation, the withdrawal process, compensation, and how reviewer-ID records will be used. No compensation amount or completed consent has been invented by the app.
