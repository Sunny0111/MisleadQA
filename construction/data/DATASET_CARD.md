# MisleadQA release 1.0.0

Schema: 1.1.0. The two JSONL files are frozen inputs; adapting the evaluation
software does not rewrite their content. Hashes are in hashes.txt.

The release contains 984 items in total: 959 main items and 25 condition items.
All model measurements use this same v1.0.0 release, with the two splits evaluated
and reported separately. The dataset version field was normalized to 1.0.0;
questions, evidence, options, scoring keys and grouping did not change.
`version_normalization.json` records the original and normalized file hashes and
the invariant dataset-content hashes. Schema version remains 1.1.0.

| File | Records | Composition |
| --- | ---: | --- |
| main_split.jsonl | 959 | FRED time 200, geography 200; SituatedQA time 183, geography 184; AmbigDocs entity 192 |
| condition_supplement.jsonl | 25 | CondAmbigQA condition 25 |

The main split contains 489 (source, cluster_id) clusters and 132 dependency
groups. The supplement contains 20 source clusters and 20 dependency groups.
There are no duplicate IDs within or between splits. Options are ordered A-E;
E is the abstention option. All scoring uses each item's answer_key.

Each record includes question, target/alternate scopes, five options, gold/lure
aliases and role mapping, G/F/S texts with hashes and intended factuality/
applicability flags, false-edit offsets, source references and grouping metadata.

Release records omit review. Consequently human-review and eligibility status
are unknown from these files alone. The separate [human audit](../audit/README.md)
has 300 completed reviews of 150 main items: 149 received two all-Yes reviews
and one remains disputed. It checks textual construction with Q1–Q3; it does
not establish factual accuracy or audit coverage of the other release items.
The stored analysis_group_size is checked against
actual group membership. analysis_group_size_evaluation is retained as inherited
metadata but cannot be independently validated without explicit eligibility
flags; it is never a model-specific B_m or A_m denominator.

G is intended to be true and applicable; F is an answer replacement at the target
scope; S is intended to be true at another scope. Structural checks validate
text hashes, substitutions, options and groups. They do not independently verify
these factuality/applicability claims. The 357 distinct referenced raw, donor
and historical archives across both splits are not supplied in this repository.
The main and supplement are evaluated and reported separately.

Superseded version 0.37.0 (schema 1.0.0, 1,103 items) is not distributed.
Its size, option assignments and hash must not be used for current release runs.
Data licensing status is recorded in [LICENSE-DATA](../../LICENSE-DATA).
