# Dataset duplication audit

Historical baseline: the split statistics and source descriptions below reflect
the dataset before split removal. The subsequent implementation removed family
split assignments, migrated `sft-v1` checkpoint metadata, and retained all 853
accepted examples in one `accepted.jsonl`; the three split exports were removed.

Audited 2026-10-24 using three independent subagent investigations (exact data,
semantic repetition, and pipeline behavior), followed by a reproducible local
lexical scan. Scope: `datasets/sft-v1`, `evaluation/prompts.jsonl`, and their
generation/training code. Source data and implementation were left unchanged.

## Findings

There is noticeable task/motif repetition, especially among refusal requests and
simple drawings. Exact duplication is already well controlled. The strongest
confirmed defect is the planner's coupled family/difficulty/skill schedule.

| Check | Result |
| --- | --- |
| Planned prompts | 1,000; all unique after NFKC, casefold, whitespace normalization |
| Accepted records | 853: 612 train, 150 validation, 91 test |
| Accepted drawings | 753; source hashes, image hashes and pixel hashes are individually unique |
| Authored refusal pairs | 100 unique requests, 66 distinct normalized answers |
| Exact overlap between retained splits | Zero IDs, normalized prompts, source/image/pixel hashes |
| Exact overlap with the 20 evaluation prompts | Zero normalized prompts |
| Conservative lexical similarity flags | 59 pairs touching 35 planned prompts |
| Same scan restricted to accepted prompts | 58 pairs touching 34 accepted prompts |
| Flagged cross-split pairs in that scan | Zero |

The lexical scan is a review aid, not a semantic duplicate estimate. It uses NFKC,
casefold and whitespace normalization; unique `[a-z0-9]+` tokens (including short
words); token Jaccard >= 0.55 AND Python `difflib.SequenceMatcher` ratio >= 0.70,
with default `autojunk=True`. Its exact methodology and all flagged ID pairs are
in `dataset-duplication-audit.json`. These thresholds deliberately differ from the
generator's stricter rejection thresholds. Different normalizations/tokenizers
change results; neither this scan nor the existing filter proves semantic separation.

Of the 59 pairs, 53 are refusal requests (touching 25 refusal prompts), three are
ring requests and three are triangle requests. Some geometry pairs specify
different constraints and should be retained. Repeated concise refusal answers
are valid shared targets; deduplicating answers alone would discard distinct
requests. Refusal-request diversity needs review independently of answer wording.

Broader motif repetition exceeds what this lexical rule detects. All 84 landscape
prompts are easy; 64 mention a sun and 31 mention both a sun and a hill (case
insensitive whole-word matching, allowing plural `s`). For example:

- `39625252829ea0779d77dc03`: pale blue sky, broad green hill in the lower third,
  warm yellow sun slightly right of center, no extra scenery.
- `2a89f76203ba105593f34a2a`: pale blue sky, broad green hill in the bottom quarter,
  golden sun in the upper right, no trees/buildings/clouds.

These illustrate a repeated task concept with changed details, not identical
specifications. Similarly, the evaluator includes a hill/sky/sun task. Exact
non-overlap therefore does not establish that evaluation tasks are novel.

## What causes it

1. `drawing/llm_prompts.py::_slots` advances family, skill and difficulty from the
   same index. Twelve scene families versus four difficulties means every family
   always receives one difficulty. Fourteen primary skills also produce strong
   correlations. Selecting refusal slots at a periodic interval removes ordinary
   slots nonuniformly: all 100 refusal slots here are medium or very-hard.
2. The fixed family split in `drawing/dataset_specs.py::FAMILIES` propagates this
   coupling into the holdouts. Accepted validation contains 80 easy and 70 hard
   examples, with zero medium/very-hard; test contains 63 medium and 28 very-hard,
   with zero easy/hard. This confounds difficulty with split and family.
3. `_too_similar` checks wording: token Jaccard >= 0.82, or Jaccard >= 0.55 plus
   sequence similarity >= 0.88. It checks all accepted prompts in the current run,
   under a lock. The recent-30 list is only a generation hint. Paraphrases, repeated
   concepts, other dataset runs and the evaluation prompt bank are not semantically
   checked by this gate.
4. Output deduplication checks exact source and decoded pixels. Color, coordinates
   or wording changes can yield a similar task with different hashes. These checks
   are useful but do not measure curriculum diversity.

The family/difficulty mapping in this run:

| Difficulty | Scene families |
| --- | --- |
| easy | landscapes, rings, triangles |
| medium | flowers, lines, orbits |
| hard | circles, grids, rows |
| very-hard | columns, houses, rectangles |

## Recommended repair, in order

1. **Fix planning before buying more generations.** Allocate ordinary/refusal slots
   first. Independently balance scene family, primary skill and difficulty using
   seeded schedules with explicit family-by-difficulty quotas; use a seeded
   permutation to order those slots. Balance refusal topic quotas separately.
   Independent shuffles alone do not guarantee joint coverage. Version the planner
   and require a fresh output directory so old frozen checkpoints stay valid.
2. **Gate task diversity before teacher/render calls.** Retrieve similar requests
   across the full plan, prior datasets and the evaluation bank. Compare task
   concepts and explicit constraints as well as wording. Use a semantic review
   step for ambiguous pairs: a changed count, tangency condition or occlusion order
   can teach a new constraint even when the scene is similar. Start perceptual
   image similarity as a review signal and validate its false positives before
   using it to reject records. Supply relevant nearest examples to avoid, rather
   than only the last 30 prompts. Add topic/motif quotas to prevent repeated poster
   requests or hill-and-sun scenes from consuming the curriculum.
3. **Curate a separate derived release from existing records.** Review flagged
   pairs and broader semantic groups, keep distinct constraint/repair examples,
   and select representatives of redundant tasks using correctness and trajectory
   value. Save retained IDs, excluded IDs, reasons and grouping policy. Preserve
   the original checkpoints and artifacts. The 59 pair flags are not 59 deletions.
4. **Make split intent explicit.** For a balanced benchmark, assign whole semantic
   task groups to splits and stratify by difficulty/skill, keeping the evaluation
   bank excluded from training. Keep related variants together. If family holdout
   is intentional, retain it as a separately named generalization benchmark and
   fill all difficulty levels within each heldout family. This requires changing
   the current family-only split contract and its validators, not merely editing
   JSONL split labels.
5. **Add regression checks.** Test per-family difficulty coverage and primary-skill
   coverage across several seeds/counts; refusal counts and topic allocation;
   paraphrase detection versus meaningful geometry variants; evaluation-bank
   exclusion; and reproducible resume behavior. Report cluster sizes and
   family/skill/difficulty cross-tabs alongside acceptance rate.

Further hardening: apply source/pixel dedup in stable plan order after concurrent
candidate generation, since completion order currently chooses the surviving
record; add duplicate-content checks when loading assembled training files (the
loader currently checks duplicate IDs). These are prevention gaps, not evidence
of retained exact duplicates in this release.

## Apparent duplication that should remain

The six JSONL exports (`candidates`, `accepted`, `rejected`, `train`, `validation`,
`test`) contain 2,853 rows representing 1,000 candidate IDs. Accepted examples
occur in candidates, accepted, and exactly one split; rejected examples occur in
candidates and rejected. They are intentional views. Train from `train.jsonl`;
concatenating all exports would duplicate records and include holdouts.

There are 778 final-image artifact PNGs and 776 distinct PNG byte hashes. One
group contains three identical validation ring images: one accepted and two
already rejected as duplicates. Keeping those rejected artifacts preserves the
audit trail; deleting them would not improve retained training diversity.
