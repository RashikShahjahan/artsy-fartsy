# Experiment log

Record completed experiments here with their setup, results, interpretation, and supporting artifacts. Append future entries with sequential experiment numbers; retain the scope and caveats of earlier results.

## Experiment 1 — Qwen3.5-4B LoRA rank: 32 vs 64 vs 128

- **Recorded:** 2026-11-04 (America/Toronto).
- **Run dates:** 2026-11-03–2026-11-04.
- **Status:** Complete: all 391 prompts evaluated on all four models and judged.
- **Question:** How does LoRA rank affect prompt adherence, visual quality, final-image reliability, and generation efficiency?
- **Comparison:** SFT LoRA ranks 32, 64, and 128 on Qwen/Qwen3.5-4B. The base model was included as a control in generation and four-way judging.
- **Recommended choice:** Rank 64 for these checkpoints, based on the observed quality/efficiency tradeoff. Quality superiority over the other ranks is not statistically established.

### Setup and provenance

| Setting | Value |
|---|---|
| Training examples | 1,763 trajectories from 906 unique prompts |
| Training seed | 42 |
| Batch size | 32 trajectories |
| Learning rate | 0.0002 with cosine schedule |
| Training duration | One epoch, 56 updates per rank |
| Controlled comparison | Saved training configurations differ only in `lora_rank` |
| Evaluation dataset | PartiPrompts, 391 prompts: original 163 plus 228 formerly reserved for preference tuning |
| Leakage check | No overlapping IDs or normalized prompt text with the exact training dataset used by the three checkpoints; training-file hash matches all three manifests |
| Generation settings | Temperature 0; 25 model turns per episode; 32,768 output tokens per response; one epoch; same seed per prompt across models |
| Concurrency | Up to 10 samples, sandboxes, and model connections per model run |
| Overall token/time limits | No per-episode total-token or time limit |
| Judge | `openai/gpt-6-luna`, temperature 0, maximum 8,192 response tokens |
| Judge inputs | Original drawing request and all available final images for that prompt in one request; shuffled anonymous labels; no drawing code or tool history |
| Judge outputs | Separate 0–4 adherence and visual-quality scores, image comments, one preferred output, and comparison comments in one response |

The training input SHA-256 was `87290b018a3f4dcf4f085f4c8ec542ab8b61edf852d64e5f7c443153fc3a0fc6`.
The combined evaluation prompt-file SHA-256 was `7c5f71ab1a10b92cb6ae6755f169c20d764e1ecee8eccf320cc5109d53edf5e2`.
The pinned upstream PartiPrompts revision was `944b156abfdad7627c3221b5ec4f6a6fb060a197`.

The original 163 prompts were evaluated and judged first. The 228 prompts were subsequently moved from the preference reserve into evaluation, checked for training overlap, and evaluated and judged separately. The combined viewer reuses the original results. This is an exploratory comparison; the analysis and expanded evaluation were not preregistered.

Two higher-rank runs were interrupted when their inference bridge stopped. Each retained 30 completed outcomes and retried only the 133 errored samples. All four models ultimately finished both prompt subsets with zero unresolved execution errors. Resource accounting below separates final outcomes from superseded interrupted attempts.

### Failure breakdown

A failure here means no valid submitted final image. A low-quality submitted image is still a successful render and receives its own quality score.

| Outcome across 391 prompts | Rank 32 | Rank 64 | Rank 128 |
|---|---:|---:|---:|
| Valid final image | 378 | 376 | 378 |
| Incomplete | 13 | 13 | 12 |
| Refused | 0 | 2 | 1 |
| No-final-image rate | 3.32% | 3.84% | 3.32% |

### Overall results

Quality means below use the same 355 prompts with a valid final image from every LoRA, independently of base-model success. This differs from the viewer’s default matched comparison, which also requires a base-model image. Failure rates, preference counts, and resource usage use all 391 prompts.

| Metric | Rank 32 | Rank 64 | Rank 128 |
|---|---:|---:|---:|
| Prompt adherence /4 | 2.563 | 2.628 | 2.535 |
| Visual quality /4 | 2.501 | 2.530 | 2.451 |
| Adherence, missing outputs = 0 /4 | 2.481 | 2.501 | 2.460 |
| Visual quality, missing outputs = 0 /4 | 2.435 | 2.435 | 2.373 |
| Valid final images | 378 | 376 | 378 |
| No final image | 13 | 15 | 13 |
| Luna top pick among four outputs | 124 | 126 | 123 |
| Input tokens, final outcomes | 214,441,499 | 193,822,527 | 195,987,635 |
| Output tokens, final outcomes | 22,566,320 | 21,213,982 | 21,506,042 |
| Total tokens, final outcomes | 237,007,819 | 215,036,509 | 217,493,677 |
| Average tokens per prompt | 606,158 | 549,965 | 556,250 |
| Average turns per prompt | 20.25 | 18.64 | 19.09 |
| Used all 25 turns | 278 | 242 | 251 |

A top pick is the single preferred output in a four-model request including the base, not a head-to-head win against another LoRA. Base received the remaining 18 preferences. Missing output includes incomplete episodes and refusals; it does not mean low visual quality. All eval runs finished with zero unresolved sample errors.

Failed final outcomes are included in the resource figures. Superseded interrupted attempts are excluded from the table: rank 64 consumed an additional 2,862,723 tokens and rank 128 another 2,831,024 tokens in those attempts. Token counts include repeated context and visual input; they are not dollars or direct latency measurements.

### Statistical comparisons

Exact two-sided paired label-swap tests for integer quality scores, delivery indicators and turns. Token tests use 100,000 sign permutations. Holm correction spans all 15 main comparisons. Confidence intervals are pointwise 95% paired percentile-bootstrap intervals from 100,000 resamples, not multiplicity-adjusted intervals. The expanded missing-as-zero quality analysis reaches the same lack of a quality winner.

| Metric | Difference | Estimate | 95% CI | Adjusted p (15 tests) |
|---|---|---:|---|---:|
| adherence_score | rank64 - rank32 | 0.0648 | [-0.0394, 0.1718] | 1.0000 |
| adherence_score | rank128 - rank32 | -0.0282 | [-0.1380, 0.0789] | 1.0000 |
| adherence_score | rank128 - rank64 | -0.0930 | [-0.1972, 0.0113] | 0.9766 |
| visual_quality_score | rank64 - rank32 | 0.0282 | [-0.0592, 0.1155] | 1.0000 |
| visual_quality_score | rank128 - rank32 | -0.0507 | [-0.1352, 0.0338] | 1.0000 |
| visual_quality_score | rank128 - rank64 | -0.0789 | [-0.1662, 0.0085] | 0.9766 |
| valid_final_image | rank64 - rank32 | -0.0051 | [-0.0307, 0.0205] | 1.0000 |
| valid_final_image | rank128 - rank32 | 0.0000 | [-0.0256, 0.0256] | 1.0000 |
| valid_final_image | rank128 - rank64 | 0.0051 | [-0.0179, 0.0281] | 1.0000 |
| tokens | rank64 - rank32 | -56192.6087 | [-100548.4983, -11704.5146] | 0.2038 |
| tokens | rank128 - rank32 | -49908.2916 | [-92414.9756, -6692.2947] | 0.3034 |
| tokens | rank128 - rank64 | 6284.3171 | [-42260.7488, 54328.5997] | 1.0000 |
| turns | rank64 - rank32 | -1.6061 | [-2.6061, -0.6087] | 0.0261 |
| turns | rank128 - rank32 | -1.1637 | [-2.1765, -0.1509] | 0.3034 |
| turns | rank128 - rank64 | 0.4425 | [-0.5781, 1.4655] | 1.0000 |

The rank-64 versus rank-32 turn reduction is significant under this correction (about 1.61 fewer turns per prompt, adjusted p≈0.026). The observed 9.3% token saving is not significant after this full correction. Quality and delivery-rate differences are not significant. Statistical conclusions concern these checkpoints and this judge under a paired-prompt analysis; training-seed replication is needed to generalize to rank as a training hyperparameter.

### Original versus added prompts

| Subset | Matched LoRA prompts | Rank-32 adherence | Rank-64 adherence | Rank-128 adherence |
|---|---:|---:|---:|---:|
| Original 163 | 150 | 2.513 | 2.660 | 2.600 |
| Added 228 | 205 | 2.600 | 2.605 | 2.488 |

The rank-64 minus rank-32 adherence gap shrinks from 0.147 on the original subset to 0.005 on the added subset. Their visual-quality means are exactly tied on the 205 matched added prompts. In the added set all three models produced 220/228 final images. The new set does not establish a rank-64 quality advantage.

### Category profiles (descriptive)

These are post-hoc small subgroups; values are prompt adherence means among prompts rendered by all three LoRAs. They are not evidence of significant category-specific superiority.

| Category | Matched prompts | Rank 32 | Rank 64 | Rank 128 |
|---|---:|---:|---:|---:|
| Abstract | 11 | 3.455 | 3.182 | 3.000 |
| Animals | 70 | 2.371 | 2.457 | 2.500 |
| Artifacts | 66 | 2.591 | 2.818 | 2.424 |
| Arts | 13 | 2.462 | 3.000 | 2.769 |
| Food & Beverage | 20 | 2.800 | 2.550 | 2.750 |
| Illustrations | 27 | 2.519 | 2.630 | 2.556 |
| Indoor Scenes | 7 | 3.000 | 2.857 | 3.000 |
| Outdoor Scenes | 27 | 2.704 | 2.815 | 2.667 |
| People | 40 | 2.500 | 2.675 | 2.450 |
| Produce & Plants | 9 | 2.333 | 2.444 | 1.778 |
| Vehicles | 21 | 2.571 | 2.667 | 2.714 |
| World Knowledge | 44 | 2.500 | 2.227 | 2.455 |

### Limits and practical choice

- One checkpoint and one training seed per rank; one Luna judgment per prompt. Human calibration and further training seeds would strengthen conclusions.
- Median generation length is 25 turns for all three models. These are budget-constrained results, not estimates of the turns needed with no limit.
- Hosted generation ran concurrently, so recorded episode durations are not a controlled speed benchmark. Token and turn usage are the defensible efficiency measures here.
- No adapter memory footprint, training cost, or dollar inference cost was measured; do not infer those from rank or token count alone.
- Keep rank 64 as the practical default if choosing now, retain rank 32 as a baseline, and do not increase to rank 128 expecting a proven quality improvement.

Methods: [paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) and [Holm correction](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html). Full numerical results are linked below.

### Checkpoints and supporting artifacts

| Model | Checkpoint metadata |
|---|---|
| Rank 32 | [final.json](checkpoints/sft-parti-qwen3.5-4b-2026-11-03/final.json) |
| Rank 64 | [final.json](checkpoints/sft-parti-qwen3.5-4b-rank64-2026-11-03/final.json) |
| Rank 128 | [final.json](checkpoints/sft-parti-qwen3.5-4b-rank128-2026-11-03/final.json) |

- [Combined 391-prompt viewer](http://127.0.0.1:4173/comparisons#comparison=parti-combined-391-four-model-2026-11-04).
- [Evaluation prompts](datasets/parti-prompts/evaluation/prompts.jsonl), [original 163](datasets/parti-prompts/evaluation/original.prompts.jsonl), and [added 228](datasets/parti-prompts/evaluation/added-from-preference.prompts.jsonl).
- [Dataset split manifest](datasets/parti-prompts/manifest.json).
- [Training manifest](checkpoints/sft-parti-qwen3.5-4b-rank64-2026-11-03/training-manifest.json).
- [Original four-model judge manifest and rubric](eval-results/parti-base-rank32-rank64-rank128-luna-2026-11-03/manifest.json).
- [Added four-model judge manifest and rubric](eval-results/parti-added-four-model-luna-2026-11-04/manifest.json).
- [Full statistical analysis](eval-results/parti-combined-391-four-model-2026-11-04/lora-comparison-analysis.json).
- [Token and turn accounting](eval-results/parti-combined-391-four-model-2026-11-04/token-turn-usage.json).
- [Combined viewer collection](eval-results/parti-combined-391-four-model-2026-11-04/collection.json).
- Evaluation outputs follow `eval-results/parti-added-{base,rank32,rank64,rank128}-qwen3.5-4b-execution-only-2026-11-04/`; the original eval locations are recorded in the original judge manifest.

Datasets, checkpoints, and raw eval artifacts are Git-ignored local files. This log retains the setup and numerical findings in the repository; the artifact links require those local files. Recreate the expanded split with `python3 scripts/prepare_parti_prompts.py --preference-count 228 --preference-to-evaluation`. Generation and judging use paid model calls; reproducing dataset preparation alone does not.

### Follow-up questions

- Does the rank-64 efficiency advantage repeat across independently trained seeds?
- Do blinded human judgments agree with Luna on adherence and visual quality?
- How much do the 25-turn cap and repeated context/preview tokens affect the efficiency ranking?
- Do category differences persist on larger, prespecified category samples?
