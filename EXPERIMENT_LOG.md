# Experiment log

Record experiments here with their setup, results, interpretation, and supporting artifacts. Mark planned runs explicitly and append future entries with sequential experiment numbers; retain the scope and caveats of earlier results.

## Experiment 1 — Original 163-prompt LoRA rank evaluation

- **Scope:** Original 163 held-out PartiPrompts, evaluated on the base model and SFT ranks 32, 64 and 128. The subsequent 228-prompt extension is Experiment 2.
- **Training:** Same three checkpoints and controlled training setup documented in Experiment 2 below, trained on 1,763 trajectories from 906 unique prompts with seed 42.
- **Generation:** Temperature 0, 25-turn limit, one saved outcome per model and prompt.
- **Luna judging:** Complete. Anonymous four-way image judging with separate 0–4 adherence and visual-quality scores.
- **HPSv3 judging:** Complete on the original 163 prompts, scoring all 616 available images (base plus three LoRAs), using 4-bit MLX language weights, bfloat16 vision weights and float32 reward head on the Mac GPU. Original-prompt scores saved before the scope change were reused.
- **Interpretation:** Rank 64 has the highest observed means. This exploratory result does not establish a quality winner. The original statistical artifact uses 133 prompts rendered by all four models and corrects across 12 comparisons. The descriptive table below instead uses the 150 prompts rendered by all three LoRAs.

| Metric | Rank 32 | Rank 64 | Rank 128 |
|---|---:|---:|---:|
| Adherence /4, 150 matched LoRA prompts | 2.513 | 2.660 | 2.600 |
| Visual quality /4, 150 matched LoRA prompts | 2.473 | 2.540 | 2.473 |
| Valid final image /163 | 158 | 156 | 158 |

- [Original prompts](datasets/parti-prompts/evaluation/original.prompts.jsonl).
- [Original Luna rubric and provenance](eval-results/parti-base-rank32-rank64-rank128-luna-2026-11-03/manifest.json).
- [Original statistical analysis](eval-results/parti-base-rank32-rank64-rank128-luna-2026-11-03/statistical-analysis.json).
- [HPSv3 run log](eval-results/parti-hpsv3-mlx-4bit/run.log) and [run manifest](eval-results/parti-hpsv3-mlx-4bit/manifest.json).

### HPSv3 results

| Pair, first model wins | Wins–losses | Win rate | Pointwise 95% CI | Holm-adjusted p |
|---|---:|---:|---|---:|
| Rank 64 vs rank 32 | 77–75 | 50.7% | 42.8%–58.5% | 1.000 |
| Rank 64 vs rank 128 | 80–73 | 52.3% | 44.4%–60.0% | 1.000 |
| Rank 32 vs rank 128 | 74–80 | 48.1% | 40.3%–55.9% | 1.000 |

These comparisons include only prompts where both models delivered an image. Two-sided exact sign tests compare wins and losses, with Holm correction over the three comparisons. Win-rate intervals are pointwise Wilson 95% intervals. Missing drawings are excluded from these preference comparisons and remain visible in the delivery counts above.

HPSv3 does not establish a quality winner among the LoRAs. On the 67 rank-32/rank-64 pairs given equal visual-quality scores by Luna, HPSv3 split 34–33 in favor of rank 32. This is an exploratory analysis under one quantized scorer, with one checkpoint per rank. Quantization preserved the three-image smoke-test ranking but changed absolute rewards.

[Per-image HPSv3 scores](eval-results/parti-hpsv3-mlx-4bit/scores.csv) and [comparison summary](eval-results/parti-hpsv3-mlx-4bit/summary.json).

### Mean HPSv3 rewards

Higher raw reward is better. The available-image mean uses every delivered image for each rank. The matched mean uses the same 150 prompts with a valid image from all three LoRAs, giving a directly comparable prompt set. Missing images are excluded from both means.

| Model | Available images | Mean reward, available images | Mean reward, same 150 prompts |
|---|---:|---:|---:|
| Rank 32 | 158 | -1.564 | -1.604 |
| Rank 64 | 156 | -1.427 | -1.482 |
| Rank 128 | 158 | -1.307 | -1.370 |

Rank 128 has the highest observed mean reward under both denominators, followed by rank 64 and rank 32. This agrees with the ordering of Rashik's blinded human preference scores in Experiment 4. These are descriptive means; no significance test of mean reward differences has been performed.

Mean reward uses the magnitude of reward differences. Pairwise wins and Bradley–Terry ratings use which score is higher. Rank 64 has the highest fitted Elo, while rank 128 has the highest mean reward. The Elo lead therefore should not be described as a general HPSv3 quality preference. The existing pairwise tests remain inconclusive.

### HPSv3 Bradley–Terry ratings on the Elo scale

The fit uses 459 available LoRA pair matches across the original 163 prompts. Missing-image comparisons are excluded. The mean rating is fixed at 1500, an arbitrary shared reference. Pointwise 95% percentile intervals come from 10,000 whole-prompt bootstrap resamples with seed 42, retaining all within-prompt comparisons together.

| Model | Elo | Pointwise 95% interval |
|---|---:|---|
| Rank 64 | 1506.8 | [1478.6, 1535.3] |
| Rank 128 | 1499.2 | [1469.7, 1529.2] |
| Rank 32 | 1493.9 | [1462.7, 1524.6] |

Rank 64 leads rank 32 by 12.9 Elo, with paired bootstrap interval [−38.9, 64.6]. Every pairwise Elo-difference interval includes zero. These ratings summarize the quantized HPSv3 preferences for these checkpoints. Uncertainty covers prompt sampling only. The ordering remains unresolved.

[Rating estimates and pairwise intervals](eval-results/parti-hpsv3-mlx-4bit/elo.json) and [reproducible fitting script](eval-results/parti-hpsv3-mlx-4bit/fit_elo.py).

### Current dataset allocation

The 228 added prompts have been restored to the SFT **prompt input** split. The active allocation is SFT 1,142, evaluation 163, RL 327, preference 0. This operation does not create teacher trajectories or retrain checkpoints. Exact ID disjointness was checked across the active splits.

Experiment 2 remains a valid historical evaluation of its original checkpoints. Its added prompts must not be treated as held-out evaluation for future models trained on the restored SFT prompts. The archived [391 prompts](eval-results/experiment-2-extended-prompts/extended.prompts.jsonl), [added 228 prompts](eval-results/experiment-2-extended-prompts/added.prompts.jsonl), and [split manifest](eval-results/experiment-2-extended-prompts/dataset-manifest.json) preserve its provenance. Recreate the current allocation by preparing the historical expanded split and then restoring its added subset to SFT, as recorded in the active dataset manifest.

## Experiment 2 — Extended 391-prompt evaluation of LoRA ranks 32, 64 and 128

- **Recorded:** 2026-11-04 (America/Toronto).
- **Run dates:** 2026-11-03–2026-11-04.
- **Status:** Historical extended evaluation complete: all 391 prompts evaluated on all four models and judged by Luna. HPSv3 is restricted to Experiment 1.
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
- [Experiment 2 evaluation prompts](eval-results/experiment-2-extended-prompts/extended.prompts.jsonl), [original 163](datasets/parti-prompts/evaluation/original.prompts.jsonl), and [added 228](datasets/parti-prompts/evaluation/added-from-preference.prompts.jsonl).
- [Historical dataset split manifest](eval-results/experiment-2-extended-prompts/dataset-manifest.json).
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

## Experiment 3 — LoRA rank comparison with the expanded SFT dataset

- **Recorded:** 2026-11-04 (America/Toronto).
- **Status:** Training complete for ranks 32, 64, and 128: all 70 optimizer updates, final optimizer checkpoints, and final sampler exports are saved. All recorded losses are finite. Saved training configurations are identical except for LoRA rank. HPSv3 evaluation is complete for all 489 episodes, with zero unresolved sample errors.
- **Question:** How do ranks 32, 64, and 128 perform after adding the restored 228 SFT prompts to the original training data?
- **Initialization:** Three independent fresh adapters from `Qwen/Qwen3.5-4B`; no continuation from Experiment 1/2 adapters.
- **Training data:** 2,219 complete trajectories from 1,134 unique prompts: the exact original 1,763-trajectory training file (906 prompts), plus 456 new Luna teacher trajectories (228 prompts, two candidates each). Eight of the original SFT prompt inputs still lack a structurally trainable trajectory in the selected original file.
- **Teacher collection:** `openai/gpt-6-luna`, reasoning enabled, Docker renderer, maximum 32,768 output tokens per response and 12 model turns. All 456 new trajectories passed structural SFT conversion and artifact integrity checks, with no generation failures.
- **Review status:** All 2,219 selected trajectories retain pending visual review. As in the earlier rank comparison, these experimental runs explicitly use `--allow-unreviewed`; structural validation does not imply human approval.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` |
| LoRA ranks | 32, 64, 128 |
| Epochs | 1 |
| Batch size | 32 complete trajectories |
| Updates per rank | 70 (last batch: 11 trajectories) |
| Learning rate | 0.0002 with cosine decay |
| Training seed | 42 |
| Loss policy | `final-write-final-preview-submit` |
| Maximum sequence length | 262,144 tokens; no truncation; preflight maximum 36,075 tokens |
| Checkpoint cadence | Every 5 updates and at completion |

The three runs use the same dataset, seed, batch order, loss policy, and training settings; only LoRA rank differs. Compared with the earlier 56-update runs, the expanded dataset raises the one-epoch schedule to 70 updates, so cross-experiment differences combine added data and additional optimizer steps.

The combined training-file SHA-256 is `76af6b58b4b6619b5a093781e22344db234a66bc2cf28242452db7a1b701ab34`. The original source training-file SHA-256 remains `87290b018a3f4dcf4f085f4c8ec542ab8b61edf852d64e5f7c443153fc3a0fc6`.

The active 163 evaluation prompts and 327 RL prompts have no exact ID or normalized-text overlap with this training set. The 228 prompts added to the historical Experiment 2 evaluation are now training inputs and cannot serve as held-out evaluation for Experiment 3. Near-duplicate or semantic overlap has not been screened.

### Artifacts and run status

- [Combined dataset and source hashes](datasets/parti-prompts/sft-experiment-3/dataset-manifest.json).
- [Local token/image preflight](datasets/parti-prompts/sft-experiment-3/preflight.log).
- [New trajectory validation](datasets/parti-prompts/sft-restored-228/validation.json).
- [Launch commands and process metadata](datasets/parti-prompts/sft-experiment-3/launches.json).
- [Verified startup status](datasets/parti-prompts/sft-experiment-3/status.json).

| Rank | Training output | Console log |
|---|---|---|
| 32 | [Run directory](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank32-2026-11-04/) | [Log](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank32-2026-11-04.console.log) |
| 64 | [Run directory](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank64-2026-11-04/) | [Log](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank64-2026-11-04.console.log) |
| 128 | [Run directory](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank128-2026-11-04/) | [Log](checkpoints/sft-experiment3-parti-qwen3.5-4b-rank128-2026-11-04.console.log) |

All three runs completed on 2026-11-04 around 15:25 America/Toronto. Final-batch training losses were 0.59334 (rank 32), 0.59269 (rank 64), and 0.59230 (rank 128). These are training losses on the same final batch, not held-out quality measurements. Final checkpoint and sampler metadata are in each run’s `final.json`. HPSv3 evaluation completed on the original 163 held-out prompts for all three checkpoints (489 episodes). Temperature 0, 25-turn limit, 32,768 output tokens per response, and ten concurrent samples per rank match the earlier generation setup. The scorer uses the same MLX 4-bit language / bfloat16 vision / float32 reward-head pipeline as Experiment 1; its HTTP service reproduced a saved Experiment 1 reward exactly. LLM judges are disabled for these runs. Raw HPSv3 rewards are reported only for valid final images, with zero normalized preference for invalid or missing submissions. One seed per rank does not establish a general rank effect.

- [HPSv3 service verification](eval-results/experiment3-hpsv3/service-validation.json).
- [Evaluation launch commands](eval-results/experiment3-hpsv3/launches.json).
- [Live evaluation status and retry counts](eval-results/experiment3-hpsv3/pipeline.json).
- Inspect aggregation was corrected to mark missing-image raw rewards as unscored. The initial runs were interrupted and resumed from their saved logs, preserving 14 completed scored images; partial generations were restarted. Superseded attempts are retained in the original logs.
- Evaluation outputs: `eval-results/experiment3-rank{32,64,128}-hpsv3-2026-11-04/`.

- **Deferred scoring migration (2026-11-04):** Generation now resumes with all image and LLM scorers disabled, preserving 426 completed outcomes (146 rank-32, 141 rank-64, 139 rank-128). HPSv3's HTTP server was stopped. After every generation run succeeds, the runner launches `evaluation.score_hpsv3` once in the MLX environment, reuses matching pinned-model scores, scores the remaining saved images, and exits to release model/GPU memory. Remaining partial generations were restarted; completed images and original logs remain intact.
- Deferred scoring writes separate checkpoints under `eval-results/experiment3-hpsv3/records/`, plus `comparison.json` and `scores.csv`; Inspect generation logs remain the execution record. Current run launch commands predate this migration; live runner status and retry logs record the generation-only resumes.



### Completed HPSv3 evaluation

Completed 2026-11-04 at 17:26 America/Toronto. All 489 generation episodes finished with zero unresolved sample errors; 462 valid final images were scored and 27 episodes delivered no valid final image. Batch scoring reused 402 saved matching scores and has exited.

| Rank | Valid images /163 | Raw HPSv3 mean, delivered images | Normalized preference, all episodes |
|---|---:|---:|---:|
| 32 | 153 | -0.70873 | 0.44349 |
| 64 | 156 | -0.83126 | 0.44670 |
| 128 | 153 | -0.60805 | 0.44588 |

Matched delivered-image wins: rank 64 over rank 32, 76–71 (147 pairs); rank 128 over rank 32, 75–72 (147 pairs); rank 128 over rank 64, 81–65 (146 pairs). These are descriptive comparisons without significance tests or training-seed replication. Raw means use different delivered-image subsets; the paired comparisons use shared prompts.

[Comparison summary](eval-results/experiment3-hpsv3/comparison.json), [per-episode scores](eval-results/experiment3-hpsv3/scores.csv), and [completed pipeline status](eval-results/experiment3-hpsv3/pipeline.json).

### Invalid-output inspection

The 27 missing final images comprise 16 submitted programs that failed final rendering, 9 episodes that exhausted 25 model turns without submitting, and 2 benign requests ended with `refuse_drawing` containing visual-quality self-critiques. Breakdown by rank (render failure / turn limit / refusal): 32 = 7/3/0; 64 = 5/1/1; 128 = 4/5/1. Repeated identical source writes occurred in 26/27 episodes. Only three had any valid intermediate preview; two ended in refusal-tool misuse and one introduced a later code regression. No scores or evaluation outcomes were changed.

[Full inspection and preview images](eval-results/experiment3-hpsv3/invalid-review/report.md), [structured per-episode evidence](eval-results/experiment3-hpsv3/invalid-review/failures.json).

## Experiment 4 — Blinded human preferences versus HPSv3

- **Recorded:** 2026-11-04 (America/Toronto).
- **Status:** Complete. Rashik ranked all 163 original Experiment 1 prompts and sealed the review before model identities were revealed.
- **Question:** How well do the quantized HPSv3 preferences agree with the reviewer's preferences on the same saved images?
- **Models:** Original Experiment 1 rank-32, rank-64 and rank-128 checkpoints. This review does not evaluate the newly trained Experiment 3 checkpoints.
- **Review procedure:** Three anonymous final-image cards per prompt, with stable randomized ordering determined by reviewer, prompt case and a stored secret. Rank 1 is best; ties are allowed. Missing-image cards remain rankable. Decode the saved order before comparing preferences.
- **Scorer:** The completed Experiment 1 HPSv3 run, with 4-bit MLX language weights, bfloat16 vision weights and float32 reward head.
- **Validation:** Reconstructed preference scores match the UI's sealed score export. Source-file hashes are saved with the analysis.

### Human preference scores

A score is the mean pairwise result across the two alternatives, averaged equally over 163 prompts and multiplied by 100. A win earns 1, a tie .5 and a loss 0. Missing outcomes are included as ranked by the reviewer.

| Model | Human preference /100 | Pointwise 95% CI | Mean HPSv3 reward, same 150 prompts | HPSv3 Elo |
|---|---:|---|---:|---:|
| Rank 128 | 55.7 | [50.5, 60.9] | −1.370 | 1499.2 |
| Rank 64 | 47.7 | [42.3, 53.1] | −1.482 | 1506.8 |
| Rank 32 | 46.6 | [40.5, 52.8] | −1.604 | 1493.9 |

### Agreement on individual image pairs

Of 489 possible pair comparisons, 459 have both images. The reviewer tied 104 of those pairs and expressed a strict preference on 355. HPSv3 agrees on 226 of 355 decisive pairs, or **63.7%**, with pointwise prompt-bootstrap 95% CI **[58.1%, 69.3%]**. The remaining 129 decisive pairs disagree. Human ties are reported separately because a strict HPS reward ordering does not provide a calibrated tie threshold.

| Pair | Human first-model wins / losses / ties, all prompts | HPS agreement on decisive delivered pairs | Human adjusted p |
|---|---|---|---:|
| Rank 32 vs rank 64 | 63 / 65 / 35 | 71/118 (60.2%) | .930 |
| Rank 32 vs rank 128 | 54 / 74 / 35 | 77/121 (63.6%) | .278 |
| Rank 64 vs rank 128 | 54 / 71 / 38 | 78/116 (67.2%) | .304 |

Human preference tests are two-sided exact sign tests excluding human ties, with Holm correction across the three comparisons. All confidence intervals use 10,000 whole-prompt percentile bootstrap resamples, seed 42, keeping each prompt's pair results together.

The reviewer's preference scores and mean HPSv3 rewards have the same observed ordering, rank 128 > rank 64 > rank 32. HPSv3's pairwise-win Bradley–Terry fit instead places rank 64 first. This difference reflects how reward magnitudes and win frequencies are aggregated. Human preference and HPSv3 pairwise tests remain inconclusive; the mean-reward ordering is descriptive and has not been significance-tested. HPSv3 shows useful but incomplete agreement with this reviewer's decisive image preferences. This is one reviewer on a previously examined prompt set, and uncertainty covers prompt sampling only. It does not establish agreement across other people, training seeds or unquantized HPSv3.

Artifacts: [sealed UI score export](eval-results/experiment-1-human-preferences/4dec3b7eb9e26fdea0644c07f56d059afbbbe0b2eeebe51c6f7e0f375a1b4bc1.scores.json), [analysis summary and source hashes](eval-results/experiment4-human-hpsv3/summary.json), [per-prompt pair comparisons](eval-results/experiment4-human-hpsv3/comparisons.csv), and [reproducible analysis](eval-results/experiment4-human-hpsv3/analyze.py).

## Experiment 5 — Rank-128 SFT with revised trajectory generation and coding harness

- **Recorded:** 2026-11-04 (America/Toronto).
- **Status:** Planned; trajectory generation and training have not started for this experiment.
- **Decision:** Keep LoRA rank 128 for the next SFT run. This is Rashik's choice based on the previous experiments, including the observed human-preference and mean HPSv3 reward results; it does not establish statistically significant rank-128 superiority or revise earlier results.
- **Question:** Does a new rank-128 SFT run with revised teacher prompts and tooling improve drawing quality and final-image reliability?

### Planned changes

1. **Avoid unsolicited image text:** In the trajectory-generation prompts, instruct the teacher not to include text in images unless specifically required by the drawing request.
2. **Use a proper coding harness:** Replace the small set of custom drawing tools with a general coding harness for trajectory generation and the resulting SFT tool interface. Selected mini-SWE-agent 2.2.3 with a persistent Docker editing workspace; see the harness selection notes below.
3. **Remove the refusal tool:** Remove `refuse_drawing` from the tool interface used for the new trajectories and SFT run. Experiment 3's invalid-output inspection found two benign requests terminated through this tool with visual-quality self-critiques.

### Run details and evaluation

Teacher model, trajectory count, adapter initialization, training settings other than rank, and harness configuration remain to be recorded before launch. Save the final prompt and tool schemas, dataset hashes, and training configuration with the run artifacts.

Use the original 163 held-out evaluation prompts for comparison with the previous rank-128 checkpoints. The restored 228 SFT prompts remain training inputs and must not be counted as held-out evaluation. Report image quality, valid final-image rate, rendering failures, turn-limit outcomes, and token/turn usage, with the generation settings and tool interface recorded for each comparison. Because all three changes are introduced together, this experiment measures their combined effect and cannot isolate each change's contribution.

### Experiment 5 harness selection — mini-SWE-agent

Selected mini-SWE-agent 2.2.3 with a persistent Docker editing workspace and the
native `bash(command)` tool. Editing utilities include Python, sed, patch, diff,
and ripgrep. `artsy-preview ENTRYPOINT` optionally returns a PNG observation.
`artsy-submit ENTRYPOINT` snapshots freely named Python modules without requiring
a preview. Final rendering remains independent. CLI generation/drawing/evaluation select this
harness by default; historical drawing-tool callers retain their contract.

New SFT records use `mini-swe-bash-v1` and preserve cumulative editing actions,
PNG observations, source hashes, and independent final render results. All
executed editing actions receive supervision. No refusal tool is exposed; use
drawing-only prompts. The teacher is instructed to omit unrequested image text.
Two single-trajectory smoke tests have been collected with the configured teacher;
the full experiment dataset and training run have not been launched.

Single-trajectory smoke-test artifacts: `datasets/mini-swe-single-demo-2026-11-04/`.
The teacher was `openai/gpt-6-luna` through OpenRouter, using the SFT prompt
"a mountain and its reflection in a lake". The recovered trajectory contains
11 assistant turns and 12 total requests, including one provider interruption.
Final independent rendering and SFT conversion passed; human review is pending.
No successful preview occurred. The trace preserves the workspace-path and
standalone-preview interface errors encountered by the teacher.

Harness follow-up: installed `artsy-preview` and `artsy-submit` as real shell
executables. They support compound commands; preview accepts existing PNGs or
Python entrypoints outside `/workspace`. Native completion markers are recognized
inside scripts. Added `file` to the image. Interrupted collection now checkpoints
binary artifacts and Python files; `--retry-errors` restores them and retains the
request budget. Empty replies use normal missing-tool recovery. The focused
157-test suite passed with three skips, and two explicit Docker integration tests
passed, including interrupted collection, PNG restoration, final rendering and
SFT conversion. Default editing and rendering Docker images were rebuilt. These
checks used stubbed model responses and made no additional teacher requests.

A second approved single-trajectory smoke test used the SFT prompt "an owl with
its wings spread out swooping over a tree" with `openai/gpt-6-luna` through
OpenRouter. It completed in four model turns with no infrastructure or tool
errors: read API documentation, create `/tmp/draw.py`, preview that entrypoint,
and submit it inside a compound shell command. Independent final rendering and
SFT conversion passed; human review is pending. Artifacts are in
`datasets/mini-swe-second-demo-2026-11-04/`, including `trajectory.md`, `draw.py`,
and `sft-example.json`. Recorded usage: 15,342 prompt tokens and 4,861 completion
tokens. No training was launched.
