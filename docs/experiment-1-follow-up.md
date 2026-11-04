# Experiment 1 follow-up: which LoRA produces the best images?
 
Quality determines the winner; tokens break a demonstrated practical tie. No human judges assumed. [Experiment 1](../EXPERIMENT_LOG.md) is an exploratory pilot.

## What two more runs buy

On the 355 pilot prompts where every LoRA delivered an image, Luna's mean visual-quality differences were small. The table models three independent batches of 355 matched prompts each, fixed checkpoints, the same prompt mix and rubric, and a true difference equal to the observed one. It is a sensitivity calculation, not a forecast. It excludes missing submissions, and models 1,065 prompts for illustration; the two fresh batches alone would contribute about 710 matched prompts. Do not count the pilot as independent confirmation of an endpoint selected after seeing it.

| Comparison | Pilot gap /4 | Paired SD | Illustrative power at 1,065, single contrast | With three pairwise comparisons |
|---|---:|---:|---:|---:|
| 64 − 32 | .028 | .837 | 20% | 10% |
| 64 − 128 | .079 | .840 | 87% | 75% |
| 32 − 128 | .051 | .808 | 54% | 37% |

Normal approximation for a two-sided paired mean test: standard error = paired SD / sqrt(n); final column uses Bonferroni alpha = .05/3. Each cell is per-contrast power, not the probability of identifying a unique winner. At 1,065 independent prompts, an effect of roughly .08/4 would be detectable with 80% power per comparison after correction. Fresh prompt families are needed for this interpretation. A different rubric requires a new variance estimate.

- Replaying identical prompts at temperature zero with the same checkpoints and settings may reproduce the same drawings. Rejudging those images measures judge variation, not additional model performance.
- Fresh sampled outputs on the same prompts measure within-prompt variability. If deployment uses stochastic decoding, vary its seeds and average repeats within prompt; do not treat triples as independent prompts. Changing temperature just for this test changes the question.
- If the claim concerns training rank rather than these checkpoints, two more *training* runs per rank, using matched independent seeds across ranks, are useful for checking reproducibility. Analyze variation across seeds and prompts; three training seeds per rank do not guarantee a precise training-level ranking.

## Quality measurement and decision

Use the 391 saved prompts to pilot judging before generating more drawings. For each of the three LoRA pairs ask a blinded automated vision judge: “Which drawing would you choose as the final illustration for this request?” Ignore token use and model identity. Assess requested content and style, subject recognizability, composition and visible defects. Use anchored options: A clearly better / A slightly better / indistinguishable / B slightly better / B clearly better. Mark a clear win for a consequential omission or unmistakable visual defect, a slight win for a small but identifiable improvement, and a tie when neither has a defensible advantage. Ask for the visible difference. Do not assume clear and slight wins have a known numeric distance.

Audit on stratified old pairs, including pairs tied and untied on the old score. Reverse presentation order; include identical-image controls. If available, use a second model family and report each judge's results. Examine repeatability of newly distinguished pairs and held-out control performance with rank labels hidden. A finer scale is useful only if the judge can use its extra distinctions consistently. Freeze the rubric before seeing rank-level pilot results. Machine agreement supports a judge-defined ranking, not human preference.

For a fresh confirmatory set score win/tie/loss as 1/.5/0 for the primary preference rate, with strength descriptive. A delivered drawing beats a missing one; two missing drawings tie. Also report image-only results on pairs where both delivered. Average over order swaps, judges and repeated generations within prompt; estimate paired uncertainty by resampling prompts (families if clustered). Use simultaneous intervals across the three pairings. Declare a unique winner only if it beats both alternatives. Define a practical tie band around 50% before judging: use tokens only among quality leaders shown equivalent within that band. A wide interval or cyclic preferences mean no unique winner yet.
