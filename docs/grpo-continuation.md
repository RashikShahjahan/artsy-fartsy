# Continuing GRPO and selecting a checkpoint

## Continue in bounded blocks

The first `grpo-qwen38-v1` run completed 100 updates over 200 training
prompts, with two prompt groups and four trajectories per group. Its first
continuation completed step 125. A continuation adds updates to a completed run
in a new output directory; this example also applies the expanded eval limits:

```sh
# Validate locally; no files are written and no Docker/model requests are made.
.venv/bin/python -m training.grpo \
  --continue-from checkpoints/grpo-qwen38-v1-step125 \
  --additional-steps 25 --match-eval-limits \
  --output-dir checkpoints/grpo-qwen38-v1-step150 \
  --dry-run

# Run the same command without --dry-run to train through cumulative step 150.
.venv/bin/python -m training.grpo \
  --continue-from checkpoints/grpo-qwen38-v1-step125 \
  --additional-steps 25 --match-eval-limits \
  --output-dir checkpoints/grpo-qwen38-v1-step150
```

Repeat the training command unchanged after interruption. The new directory's
latest committed checkpoint takes precedence over the inherited starting state.
After completing and evaluating this block, another block can use
`--continue-from checkpoints/grpo-qwen38-v1-step150`, `--additional-steps 25`,
and a new `--output-dir checkpoints/grpo-qwen38-v1-step175`. A run already using
the expanded limits inherits them without needing another migration flag.

Continuation restores the full optimizer checkpoint through Cookbook's resume
path. Cookbook's `load_checkpoint_path` alone starts a fresh optimizer, so the
wrapper seeds a resume record in the new directory. The original directory is
read only. The manifest records source paths and hashes, the starting step and
the additional update budget. The reported `total_steps` is cumulative;
`episodes` counts the additional block only.

Model, judge, reward, learning rate, KL coefficient, generation limits, prompt
selection and concurrency are inherited from the saved configuration. The KL
reference remains the original SFT sampler. Judge credentials still come from
the environment. A changed training dataset or runtime is rejected. Pass
`--dataset` explicitly when the source used a nondefault dataset path.

### Match the expanded eval limits

New training runs use the same token limits as the expanded evals: **262,144
tokens of context per request**, **8,192 generated tokens per response**, and
**no separate aggregate token budget**. History and preview image tokens count
toward the context window. The next response is clipped to the remaining space;
the rollout no longer stops early merely because a full response would not fit.
The sixteen-turn limit and Docker execution limits remain aligned with evals.

Existing runs keep their recorded limits unless explicitly migrated. To apply
the expanded limits after the completed step-125 run:

```sh
.venv/bin/python -m training.grpo \
  --continue-from checkpoints/grpo-qwen38-v1-step125 \
  --additional-steps 25 --match-eval-limits \
  --output-dir checkpoints/grpo-qwen38-v1-step150 \
  --dry-run
```

Remove `--dry-run` to train. Repeat the exact command after interruption. The
migration records the old settings and source hashes alongside the new rollout
policy, retaining the optimizer state and original SFT reference. It accepts
only the recorded limit-related source changes; dataset, reward and unrelated
runtime changes still fail validation. Completed historical runs are preserved.
Evaluate competing checkpoints under identical expanded eval settings before
attributing any score change to the additional training.

The wrapper extends deterministic, epoch-seeded prompt ordering and starts at
the next global batch index. Step 100 begins the next shuffled epoch for this
200-prompt dataset; continuing from a partial epoch consumes its remaining
batches before starting another. This preserves prompt order, not bit-for-bit
reproduction of hosted sampling or optimizer execution.

Each completed block produces a `final.json` and is discoverable by the model
catalog as `artsy/<output-directory-name>`. Evaluate it with the existing
evaluation CLI using the same prompt file, judge settings and generation limits
as the preceding checkpoint. Continuation does not automatically evaluate or
stop early within a block.

## Proposed stopping policy for this pilot

These are initial experiment rules, not calibrated universal thresholds.
Choose them before the next block and record decisions alongside the evaluation
artifacts.

1. **Use development quality to select checkpoints.** Treat the already inspected
   `evaluation/generated-110.jsonl` as development data from now on. Its 90
   drawing prompts and 20 refusal prompts must stay out of training. Reserve a
   separate, unseen test set for the final chosen checkpoint. Freeze the
   development prompt text, seeds, harness, judge model/version and sampling
   budgets across comparisons. Keep the evaluation correctness judge independent
   of the Qwen3.8 training judge, and review a fixed sample of PNGs manually.
2. **Evaluate every 25 additional updates.** Start with steps 100, 125, 150 and
   175. Use `by_kind.drawing.adherence_mean` as the primary metric, counting
   invalid drawings and unnecessary refusals as zero. Check drawing render
   rate, scores by difficulty, refusal behavior, repair success and token use
   alongside it. Require complete evaluation coverage and resolve infrastructure
   errors before comparing checkpoints. The step-100 development baseline is
   adherence **0.568**, valid renders **76/90**, and correct refusals **20/20**.
3. **Keep the best eligible checkpoint.** Use a provisional meaningful improvement
   of **0.02 absolute adherence** over the incumbent. Compare per-prompt score
   differences on matching prompts, with a paired bootstrap 95% interval rather
   than treating each checkpoint's mean as independent. When results are close,
   rerun both candidates with matched additional sampling seeds or expand the
   development set; do not declare a tiny change a win. Repeat stochastic
   comparisons at the prompt level, not by treating multiple draws from one
   prompt as independent tasks. With only 90 drawing prompts, small gains may
   remain unresolved.
4. **Stop allocating updates after three evaluations without a meaningful gain.**
   The patience counter resets only for a supported improvement of at least
   0.02 over the incumbent, subject to the checks below. If three checkpoints
   are inconclusive because evaluation is noisy, pause training and improve
   measurement. Keep the incumbent, which may precede the latest checkpoint.
   Cap this first extension at **100 additional updates** before reviewing the
   curriculum and cost, even if patience has not expired.
5. **Pause on regressions.** Investigate a drawing render-rate drop greater than
   five percentage points against the incumbent, any newly failed refusal case,
   a harmful-image or malicious-code judgment, or a sharp increase in incomplete
   episodes. Review the artifacts and repeat uncertain judgments before deciding
   whether to roll back. Twenty successful refusal cases provide a small smoke
   test, not evidence of general safety. A checkpoint with a confirmed regression
   does not replace the incumbent merely because its mean adherence is higher.

Training reward, KL, entropy, within-group reward variation, termination counts
and token cost are diagnostics. They help explain plateaus, unstable updates and
expensive behavior; they do not replace development evaluation. Rising training
reward with flat or falling independent quality is a reason to stop and inspect
the reward/curriculum. Low reward variation can also mean that the current
prompts no longer distinguish candidates. There is no calibrated KL cutoff for
this run.

Optimizing an imperfect reward proxy can eventually lower independent quality;
see [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).
Cookbook's checkpoint and metrics files are described in
[RL Training Outputs](https://tinker-docs.thinkingmachines.ai/cookbook/rl/rl-logging/).
