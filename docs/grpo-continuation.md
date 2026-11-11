# Continuing RLHF and selecting a checkpoint

A continuation extends a completed, compatible adherence/safety RL run in a new
output directory while retaining its optimizer state and original SFT KL reference.
The paths below are examples for future runs, not bundled checkpoints.

```sh
# Validate without writing files or making Docker/model requests:
.venv/bin/python -m training.grpo \
  --continue-from checkpoints/grpo-v4 \
  --additional-steps 25 \
  --output-dir checkpoints/grpo-v4-continued \
  --dry-run
```

Remove `--dry-run` to train. Repeat the exact command after interruption; the new
directory's latest committed checkpoint takes precedence over its inherited
starting state. Cookbook restores optimizer state through a seeded resume record.
The source directory is read only.

The manifest records source paths and hashes, the starting step, and the extra
update budget. `total_steps` is cumulative; `episodes` counts only the new block.
Prompt order follows deterministic, epoch-seeded batches and continues at the
next global batch index. This preserves ordering, not bit-for-bit hosted sampling.

Model, dataset, prompt selection, reward, judges, learning rate, KL coefficient,
generation limits, and concurrency are inherited. Judge credentials still come
from the environment. Source/dependency or dataset drift is rejected. Pass
`--dataset` explicitly if the original run used a nondefault file. CLI overrides
of inherited training parameters are rejected; output and save cadence may change.

Completed blocks produce `final.json` and appear in the model catalog as
`artsy/<output-directory-name>`. Continuation does not automatically evaluate or
early-stop a block.

## Selecting a checkpoint

Evaluate candidates with Inspect using identical held-out prompts, seeds, judge
versions, and token/tool budgets. Keep evaluation prompts out of SFT and RL training. Track human preference against SFT, adherence, render validity,
harmful output, code violations, benign over-refusal, correct refusals, and cost.

Choose update budgets and stopping thresholds before a run. Use paired per-prompt
comparisons and uncertainty estimates to assess small changes. Training reward,
KL, entropy, and within-group reward variation are diagnostics; select checkpoints
using independent quality and safety measurements. See the [reward architecture](rlhf-architecture.md).

Legacy RL runs initialized from another policy stage are incompatible with the SFT-initialized algorithm. Start a fresh RL run from the SFT checkpoint.

## Recovering the interrupted hosted-context run

The interrupted RLHF run used a 262,144-token rollout budget while its hosted
policy sampler selected the 65,536-token default. The explicit hosted-context
adapter selects `Qwen/Qwen3.5-4B:peft:262144` for training restoration and policy
sampling. Recover checkpoint 2 and its optimizer into a new output directory:

```sh
.venv/bin/python -m training.grpo \
  --recover-from checkpoints/rlhf-rlvr-g4-epoch1-b32-g4-c128 \
  --output-dir checkpoints/rlhf-rlvr-g4-epoch1-b32-g4-c128-context262k
```

Use `training.grpo` directly: `training.rlhf` injects stage flags that recovery
rejects. Recovery inherits the source settings and original four-update total;
two updates remain, without adding updates or resetting the optimizer. Add
`--dry-run` to validate the plan without creating the output directory.

This migration accepts only the exact audited old driver hash and the new
hosted-context adapter; every other source and dataset hash must match. It
records immutable source manifest and checkpoint-history hashes, preserves the
original run, and seeds the new history with the saved checkpoint 2 paths.
