# Continuing RLHF and selecting a checkpoint

A continuation extends a completed, compatible multi-signal RLHF run in a new
output directory while retaining its optimizer state and original SFT KL reference.
The paths below are examples for future runs, not bundled checkpoints.

```sh
# Validate without writing files or making Docker/model requests:
.venv/bin/python -m training.grpo \
  --continue-from checkpoints/grpo-rlhf-v1 \
  --additional-steps 25 \
  --output-dir checkpoints/grpo-rlhf-v1-continued \
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
versions, and token/tool budgets. Keep evaluation prompts out of policy and visual
judge training. Track human preference against SFT, adherence, render validity,
harmful output, code violations, benign over-refusal, correct refusals, and cost.

Choose update budgets and stopping thresholds before a run. Use paired per-prompt
comparisons and uncertainty estimates to assess small changes. Training reward,
KL, entropy, and within-group reward variation are diagnostics; select checkpoints
using independent quality and safety measurements. See the [RLHF setup](preference-rlhf.md)
and [reward architecture](rlhf-architecture.md).
