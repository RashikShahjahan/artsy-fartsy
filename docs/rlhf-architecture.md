# Artsy RLHF architecture

The Artsy policy is the existing Tinker-hosted `Qwen/Qwen3.5-4B` SFT checkpoint. RL optimizes full Mini SWE agent trajectories, including tool calls and optional visual repairs. Inspect remains the independent evaluation harness.

## Reward components

For every prompt, sample a group of independent agent rollouts (default four). Keep each rollout's final source, final trusted render, refusal artifact, full bash command history, and metadata.

- **Safety:** independent content-image and code/action-history checks. A violation earns -1. Never allow good visuals to compensate for unsafe behavior, and never ignore earlier unsafe commands after a refusal or a later harmless image.
- **Instruction adherence:** a rubric judge scores the final image from 0 through 4; divide by 4. This judge sees the original prompt and the trusted final PNG.
- **Visual aesthetics:** a *separate, frozen* Qwen3.5-9B VLM compares safe, render-valid images for the *same prompt*. The judge predicts A, B, or Tie, without grading prompt adherence. For each rollout the visual reward is `(wins + 0.5 * ties) / (group_size - 1)`. Compare both image orderings to check position bias; render-invalid candidates lose automatically.

The composite reward is:

| Outcome | Reward |
| --- | --- |
| Safety violation (image, code, or action history) | -1 |
| Correct refusal of a disallowed request | +1 |
| Unnecessary refusal of a benign request | 0 |
| Invalid or missing trusted render on a benign request | 0 |
| Safe rendered response to a benign request | 0.5 × visual + 0.5 × adherence |

The prompt kind must come from trusted dataset metadata, never from the policy's response. Safety-judge failures must not silently become safe verdicts.

## Preference model

Provide human-labeled comparison records with two PNG paths and preference `A`, `B`, or `Tie`. Train a Qwen3.5-9B visual comparison judge on Tinker with supervised assistant-label prediction and LoRA. Save an independent judge sampler checkpoint; never update it in the policy optimizer. Public aesthetic-preference pairs may supplement domain-specific Artsy labels, but the held-out human preference benchmark remains separate.

## Policy training

The Tinker Cookbook RL environment carries the full tool trajectory and trusted terminal artifacts. Its `compute_group_rewards` compares final images across rollouts, merges the safety/adherence/visual signals and returns terminal reward adjustments. Cookbook computes advantages by subtracting the *group mean*; it does not divide by the group's standard deviation. Therefore there is no additional per-component z-score normalization or reward-variance scaling. The policy optimizer may optionally use KL against the frozen SFT reference checkpoint.

The visual judge should not see generated source code or hidden rubric output. The safety and adherence judges should not see the visual ranking. Keep the policy and all judges on separate model clients/checkpoints.

## Validation

Keep testing separate from policy training, using the existing Inspect evaluation pipeline and a held-out prompt family where practical. Track:

- human preference win rate against SFT and judge agreement with held-out human comparisons;
- reversal consistency and per-style aesthetic preference accuracy;
- adherence distribution, visual win fractions and composite reward;
- harmful-compliance rate, code-safety violations, benign over-refusals, successful safe refusals;
- render validity and tool/turn metrics as diagnostics only (no efficiency reward).

Do not run large-scale RL until the native two-image Tinker bridge, independent judges, refusal path, and group reward calculation all pass offline tests and a small live smoke test.
