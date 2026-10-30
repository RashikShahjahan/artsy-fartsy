# Artsy training architecture

Training proceeds through **SFT → RL**. GRPO defaults to the latest completed SFT checkpoint, `checkpoints/sft-v2-qwen3.8-27b-final-preview-masked-cosine/final.json` (`Qwen/Qwen3.8-27B`), and derives its model and renderer from the checkpoint. Inspect remains the independent evaluation harness.

## SFT: learn the agent workflow

Supervise complete drawing trajectories, including source-writing tool calls, previews, repairs and submission. `training.sft` exports state and sampler paths in `checkpoints/sft-v1/final.json`.

## RL: improve adherence and safety

`training.grpo` initializes directly from the selected SFT state. Its frozen reference sampler is the same SFT checkpoint. It optimizes complete Mini SWE agent trajectories, including tool calls and optional visual repairs. Independent judges provide:

- **Safety:** content-image and code/action-history checks. A violation earns −1. Earlier unsafe commands still count after a refusal or a later harmless image.
- **Refusal:** a trusted prompt that should be refused earns +1 only for a submitted, protocol-valid refusal with no program. A refusal on a benign prompt earns 0.
- **Instruction adherence:** a VLM grades the final rendered image against the original request on a 0–4 rubric, normalized to 0–1. It does not grade aesthetics.

| Outcome | Terminal reward |
| --- | ---: |
| Unsafe code/actions or image | −1 |
| Correct refusal of a trusted harmful request | +1 |
| Benign over-refusal, incorrect refusal, render failure or missing submission | 0 |
| Safe rendered response to a benign request | adherence |

Budget-limit terminations cannot earn positive reward. There is currently **no separate aesthetic optimization stage or visual-quality RL reward**. Image safety reviews the final render, not every intermediate preview.

The Tinker Cookbook environment carries the full tool trajectory and trusted terminal artifacts. `compute_group_rewards` applies safety/refusal outcomes and adherence, returning terminal reward adjustments. Cookbook subtracts the group mean without dividing by standard deviation. There is no per-component z-score normalization, efficiency reward or intermediate-render reward. Optional KL regularization uses the initial SFT sampler.

The policy and safety/adherence judges use separate clients. Judge failures stop scoring instead of producing fabricated rewards. The SFT-initialized run uses a new algorithm version; older RL runs cannot be continued under it. Compatible runs can resume or continue with the original SFT KL reference.

## Independent validation

Keep held-out evaluation prompts out of SFT and RL training. Track human visual preference against SFT, adherence, safety failures, benign over-refusal, refusal success and render validity. Tool/turn metrics and RL reward are diagnostics; checkpoint selection still needs independent visual evaluation because RL does not explicitly optimize aesthetics.

Before large runs, validate SFT checkpoint compatibility, refusal behavior and reward calculation offline, then run small live Docker/RL smoke tests.
