# Artsy training architecture

The drawing policy is Tinker-hosted `Qwen/Qwen3.5-4B`. Training proceeds through **SFT → visual preference DPO → RL**. Inspect remains the independent evaluation harness.

## SFT: learn the agent workflow

Supervise complete drawing trajectories, including source-writing tool calls, previews, repairs and submission. `training.sft` exports state and sampler paths in `checkpoints/sft-v1/final.json`.

## DPO: learn visual quality preferences

`training.dpo` starts from the SFT state and optimizes preferred versus rejected drawing trajectories for the same request and preview seed. Preferences concern composition, coherence, color, balance, craftsmanship and expressive appeal. Both candidates must be approved, safe, render-valid drawings; do not use adherence failures or unsafe outputs as aesthetic negatives.

Image comparisons alone are insufficient: each label must identify both recorded policy trajectories. The trainer validates image hashes against those records, retains the full multimodal tool history and applies loss only to assistant actions. It sums assistant-token log probabilities for each trajectory and uses the standard DPO loss against a **frozen SFT sampler**, including after resume. Ties and validation/test pairs do not receive training updates. Keep prompt families in one split; exact prompts, trajectories, sources and images cannot cross splits.

DPO exports the same state/sampler `final.json` format as SFT. The default destination is `checkpoints/dpo-v1/final.json`, which becomes the RL initialization and KL reference. See [data format and launch commands](preference-rlhf.md).

The existing `preference.train` and `preference.evaluate` tools may still train and calibrate a separate aesthetic comparison judge. That judge is optional for offline preference labeling and evaluation; it is not part of RL reward computation.

## RL: improve adherence and safety

`training.grpo` requires the DPO checkpoint and optimizes full Mini SWE agent trajectories, including tool calls and optional visual repairs. Independent judges provide:

- **Safety:** content-image and code/action-history checks. A violation earns −1. Earlier unsafe commands still count after a refusal or a later harmless image.
- **Refusal:** a trusted prompt that should be refused earns +1 only for a submitted, protocol-valid refusal with no program. A refusal on a benign prompt earns 0.
- **Instruction adherence:** a VLM grades the final rendered image against the original request on a 0–4 rubric, normalized to 0–1. It does not grade aesthetics.

| Outcome | Terminal reward |
| --- | ---: |
| Unsafe code/actions or image | −1 |
| Correct refusal of a trusted harmful request | +1 |
| Benign over-refusal, incorrect refusal, render failure or missing submission | 0 |
| Safe rendered response to a benign request | adherence |

Budget-limit terminations cannot earn positive reward. Visual quality contributes no RL score, metric or reward, and RL requires no visual judge checkpoint. Image safety reviews the final render, not every intermediate preview.

The Tinker Cookbook environment carries the full tool trajectory and trusted terminal artifacts. `compute_group_rewards` applies safety/refusal outcomes and adherence, returning terminal reward adjustments. Cookbook subtracts the group mean without dividing by standard deviation. There is no per-component z-score normalization, efficiency reward or intermediate-render reward. Optional KL regularization uses the initial DPO sampler.

The policy and safety/adherence judges use separate clients. Judge failures stop scoring instead of producing fabricated rewards. The new reward/algorithm version rejects continuation of legacy RL runs that included visual rewards; start a fresh RL run from DPO. Compatible runs can resume or continue with the original DPO KL reference.

## Independent validation

Keep held-out prompts and human labels out of SFT, DPO and judge training. Track human visual preference against SFT and DPO, adherence, safety failures, benign over-refusal, refusal success and render validity. Tool/turn metrics and RL reward are diagnostics; checkpoint selection also needs independent visual evaluation because RL no longer optimizes aesthetics directly.

Before large runs, validate pair-to-trajectory bindings, the frozen reference, refusal behavior and reward calculation offline, then run small live DPO and Docker/RL smoke tests.
