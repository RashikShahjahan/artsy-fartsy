# Artsy training architecture

Training proceeds through **SFT → RL**. GRPO defaults to the latest completed SFT checkpoint, `checkpoints/sft-v2-qwen3.8-27b-final-preview-masked-cosine/final.json` (`Qwen/Qwen3.8-27B`), and derives its model and renderer from the checkpoint. Inspect remains the independent evaluation harness.

## SFT: learn the agent workflow

Supervise complete drawing trajectories, including source-writing tool calls, previews, repairs and submission. `training.sft` exports state and sampler paths in the selected output directory's `final.json`. Current SFT and RL prompt assignments come from `datasets/parti-prompts/`; keep its 391 evaluation prompts out of training.

## RL: improve adherence and visual quality

`training.grpo` initializes from the selected SFT state and keeps that SFT sampler as its frozen reference. It optimizes complete drawing-tool trajectories, including optional visual repairs. RL uses the shared `update_drawing`, `read_drawing`, `preview_drawing`, `submit_drawing`, and `refuse_drawing` schemas and drawing prompt from `drawing/agent_contract.py`. Tools run sequentially; successful submission or refusal ends the episode and prevents later tool mutations. Model arguments enter fixed Docker commands as stdin, without a model-facing shell.

One vision judge grades only the trusted final render using the absolute rubric shared with the latest paired evals in `drawing/visual_judge.py`:

- **Adherence (0–4):** from no meaningful requested content to all material constraints fulfilled.
- **Visual quality (0–4):** from unusable/blank to exceptionally polished and visually effective, judged within the requested style. Simple drawings can score highly.

The judge returns both integer scores and concrete comments in one response. RL uses absolute judgments per output, rather than model comparisons or position-dependent preferences. Terminal reward is `(adherence_score + visual_quality_score) / 8`. Artifacts retain the raw judgment, normalized `adherence` and `visual_quality` components, and the final reward.

Missing submissions, invalid renders, refusals, and budget-limit terminations earn zero. The previous code-safety, image-safety, and refusal rewards are removed from RL. Prefer drawing-only datasets: refusal examples from older datasets cannot earn positive reward under this objective. Independent evaluation still supports its safety and refusal metrics.

The final source is rerendered in a fresh Docker container without network access. Agent previews cannot substitute for the final image. Only a successful explicit submission reaches rendering and judging. Configure the single judge through `ARTSY_CORRECTNESS_JUDGE_*`; the policy and judge use separate clients. Invalid, truncated, or failed judge responses stop scoring instead of creating zero rewards.

`compute_group_rewards` combines the normalized components and adjusts terminal rewards. Cookbook subtracts the group mean without dividing by standard deviation. There is no efficiency or intermediate-render reward. Optional KL regularization uses the initial SFT sampler. The algorithm version is `tinker-cookbook-rlhf-v6` and rollout artifacts record drawing-tool trajectory contract v3; older RL runs cannot resume or continue under this setup.

## Independent validation

Keep held-out eval prompts out of SFT and RL training. Track independent human visual preference, adherence, visual quality, render validity, safety, and refusal behavior against the SFT checkpoint. Reward gains still need held-out evaluation to detect judge overfitting.

Before large runs, validate checkpoint compatibility and reward calculation offline, then run small live Docker/RL smoke tests.
