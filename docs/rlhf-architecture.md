# Artsy training architecture

Artsy trains drawing agents with **SFT → RLVR → RLHF**, while Inspect remains
the independent evaluation harness. Each RL stage uses Tinker Cookbook PPO
with grouped rollouts. The original frozen SFT sampler is kept for optional
reference-policy KL regularization across both RL stages.

## SFT: learn the mini-SWE drawing workflow

The current SFT dataset records actual mini-SWE-agent 2.2.3 trajectories:
a system prompt, a drawing request, `bash(command)` edits in a persistent
workspace, optional PNG previews via `artsy-preview`, and explicit submission
via `artsy-submit ENTRYPOINT` or native completion. Multi-file workspaces are
serialized as validated Python bundles before final rendering. The final
checkpoint includes `state_path`, `sampler_path`, and, on newly produced SFT
runs, a `config.tool_contract` identifying `mini-swe` or `drawing-tools`.

## RLVR: learn verifiable tool use and execution

`training.rlvr` (or `make rlvr`) starts from a compatible mini-SWE SFT
`final.json`. It shares the mini-SWE 12-turn bash harness, network-disabled
Docker workspaces, submission snapshots, and a fresh independent Docker final
renderer with RLHF. It does not initialize or call an HPSv3 client.

The canonical `datasets/parti-prompts/rl/prompts.jsonl` pool contains 327
prompts and is split deterministically into **227 RLVR** and **100 RLHF**
examples, stratified by Category/Challenge. Neither stage sees the other's
prompts. RLVR defaults to **2 independent rollouts per prompt** (454 per epoch),
while RLHF defaults to **4** (400 per epoch); `--group-size` overrides either. The subset is chosen before the `--limit` cap, pinned in run metadata,
and revalidated on continuation. Alternative custom dataset paths are not
automatically partitioned. The prompt preparation script can also materialize
`rl/rlvr.prompts.jsonl` and `rl/rlhf.prompts.jsonl` for inspection.

The terminal RLVR reward is **penalty-only**. A clean submission receives
**0**. An absent or invalid final 1920×1200 PNG costs **−1.0**. Tool protocol
errors and execution errors each cost **−0.025** per error, with a separate
**−0.10** cap for each category. An episode with no tool calls counts as one
protocol error. Every verified submission scores above every incomplete
episode, and successful commands leave earlier penalties intact. The rollout's
`result.json` records raw counts, diagnostic success ratios, individual
penalties, and the total reward.

The final RLVR checkpoint has algorithm `tinker-cookbook-rlvr-v1` and carries
its own policy state and sampler as well as the original SFT KL reference.
RLVR cannot take another RLVR/RLHF checkpoint as initialization.

## RLHF: improve final-image preference

`training.rlhf` (or `make rlhf`) defaults to the completed
`checkpoints/rlvr-v1/final.json` checkpoint. The historical `training.grpo`
entrypoint retains direct SFT initialization for older experiments.
Both default to `--harness mini-swe`. Cookbook still controls
sampling, PPO updates, KL, grouped advantages, and checkpoints. Its model-facing
tool is now the same `bash(command)` schema used for SFT and evaluation, with
the same mini system prompt and a maximum of 12 model turns per episode.

Each independent rollout has its own network-disabled Docker workspace.
The agent can create and modify Python modules and run ordinary shell commands.
`artsy-preview` returns execution feedback plus a native model-visible PNG;
`artsy-submit` freezes a validated snapshot of the submitted Python modules.
A successful explicit submission terminates the episode. Without submission,
no workspace code is eligible for positive reward.

Final scoring runs that snapshot as a bundled Python source in a **fresh**,
network-isolated Docker container. Only the independently rendered valid
1920×1200 PNG reaches HPSv3. The configured scorer returns a raw score and a
normalized `preference` reward using
`sigmoid((hpsv3_score - ARTSY_HPSV3_CENTER) / ARTSY_HPSV3_SCALE)`.
Invalid renders, missing submissions, refusals and budget-limit terminations
earn zero. Source, final PNG, judge response, reward and tool-call history are
saved per rollout. Group rewards are mean-centered by Cookbook, without
standard-deviation scaling.

The historical `--harness drawing-tools` option retains the older named tools
(`update_drawing`, `read_drawing`, `preview_drawing`, `submit_drawing`,
`refuse_drawing`) for matching SFT checkpoints. A mismatched *declared*
checkpoint tool contract is rejected. Checkpoints created before this
metadata was added must be checked manually against their training dataset.

The preference algorithm is `tinker-cookbook-rlhf-v8`: mini-SWE rollouts record
trajectory contract v4; legacy drawing-tool rollouts retain v3. Older GRPO
runs are not resumable or continuable across this implementation change,
because optimizer and trajectory provenance must remain pinned.

## Independent validation

Keep held-out evaluation prompts separate from SFT, RLVR and RLHF prompts.
Compare checkpoints with Inspect using the **matching** harness, prompt set,
seed, and budgets; track human visual preference, adherence, render validity,
safety, and refusals. Run a one-update Docker/Tinker pilot before a longer
GRPO run to validate the model's bash calls, preview images, submission
snapshots, and terminal HPSv3 reward.
