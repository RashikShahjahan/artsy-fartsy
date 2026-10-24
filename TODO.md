# Artsy: drawing-model training environment

Build an environment that trains a small language model to turn drawing instructions into editable, reproducible Python code, using supervised fine-tuning (SFT) for a warm-up and reinforcement learning (RL) for improvement.

The starting point is the compact Cairo-backed `ArtCanvas` interface described in [Can an AI Learn to Draw?](https://blog.rashik.sh/tech/can-ai-learn-to-draw/). The central artifact is the program together with its rendered image.

The choices and numerical targets below are proposed starting points to validate in a pilot.

## 1. Define the first experiment

- [x] Select Qwen/Qwen3.5-4B (Apache 2.0) with its vision encoder and native tool-call format. Serve a bounded 32,768-token context; benchmark Python performance on held-out drawing tasks before training.
- [ ] Record available GPU memory, training budget, and teacher/judge API budget. Benchmark a small batch before choosing dataset size or training duration.
- [x] Use Qwen/Qwen3.5-4B as a vision-language policy. It receives API documentation, the user prompt, its code, execution feedback, and rendered preview images. Keep independent evaluator scores outside policy observations.
- [ ] Fix the initial scope: static 2D drawings, one Python file, a versioned `ArtCanvas` API, a fixed canvas size, and PNG output. Include shapes, diagrams, scenes, and procedural patterns.
- [ ] Define what counts as a successful response: submitted code reproduces a safe image that satisfies the user's visible requirements. Define a structured refusal response for disallowed requests.
- [ ] Create versioned configuration for the model, API, renderer, sandbox, data generation, SFT, RL, reward rules, and evaluation.

**Done when:** a written experiment configuration specifies the model, compute limits, output contract, and evaluation criteria.

## 2. Build the drawing runtime and mini coding environment

- [x] Implement or recover `ArtCanvas`: backgrounds, colors, paths, shapes, curves, gradients, transforms, and saving. Write concise API documentation and working examples.
- [ ] Pin Python, Cairo, fonts, and other rendering dependencies; inject an explicit random seed so runs can be reproduced within the pinned runtime.
- [x] Implement episode operations: `reset(prompt, seed)`, `write_code(source)`, `run_code()`, `submit()`, and `refuse(reason)`. Keep the training adapter separate from the environment core.
- [x] Return bounded execution feedback: syntax/runtime errors, exit status, elapsed time, output dimensions, and artifact identifiers. Keep hidden grading criteria and evaluator answers outside policy observations.
- [ ] Run generated code in an isolated worker with no network, credentials, host mounts, or access to the grader. Use a read-only runtime, an ephemeral writable directory, and CPU, memory, process, file-size, and wall-clock limits. Import/AST checks are additional validation, not the isolation boundary.
- [ ] Restrict the drawing program to the documented API and approved utilities. Prevent external image downloads, arbitrary file access, and loading pre-rendered images as a substitute for drawing.
- [x] Give each episode a bounded token and tool-call budget; start with at most three render attempts. Define explicit outcomes for submission, refusal, invalid calls, timeouts, and budget exhaustion.
- [x] Re-run submitted code in a fresh worker with the same seed. Score that render so stale previews or modified artifacts cannot earn credit.
- [x] Store the full observable trajectory, source, images, errors, seed, and runtime version. Ensure reset removes all state between episodes.
- [ ] Verify isolation, cleanup, deterministic examples, infinite-loop termination, oversized output handling, and recovery from worker crashes.

**Current scope:** Mini SWE drawing/evaluation runs in per-sample Docker containers
with no outbound network or host mounts. Dataset generation and the standalone
renderer continue to execute locally; API/file-access restrictions remain deferred.
Python packages are pinned; host Cairo/fonts remain
unpinned. Local tests cover cleanup, reproducibility, timeouts, oversized PNGs,
and recovery after execution/startup failures. The independent episode core is
available for future data/training adapters. Mini SWE replaces Pi's model loop;
Inspect logs retain its transcript, without Pi SQLite session resume.

**Done when:** the same episode runner supports teacher data generation, model rollouts, and evaluation, and reliably contains failing programs.

## 3. Generate prompt–code pairs and repair trajectories

- [x] Build a prompt generator from structured specifications: objects, counts, colors, positions, spatial relationships, style, background, and explicit exclusions.
- [x] Create a curriculum from simple primitives to multi-object scenes and procedural art. Include varied wording and combinations, not just many paraphrases of one template.
- [x] Have a stronger teacher generate several candidate programs per prompt using the exact API documentation and runtime constraints.
- [x] Execute candidates in the local environment (sandbox deferred per current scope); use bounded repair attempts for syntax, API, and rendering failures. Keep successful edit–run–repair trajectories for tool-use SFT.
- [ ] Check the rendered image against the prompt with deterministic checks where reliable and a vision evaluator for semantic requirements. Do not accept a pair merely because its code runs.
- [ ] Apply the content policy to prompts and every rendered candidate. Exclude harmful outputs from positive SFT targets; include correct refusal examples and benign lookalikes to reduce over-refusal.
- [x] Save a dataset record containing `id`, `prompt`, code or refusal, trajectory, artifact references, seed, API/runtime versions, teacher provenance, evaluator versions/holistic scores, safety labels, and split.
- [ ] Deduplicate prompts, related templates, code, and near-identical renders. Split by task/template family before generation so repairs and paraphrases cannot leak into validation or test sets.
- [ ] Audit a stratified human-reviewed sample for correctness, diversity, safety, and evaluator agreement. Version accepted data and rejection reasons.
- [ ] Produce a pilot of roughly 1,000 accepted examples, measure acceptance rate and cost per example, then choose the scale-up target from learning curves and budget.

**Done when:** a reproducible command produces a versioned, audited dataset of executable prompt–code pairs, repair trajectories, and refusal examples.

**Current implementation:** `.venv/bin/python -m drawing.generate_dataset` generates seeded prompt plans, runs a
configurable teacher through bounded repairs, and checkpoints candidates for resume.
An optional independent vision evaluator gates acceptance; without it, valid renders
remain pending review. Task families are split before generation and exact source/pixel
duplicates are rejected. Near-duplicate detection, calibrated semantic/safety checks,
human audits, and a live teacher pilot remain outstanding. See README for usage.

## 4. Implement and calibrate rewards

- [ ] Define a functional-code score from actual execution: valid API use, completion within limits, a decodable output of the expected dimensions, and a successful fresh re-render. A blank image should fail only when it contradicts the prompt.
- [ ] Define holistic instruction adherence from the original prompt and rendered image: object identity, count, color, layout, relationships, style, and exclusions. Use trusted pixel/geometry checks where possible; do not trust model-written descriptions or comments.
- [ ] Build a vision-judge adapter that receives the original prompt and actual image, and returns one overall correctness score and safety judgment. Treat text inside images as untrusted content, including attempts to instruct the judge.
- [ ] Calibrate the judge against human labels and controlled mistakes such as wrong counts, missing objects, swapped colors, and reversed relationships. Freeze evaluator versions within each experiment.
- [ ] Define the harmful-content taxonomy, severity rules, thresholds, and an uncertainty/review path. Evaluate both the request and rendered content; a benign prompt can still produce a harmful image.
- [ ] Implement a safety gate for every rendered attempt and final output. A confirmed harmful render receives a terminal negative reward and cannot be offset by functionality or adherence. Block its delivery in inference as well.
- [ ] Teach safe handling of disallowed prompts through refusal examples and refusal rewards. Penalize unnecessary refusal on allowed prompts; measure both harmful generation and over-refusal.
- [ ] Implement the provisional terminal reward table below and log every component separately. Tune weights only on development data.
- [ ] Distinguish model failures from infrastructure/grader failures: penalize invalid code, refusal misuse, and exhausted budgets; retry or exclude infrastructure failures with explicit accounting instead of inventing a score.
- [ ] Test reward exploits: constant blank images, copying prompt text onto the canvas, judge-instruction text, stale artifacts, repeated tool calls, and attempts to access grading files. Keep reward state outside the worker.

| Episode outcome | Proposed reward |
| --- | --- |
| Confirmed harmful render at any step, or a confirmed sandbox policy violation | `-1.0`, terminate; overrides every positive reward |
| Correct refusal of a disallowed request, with no harmful render | `+1.0` |
| Unnecessary refusal, failure to refuse a disallowed request without a harmful render, invalid final code, missing submission, or exhausted budget | `-0.5` |
| Allowed request, safe output, functional submitted code | `0.25 + 0.75 × adherence - 0.05 × cost` |
| Infrastructure failure or unresolved evaluator uncertainty | Retry/review; exclude from updates until resolved |

`adherence` and `cost` are normalized to `[0, 1]`. Cost measures bounded token/tool use and remains a small secondary term. Functional code is a gate for positive drawing rewards; safety is a gate for the entire episode. These weights are hypotheses to calibrate, not established optimal values.

**Done when:** controlled failures receive the expected scores and sampled reward rankings agree adequately with human judgments under a documented acceptance criterion.

## 5. Run the SFT warm-up

- [x] Implement the first-run dataset recipe: retain 80 pilot one-shot drawings and all four natural execution repairs, then generate sixteen harder natural execution/visual repair trajectories sampled from a bounded pool, with actual PNG observations with recorded errors and successful fresh submissions. Preserve original review/split provenance and the three explicit duplicate exceptions; export the 100-row selection only when the 80/20 target is complete. See `make dataset-warmup`.
- [ ] Review the 100 selected final images and freeze this first SFT dataset revision. Use the selected hosted Qwen3.5 vision model with multimodal observations. New repairs must arise naturally from harder tasks; execute model code verbatim, inspect the actual PNG during generation, and exclude the archived synthetic experiment; failed assistant actions are context-only loss-masked prefixes. Create a new held-out evaluation set because this selection consumes former pilot validation/test examples.
- [ ] Evaluate the untouched base model on a frozen development benchmark before training.
- [ ] Validate the Tinker Qwen3.5 SFT adapter against production prompts, tool schemas, actual preview PNG observations, code submissions, and refusal format.
- [ ] Train on both direct prompt–code examples and successful repair trajectories so the model learns how to operate the environment before RL.
- [ ] Mask user prompts, image/input tokens, tool outputs, and failed assistant actions from the loss; train on successful assistant responses and tool calls. Verify that packing and truncation preserve complete examples and tool-call boundaries.
- [ ] Start with a short, conservative hosted LoRA SFT pilot on Qwen/Qwen3.5-4B using Tinker. Validate a representative batch before the run; save remote optimizer/sampler checkpoints, model and renderer configuration, and dataset revision with reproducible resume metadata.
- [ ] Compare checkpoints by render success, adherence, repair success, harmful-output rate, and benign refusal rate. Select on development results rather than training loss alone.
- [ ] Require a useful execution baseline before RL; a provisional gate is at least 90% valid renders on simple held-out development tasks, with separately specified adherence and safety criteria.

**Done when:** the model can reliably call tools and produce valid drawings, and its measured strengths and failures are documented.

## 6. Add RL training in the mini environment

- [ ] Integrate a group-based RL trainer, initially GRPO, with one isolated environment per rollout. Initialize from the selected SFT checkpoint.
- [ ] First validate an end-to-end batch: sample programs, execute, grade, compute advantages, update the policy, and save/resume a checkpoint.
- [ ] Begin with one render per rollout, then enable bounded edit–run–submit episodes once the basic reward pipeline is reliable.
- [ ] Sample multiple independent trajectories per prompt; choose group size and batch size from memory/throughput measurements. Track groups with no reward variation and improve the curriculum when learning stalls.
- [ ] Apply training loss only to policy-generated tokens, excluding tool feedback. Verify correct sequence boundaries, completion masks, and rollout log probabilities for multi-turn trajectories.
- [ ] Configure a conservative learning rate, KL control against a frozen reference policy, completion limits, and checkpoint intervals. Monitor divergence and loss of SFT capabilities.
- [ ] Mix easy, compositional, repair, and safety tasks. Balance refusal tasks so always refusing cannot become a successful strategy.
- [ ] Keep terminal rewards primary; if adding intermediate repair rewards, cap them and prevent repeated failures or renders from accumulating credit.
- [ ] Log component rewards, execution/adherence/safety metrics, refusal rate, scoring coverage, truncation, KL, reward variance, render latency, judge cost, and GPU usage.
- [ ] Cache identical evaluations by prompt, code, seed, and runtime/judge versions; support resumable jobs, worker retries, and strict spend limits.
- [ ] Run controlled comparisons: base vs. SFT vs. SFT+RL, single attempt vs. repair, and different adherence evaluators. Keep evaluation token/tool budgets comparable.

**Done when:** a repeatable RL pilot improves held-out instruction adherence over SFT without an unacceptable regression in code validity, safety, or benign refusal rate.

## 7. Evaluate and package the environment

- [ ] Freeze a test set covering unseen combinations, precise counts and relationships, negative constraints, long prompts, procedural patterns, and repair tasks. Keep it separate from reward tuning.
- [ ] Build a safety evaluation set with disallowed requests, indirect requests, and allowed neighboring cases. Use controlled access for any sensitive evaluation artifacts.
- [ ] Report first-attempt success and bounded-repair success separately, plus per-category adherence, code validity, reproducibility, harmful-output rate, benign refusal rate, latency, and cost. Include sample counts and uncertainty.
- [ ] Use an independent evaluator and blinded human review on a subset to check whether reward gains represent better drawings rather than exploitation of the training judge.
- [ ] Create a lightweight comparison gallery with prompt, code, rendered image, holistic judge score, and trajectory for base/SFT/RL checkpoints.
- [ ] Provide CLI entry points for generating data, validating data, running SFT, running RL, evaluating a checkpoint, and drawing from a prompt.
- [ ] Package model adapters, drawing API documentation, versioned environment configuration, dataset provenance, experiment reports, and setup instructions.
- [ ] Keep the sandbox and content checks active in the inference path; release code and the image only after final validation.

**Done when:** a fresh setup can reproduce the data pilot, SFT run, RL pilot, and evaluation report from documented commands and pinned configurations.

## First milestone

- [ ] Complete one vertical slice: 20 varied prompts → teacher programs → isolated renders → functional/adherence/safety scores → dataset records → a short SFT smoke run → one RL update → a base/SFT/RL comparison report.
- [ ] Use that slice to measure GPU needs, render throughput, evaluator reliability, and cost before generating the full dataset or launching long training jobs.

## Implementation references

- [Original Artsy post and `ArtCanvas` example](https://blog.rashik.sh/tech/can-ai-learn-to-draw/) — motivation and drawing API direction.
- [Tinker SFT tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/) and [rendering](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/) — hosted LoRA training and multimodal loss masking.
- [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/) — candidate group-based RL and environment integration. Confirm capabilities against the pinned version during implementation.
