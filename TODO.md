# Artsy: drawing-model training environment

Build an environment that trains a small language model to turn drawing instructions into editable, reproducible Python code, using supervised fine-tuning (SFT) for a warm-up and reinforcement learning (RL) for improvement.

The starting point is the compact Cairo-backed `ArtCanvas` interface described in [Can an AI Learn to Draw?](https://blog.rashik.sh/tech/can-ai-learn-to-draw/). The central artifact is the program together with its rendered image.

Last reconciled: **2026-10-27** against the implementation and local dataset/checkpoint manifests. Checked items indicate implemented capabilities or recorded runs; they do not establish held-out quality or safety gains. Dataset and checkpoint artifacts are local and may be Git-ignored.

Current state:

- SFT: `checkpoints/sft-v1/final.json` records 29 optimizer steps on 903 complete trajectories (708 one-shot drawings, 45 repairs, 150 refusal pairs), using Qwen/Qwen3.5-4B with hosted LoRA.
- RL: the multi-signal RLHF trainer combines a frozen visual preference judge, rubric adherence, content safety, and code/action safety. A trained visual judge and explicit safety-role configuration are required before a live run.
- Still open: a comparable base/SFT/RL held-out report, human judge calibration, an audited mixed drawing/refusal RL curriculum, and live validation of the current reward pipeline.

Evidence: `checkpoints/sft-v1/training-manifest.json`, `checkpoints/sft-v1/final.json`, `datasets/sft-v1/summary.json`, and `datasets/grpo-v1/manifest.json`. See `docs/preference-rlhf.md` for the current RL workflow. Unchecked numerical targets remain proposals to validate.

## 1. Define the first experiment

- [x] Select Qwen/Qwen3.5-4B (Apache 2.0) with its vision encoder and native tool-call format, a pinned tokenizer revision, and a 262,144-token context window.
- [ ] Document a frozen, untouched-base held-out benchmark and compare it with the trained checkpoints.
- [ ] Consolidate hosted training, teacher, and judge spend limits and measured throughput/cost. Training uses Tinker; local GPU memory is not a requirement for the current hosted path.
- [x] Use Qwen/Qwen3.5-4B as a vision-language policy. It receives API documentation, the user prompt, its code, execution feedback, and rendered preview images. Keep independent evaluator scores outside policy observations.
- [x] Fix the drawing contract: static 2D drawings, one `program.py`, a documented/versioned `ArtCanvas` API, a 1920×1200 canvas, and PNG output. Include shapes, diagrams, scenes, and procedural patterns.
- [x] Define fresh final rendering and structured refusal submission (`refusal.json` with a reason and no `program.py`). Correctness, image safety, and code-safety scorers are available; complete inference safety enforcement remains open below.
- [x] Record model/tokenizer, API hashes, dataset revisions, training settings, and evaluator provenance in dataset/training manifests; share drawing limits through `drawing/config.py`.
- [ ] Complete runtime/dependency pinning and unify the configuration/provenance contract across local generation, Docker evaluation, and RL.

**Done when:** a written experiment configuration specifies the model, compute limits, output contract, and evaluation criteria.

## 2. Build the drawing runtime and mini coding environment

- [x] Implement or recover `ArtCanvas`: backgrounds, colors, paths, shapes, curves, gradients, transforms, and saving. Write concise API documentation and working examples.
- [x] Inject explicit random seeds into local and trusted final rendering.
- [ ] Pin Python images, Cairo, fonts, and remaining rendering dependencies so seeded output is reproducible across machines.
- [x] Implement episode operations: `reset(prompt, seed)`, `write_code(source)`, `run_code()`, `submit()`, and `refuse(reason)`. Keep the training adapter separate from the environment core.
- [x] Return bounded execution feedback: syntax/runtime errors, exit status, elapsed time, output dimensions, and artifact identifiers. Keep hidden grading criteria and evaluator answers outside policy observations.
- [x] Isolate Mini SWE evaluation and Cookbook RL rollouts in per-sample Docker containers with no outbound network or host mounts, CPU/memory limits, command timeouts, and a separate fresh final render.
- [ ] Extend isolation to dataset generation and the standalone renderer; harden Docker with a read-only runtime and explicit process/file-size limits. Import/AST checks are additional validation, not the isolation boundary.
- [ ] Restrict the drawing program to the documented API and approved utilities. Prevent external image downloads, arbitrary file access, and loading pre-rendered images as a substitute for drawing.
- [x] Bound episode execution and model interaction. The independent episode core permits three previews; Mini SWE/Cookbook use their configured turn, token, and command limits. Define outcomes for submission, refusal, invalid calls, timeouts, and budget exhaustion.
- [x] Re-run submitted code in a fresh worker with the same seed. Score that render so stale previews or modified artifacts cannot earn credit.
- [x] Store the full observable trajectory, source, images, errors, seed, and runtime version. Ensure reset removes all state between episodes.
- [x] Add local regression tests for cleanup, seeded rendering, timeouts, oversized output, fresh final rendering, and recovery after execution/startup failures.
- [ ] Complete live Docker isolation/adversarial checks and worker-crash recovery validation across evaluation and RL.

**Current scope:** Mini SWE drawing/evaluation and Cookbook RL run in per-sample Docker containers
with no outbound network or host mounts. Dataset generation and the standalone
renderer continue to execute locally; API/file-access restrictions remain deferred.
Python packages are pinned; host Cairo/fonts and Docker base/apt packages remain
unpinned. Local tests cover cleanup, reproducibility, timeouts, oversized PNGs,
and recovery after execution/startup failures. The independent episode core supports
local dataset generation; evaluation and RL use separate adapters sharing the
drawing/tool contract. Mini SWE replaces Pi's model loop;
Inspect logs retain its transcript, without Pi SQLite session resume.

**Done when:** teacher data generation, model rollouts, and evaluation share a verified drawing contract and isolated execution that reliably contains failing programs.

## 3. Generate prompt–code pairs and repair trajectories

- [x] Build a prompt generator from structured specifications: objects, counts, colors, positions, spatial relationships, style, background, and explicit exclusions.
- [x] Create a curriculum from simple primitives to multi-object scenes and procedural art. Include varied wording and combinations, not just many paraphrases of one template.
- [x] Have a stronger teacher generate several candidate programs per prompt using the exact API documentation and runtime constraints.
- [x] Execute candidates in the local environment (sandbox deferred per current scope); use bounded repair attempts for syntax, API, and rendering failures. Keep successful edit–run–repair trajectories for tool-use SFT.
- [x] Gate drawing acceptance on successful fresh rendering and separate correctness/image-safety judgments when configured. Correctness must reach the configured threshold (default 0.85); executable output without judges stays pending review.
- [ ] Add trusted task-specific pixel/geometry checks where reliable and calibrate semantic acceptance against human labels.
- [ ] Apply the content policy to prompts and every rendered candidate. Exclude harmful outputs from positive SFT targets; include correct refusal examples and benign lookalikes to reduce over-refusal.
- [x] Save a dataset record containing `id`, `prompt`, code or refusal, trajectory, artifact references, seed, API/runtime versions, teacher provenance, evaluator versions/holistic scores, and safety labels.
- [x] Reject exact source/pixel duplicates, filter lexical prompt duplicates, export a combined accepted dataset, and document the historical duplication audit in `docs/dataset-duplication-audit.md`. Automatic train/validation/test splitting has been removed.
- [ ] Fix the documented family/difficulty/skill scheduling correlations; add semantic task grouping and perceptual near-duplicate review across datasets and evaluation prompts.
- [ ] Audit a stratified human-reviewed sample for correctness, diversity, safety, and evaluator agreement. Version accepted data and rejection reasons.
- [x] Run a 1,000-prompt generation pilot: 853 accepted records, 67 rejected, and 80 infrastructure errors; record acceptance rate (85.3%) and token/request accounting. Add 50 malicious-code refusal pairs for the 903-record SFT release.
- [ ] Convert usage accounting into cost per accepted example and choose further data scale from learning curves, diversity audits, and budget.

**Done when:** a reproducible command produces a versioned, audited dataset of executable prompt–code pairs, repair trajectories, and refusal examples.

**Current implementation:** `.venv/bin/python -m drawing.generate_dataset` generates
LLM-authored seeded prompt plans, runs a configurable teacher through bounded
repairs, and checkpoints candidates for resume. Separate correctness and image-safety
judges inspect preview/final PNGs; a code-safety judge can block malicious source
before local execution and is required by the current automated drawing-acceptance
path. Missing judgments leave drawings pending review. Authored refusal pairs are
also supported. Exact source/pixel deduplication and lexical prompt filtering exist;
semantic diversity, human audits, and calibrated safety remain open. Historical
pilot acceptance does not prove conformance to every later pipeline check.

## 4. Implement and calibrate rewards

- [x] Gate positive RL correctness reward on a successful fresh render with a valid PNG at the expected dimensions; record execution validity separately. Invalid/missing output receives zero correctness reward, and budget-limited episodes cannot earn positive final reward.
- [ ] Enforce documented-API use and calibrate blank-image handling against prompt intent.
- [x] Judge holistic correctness from the original request and actual rendered PNG, returning a normalized score and reason.
- [ ] Add reliable pixel/geometry checks and controlled count/color/layout/relationship error cases.
- [x] Implement shared correctness/image-safety prompts and strict JSON parsers, treating image text as untrusted. Support HTTP judges and a native Tinker Qwen3.8 vision correctness judge; record renderer/revision/token budget and reject malformed or truncated responses. RL uses separate adherence, visual preference, image-safety, and code-safety judge clients.
- [ ] Calibrate the judge against human labels and controlled mistakes such as wrong counts, missing objects, swapped colors, and reversed relationships. Freeze evaluator versions within each experiment.
- [ ] Define the harmful-content taxonomy, severity rules, thresholds, and an uncertainty/review path. Evaluate both the request and rendered content; a benign prompt can still produce a harmful image.
- [ ] Implement a safety gate for every rendered attempt and final output. A confirmed harmful render receives a terminal negative reward and cannot be offset by functionality or adherence. Block its delivery in inference as well.
- [x] Include harmful-image and malicious-code refusal demonstrations in SFT; provide refusal/category scoring and code-safety review in Inspect.
- [x] Implement hard safety/refusal rewards in RL.
- [ ] Assemble an audited mixed drawing/refusal RL prompt set and measure harmful generation and over-refusal together.
- [x] Implement the terminal reward table below and log reward components separately.
- [ ] Calibrate reward weights on development data and validate the frozen visual judge against human preferences.
- [x] Record generation infrastructure errors separately, support `--retry-errors`, and raise on trusted-scorer/judge failures in RL rather than inventing correctness scores.
- [ ] Validate retry/exclusion and spend accounting end to end for live training infrastructure failures.
- [ ] Test reward exploits: constant blank images, copying prompt text onto the canvas, judge-instruction text, stale artifacts, repeated tool calls, and attempts to access grading files. Keep reward state outside the worker.

**Current terminal reward:**

| Episode outcome | Reward |
| --- | --- |
| Malicious code/actions or harmful final image | `-1.0` |
| Correct submitted refusal on a trusted refusal prompt | `+1.0` |
| Over-refusal, failed refusal, invalid final drawing, or missing submission | `0.0` |
| Benign request with safe, valid submitted drawing | `0.5 × visual + 0.5 × adherence` |
| Trusted scorer or judge failure | Raise; do not invent a score |

`visual` is the frozen judge's within-group pairwise win fraction, including half
credit for ties and both A/B orders. `adherence` is the 0–4 rubric score divided by
four. There is no cost reward. Budget-limit terminations cannot earn positive
reward. Image safety reviews final renders; per-preview safety gating remains open.

**Done when:** controlled failures receive the expected scores and sampled reward rankings agree adequately with human judgments under a documented acceptance criterion.

## 5. Run and evaluate SFT

- [x] Replace the historical 100-row 80/20 warm-up selection with the combined dataset workflow. There is no `make dataset-warmup` target. The recorded SFT release has 903 trajectories: 753 drawings and 150 refusal pairs.
- [x] Convert recorded code, execution feedback, and actual preview PNGs into separate bash write → preview → submit demonstrations using the native Qwen3.5 tool format. Refusals write only `refusal.json`. Wrappers are synthesized; these are converted demonstrations, not collected Mini SWE transcripts.
- [x] Train all converted assistant actions, including unsuccessful executed attempts in successful trajectories. Mask instructions, user/tool messages, image context, and assistant headers; omit malformed teacher replies that were never executed. The current policy is `assistant-only`, conversion v9.
- [x] Validate artifact hashes and complete trajectory lengths before hosted training; never pack or truncate examples. Support offline `--dry-run` export and `--check-batch` tokenization/preflight.
- [x] Complete hosted LoRA SFT on Qwen/Qwen3.5-4B: 903 trajectories, batch size 32, rank 32, learning rate 2e-4, 29 optimizer steps. Save remote optimizer/sampler paths and dataset/model/configuration provenance in `checkpoints/sft-v1/`.
- [x] Implement deterministic batch order, checkpoint compatibility checks, optimizer resume, and constant/cosine learning-rate schedules.
- [ ] Complete a stratified human correctness/diversity/safety audit of the final dataset. The training manifest reports zero unreviewed records; that is not evidence of independent human review.
- [ ] Establish semantic separation of held-out tasks from all SFT data; former pilot split labels no longer define holdouts.
- [ ] Compare base and SFT checkpoints on render validity, adherence, repair success, harmful-output rate, benign refusal rate, latency, and cost.
- [ ] Document whether SFT meets the provisional 90% valid-render gate on simple held-out tasks and separate adherence/safety criteria. This gate still needs a reported measurement before the next RL experiment.

**Done when:** the trained model's held-out strengths and failures are documented, beyond training loss and successful checkpoint export.

## 6. Run and evaluate RL in the mini environment

- [x] Integrate Tinker Cookbook `EnvFromMessageEnv`, grouped environment/dataset builders, and `rl.train.main`, initialized from the selected SFT state. Cookbook owns token-level PPO updates, grouped advantages, and reference-KL reward shaping.
- [x] Support bounded multi-turn bash write → run → fix → submit rollouts with actual PNG observations, one network-isolated Docker workspace per trajectory, and a separate fresh Docker final render. Inspect remains the independent evaluation harness.
- [x] Freeze 200 benign training prompts in `datasets/grpo-v1/` with difficulty/skill coverage, provenance, and cross-source lexical overlap checks. These are training prompts, not a held-out benchmark.
- [x] Delegate policy-token loss/log-probability handling to Cookbook; support per-prompt reward mean-centering without variance normalization, and removal of constant-reward groups.
- [x] Combine terminal visual preference and rubric adherence equally, with hard safety/refusal outcomes; zero positive reward on budget-limit termination. No intermediate render reward is accumulated.
- [x] Save Cookbook metrics/checkpoints, grouped rollout logs, and per-episode source/final PNG/judgments/reward components. Validate dataset/source/configuration compatibility on resume and support `--finalize-only` checkpoint-export recovery.
- [ ] Validate live optimizer-resume equivalence and failure recovery; existing configuration/export tests do not establish a successful interrupted-run recovery.
- [ ] Measure reward variance, throughput, judge cost, and loss of SFT capabilities; use them to tune group sizes, curriculum, and concurrency.
- [x] Add independent visual, adherence, image-safety, and code-safety rewards with refusal support.
- [ ] Train and calibrate the frozen visual preference judge; run a small live multi-signal policy smoke test.
- [ ] Complete component/safety/refusal/cost reporting, evaluation caching keyed by prompt/code/seed/runtime/judge version, worker retries, and strict spend limits.
- [ ] Run controlled base vs. SFT vs. SFT+RL, single-attempt vs. repair, and evaluator comparisons with equal token/tool budgets. Confirm held-out improvement without unacceptable validity, safety, or benign-refusal regressions.

**Done when:** a repeatable RL run shows measured held-out improvements over SFT under documented validity and safety criteria. Training completion alone does not satisfy this gate.

## 7. Evaluate and package the environment

- [ ] Freeze a test set covering unseen combinations, precise counts and relationships, negative constraints, long prompts, procedural patterns, and repair tasks. Keep it separate from reward tuning.
- [x] Add harmful-image refusal cases and `evaluation/malicious-code-10.jsonl`, combined with drawing tasks in `evaluation/generated-110.jsonl`.
- [ ] Audit safety-evaluation coverage, add indirect requests and allowed neighboring cases, and document semantic separation from training data. Use controlled access for sensitive artifacts.
- [x] Implement Inspect summaries for render validity, execution-repair behavior, authored difficulty, refusal categories, code-safety judgments, usage, and incomplete/error accounting.
- [ ] Publish comparable first-attempt/bounded-repair and per-category adherence/validity/safety/refusal/latency/cost results with sample counts and uncertainty; distinguish execution recovery from visual correction.
- [ ] Use an independent evaluator and blinded human review on a subset to check whether reward gains represent better drawings rather than exploitation of the training judge.
- [x] Provide a local generation UI with checkpoint discovery, saved-run reopening, PNG/source downloads, and Inspect trajectory/preview inspection.
- [ ] Build a side-by-side base/SFT/RL comparison gallery with prompts and independent judge scores.
- [x] Provide Python CLI entry points for dataset generation, SFT validation/training, Cookbook RL, evaluation, and drawing; document setup and Makefile shortcuts.
- [ ] Package model adapters, drawing API documentation, versioned environment configuration, dataset provenance, experiment reports, and setup instructions.
- [ ] Keep the sandbox and content checks active in the inference path; release code and the image only after final validation.

**Done when:** a fresh setup can reproduce the data pilot, SFT run, RL pilot, and evaluation report from documented commands and pinned configurations.

## Next milestone

- [x] Complete data generation → SFT → isolated correctness-scored RL → saved checkpoint export. The recorded runs exceed the original one-update smoke target.
- [ ] Finish the vertical slice with a frozen, semantically separated base/SFT/RL evaluation and comparison report, including independent/human review and safety/refusal outcomes.
- [ ] Summarize measured throughput, judge reliability, and actual hosted generation/training/evaluation cost before choosing the next data or training scale.

## Implementation references

- [Original Artsy post and `ArtCanvas` example](https://blog.rashik.sh/tech/can-ai-learn-to-draw/) — motivation and drawing API direction.
- [Tinker SFT tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/) and [rendering](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/) — hosted LoRA training and multimodal loss masking.
- [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/) — current grouped-rollout PPO trainer and environment integration; dependencies are pinned in the training requirements.
