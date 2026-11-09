# Artsy

Train and evaluate models that draw with Python. An agent writes ArtCanvas code,
inspects PNG previews, repairs its work, and submits an image with editable source.
The project includes a local web UI, teacher dataset generation, hosted LoRA SFT,
RLVR followed by preference-based GRPO training, and an independent Inspect evaluation harness.

Completed experiment setups, results, and conclusions are recorded in the
[experiment log](EXPERIMENT_LOG.md).

The current experiments use PartiPrompts: 914 SFT prompts, 327 reserved RL
prompts, and 391 evaluation prompts. Experiment 1 compares Qwen3.5-4B LoRA
ranks 32, 64, and 128. See [experiment evaluation](docs/benchmark.md) for the
generation and judging workflow. GRPO and all UI workspaces remain supported.

Model sampling and training run on Tinker; tokenization and image processing run
locally. The bundled inference bridge defaults to `Qwen/Qwen3.5-4B`. Dataset
authors and evaluation/RL judges default to GPT-6 Luna through OpenRouter when
configured with [`.env.example`](.env.example).

## Setup

Run commands from the repository root. Use Python 3.14 and `uv` for the project
environment. Evaluation can use a separate Python 3.12 environment because Inspect
and inference have different dependency constraints.

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
  -r inference/requirements-tinker.txt -r drawing/requirements.txt

uv venv --python 3.12 .eval-venv
uv pip install --python .eval-venv/bin/python \
  -r evaluation/requirements.txt -r runtime/requirements.txt
```

For a new setup, copy `.env.example` to `.env` and fill in the keys:

- `TINKER_API_KEY` authenticates hosted inference and training. Alternatively,
  authenticate with `.venv/bin/tinker auth login`.
- `OPENROUTER_API_KEY` authenticates the configured prompt writer, teacher, and
  evaluation/RL judges.

Drawing, evaluation, and GRPO load the repository's `.env`; exported environment
variables take precedence. For the standalone SFT CLI, export `TINKER_API_KEY` in
your shell or use Tinker CLI credentials. Generated datasets, episodes, evaluation results, checkpoints, caches,
and `.env` are Git-ignored. A fresh checkout does not include training data or
trained checkpoints. Paths under `datasets/` and `checkpoints/` below refer to
local artifacts you must generate or supply.

Drawing through the UI, dataset generation, Inspect evaluation, and RL require
Docker Engine 24.0.6+ with Compose 2.22+. Start Docker before those workflows.
Build the isolated renderer image before dataset generation or `drawing.run`:

```sh
docker build -f evaluation/Dockerfile -t artsy-rl-runtime:latest .
```

Rebuild this image after changes to the runtime, renderer, or rendering contract.
`ARTSY_RENDER_IMAGE` can select another locally built image; rendering never pulls
an image automatically. For explicitly trusted programs, `--renderer local` on
`drawing.run` or dataset generation uses host permissions and requires system Cairo:

```sh
# macOS
brew install cairo pkg-config

# Debian/Ubuntu
sudo apt-get install build-essential pkg-config libcairo2-dev fonts-dejavu-core
```

## Draw and review in the UI

Start the native Tinker bridge in one terminal:

```sh
HF_HOME="$PWD/.cache/huggingface" make serve
```

Install the local Kokoro voice once, then start the UI in another terminal:

```sh
make tts-setup
make ui
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173) for the voice-first Artsy
experience: describe a drawing, watch its progress, then ask for changes by voice
or text. It uses GPT-6 Luna through OpenRouter with ten internal turns per
response; configure `OPENROUTER_API_KEY` and start Docker. The native bridge is
only needed for Qwen. The developer [Studio](http://127.0.0.1:4173/studio) generates drawings;
Prompt review and SFT candidates review training data; Runs launches evaluation,
SFT, RLVR, and preference RLHF jobs. RL prompts and RL rollouts browse saved training artifacts.
The `/preferences` workspace supports blind ranking of existing model outputs.
Browsing and review do not require Docker or a running bridge. Generation and
training consume provider credits.

To review a local dataset, pass its files explicitly:

```sh
make ui ARGS="--dataset datasets/my-run/prompts.jsonl --candidates datasets/my-run/candidates.jsonl"
```

Without `--dataset`, Studio and Prompt review use the frozen evaluation suite.
Studio offers Qwen3.5 4B, Qwen3.6 35B-A3B, Qwen3.8 27B, GPT-6 Luna, and compatible
saved checkpoints advertised by the bridge. GPT-6 Luna uses OpenRouter directly
and does not require the local bridge. See [UI configuration, reviews, job
controls, and RL monitoring](ui/README.md).

## Draw from the CLI

With Docker and `make serve` running:

```sh
# Paid preflight: check native tool calling and inline PNG input.
.eval-venv/bin/python -m evaluation.check_tinker

make draw ARGS='"Draw a blue circle centered on a white background."'

.eval-venv/bin/python -m evaluation.draw "Draw three red squares." \
  --episode episodes/squares --seed 42
```

The drawing and evaluation CLIs now default to **mini-SWE-agent 2.2.3**
(`--harness mini-swe`). The model gets a general `bash(command)` tool and a
persistent `/workspace`, with Python, `sed`, `patch`, `diff`, and `rg` available.
It can inspect files and apply targeted edits instead of replacing the entire
program. When editing an existing drawing, its source is placed directly in
`/workspace/program.py` before the first model turn.

The agent chooses its Python filenames and can create helper modules. Submit with
`artsy-submit PATH_TO_ENTRYPOINT`. The native mini completion command also works
when there is only one Python file or an entrypoint was selected by a preview.
Previews are optional: `artsy-preview PATH_TO_ENTRYPOINT` renders Python, while
`artsy-preview output.png` views an existing PNG. Both return actual image feedback.
`artsy-preview` and `artsy-submit` are installed shell executables and work inside
compound commands and scripts. Python entrypoints may live anywhere
readable in the container; external entrypoints snapshot their sibling modules. No preview is
required to submit, including after an edit.

Submission snapshots the Python files in the workspace. Final scoring recreates
those modules in a fresh directory and runs the selected entrypoint. Non-Python
workspace assets are not included. The backend stores this snapshot as its own
`program.py` artifact; that name is not imposed on the agent's source files.
The mini-SWE prompt includes the complete [ArtCanvas API reference](docs/artcanvas.md)
instead of asking the agent to search a large pycairo documentation directory.
The original pycairo reference remains available for debugging at
`/opt/artsy/pycairo-api.pyi`, but is not the model-facing drawing API.

This is trajectory contract v4. Use `--harness drawing-tools` to run historical
checkpoints trained on `update_drawing`, `read_drawing`, `preview_drawing`,
`submit_drawing`, and `refuse_drawing` (contract v3). Existing SFT/GRPO checkpoints
retain their original tool contract; the new mini interface needs a new SFT run.
The programmatic task and dataset APIs retain their historical default for
existing training callers; pass `harness="mini-swe"` to select the new harness.

Episodes retain Inspect logs, scored artifacts, and final `program.py`,
`image.png`, and `result.json` with hashes. Completed matching requests can be
reused. Interrupted requests or changed artifacts require a new episode directory;
only one process should own each directory. Studio saves its runs under
`episodes/ui/` and exports completed traces for inspection.

### Inference configuration

The bridge listens at `http://127.0.0.1:8080/v1` and uses Tinker's native
`SamplingClient`. It downloads tokenizer/image-processor metadata rather than
model weights and translates model-native tool calls into OpenAI tool calls.
Qwen sampling uses the family's `disable_thinking` renderer. Requests must use
`stream=false`; streaming is rejected before sampling.

Set `ARTSY_TINKER_MODEL_PATH=tinker://...` to serve a sampler checkpoint through
the default model alias. Compatible completed checkpoints are also discovered
from `checkpoints/*/final.json`; intermediate GRPO samplers are discovered from
training manifests and `checkpoints.jsonl`. The catalog refreshes periodically.
Model profiles live in [`inference/model_catalog.py`](inference/model_catalog.py).
The bridge is a local development service.

For Qwen3.5 vision, use the native bridge rather than Tinker's beta
OpenAI-compatible endpoint, which the project's preflight identifies as rejecting
image input for this model. Leave `ARTSY_MODEL_URL` unset for the bridge. To use
another OpenAI-compatible provider, set `ARTSY_MODEL_URL` and, if needed,
`ARTSY_MODEL_API_KEY`; judge credentials are configured independently.

For another supported base model, keep the bridge running and select it explicitly:

```sh
.eval-venv/bin/python -m evaluation.check_tinker --model Qwen/Qwen3.6-35B-A3B
.eval-venv/bin/python -m evaluation.evaluate \
  --model Qwen/Qwen3.6-35B-A3B \
  --execution-only --limit 1 \
  --output eval-results/qwen3.6-smoke
```

## Evaluate models

For the current PartiPrompts experiments, select the held-out dataset explicitly:

```sh
make eval ARGS="--dataset datasets/parti-prompts/evaluation/prompts.jsonl --execution-only --output eval-results/parti-new-run"
```

For HPSv3 evaluation, generate final images with `--execution-only`, then use
`evaluation.compare_pipeline --judge hpsv3` to score them after every generation
run completes. HPSv3 loads only for the scoring batch and exits afterward; no
persistent scorer is required. See [HPSv3 setup](docs/hpsv3.md) and the
[two-phase evaluation workflow](docs/benchmark.md).

Without `--dataset`, the CLI still selects the 50-prompt regression suite at
[`evaluation/generated-50-2026-10-28/prompts.jsonl`](evaluation/generated-50-2026-10-28/prompts.jsonl),
frozen by [`evaluation/benchmark-v3/manifest.json`](evaluation/benchmark-v3/manifest.json).
It contains 35 ordinary drawings, five benign boundary prompts, five harmful-image
refusals, and five malicious-code refusals. Its prompt/safety review is recorded
in the manifest. Keep these prompts and evaluation labels out of SFT/RL training.

```sh
# Local validation and plan: no model requests or Docker.
make eval ARGS="--dry-run"

# Execution smoke test with Docker and the bridge running; skips all judges.
make eval ARGS="--execution-only --limit 1 --output eval-results/base-smoke"

# Full scoring; configure all judge roles from .env.example first.
make eval ARGS="--output eval-results/base"

# Rebuild metrics from saved artifacts without repeating model calls.
make eval ARGS="--report-only --output eval-results/base"
```

Use a fresh output directory for each evaluation. The first run builds the Docker
image. `--dataset PATH` selects custom JSONL prompts with `id`, `prompt`, and
`seed`; `--epochs 3` repeats each prompt three times.

Evaluation and checkpoint comparisons use these defaults from
[`evaluation/harness.py`](evaluation/harness.py):

| Setting | Default |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B`, overridden by `ARTSY_MODEL` or `--model` |
| Model turns | 25, with a countdown and final-turn submission reminder |
| Temperature | 0 |
| Concurrent samples / model connections | 10 / 10 |
| Docker sandbox cap | Same as concurrent samples |
| Episode and model-request deadline | Disabled |
| Aggregate output-token budget | Disabled |
| Per-response output-token cap | 32,768 tokens |
| Render / preview tool timeout | 60 / 65 seconds |
| Source / PNG size limit | 64 KiB / 8 MiB |

Use `--max-samples`, `--max-connections`, and `--max-sandboxes` to control
concurrency. Each active sample has its own container; the evaluation Compose
workspace sets no additional CPU or memory quota. `--time-limit 600` adds an
episode/model-request deadline. `--max-tokens N` caps each response;
`--token-limit N` sets a cumulative output budget, or add `--token-limit-type all`
to count input and output. Zero disables those optional caps. Qwen bridge profiles
use 262,144-token contexts and cap responses at 32,768 tokens or remaining context capacity, whichever is smaller;
`/health` reports the selected profile and limits.

Full scoring makes independent adherence, image-safety, and code-safety calls.
Adherence is an integer 0–4 rubric normalized to 0–1. Code safety audits final
Python and the complete tool history, including replaced source revisions, before
final rendering; malicious behavior blocks that render and gives zero adherence.
The audit does not intercept earlier preview executions.
A valid submitted refusal on a refusal-labeled task is scored without vision calls.
Benign over-refusals receive zero adherence.

Reports separate drawing render rates, refusal success by category, harmful-image
findings, code violations, and sample errors. A missing image is not evidence of
image safety. Malformed judge replies and infrastructure failures produce errors
and a nonzero exit; inspect `sample_errors` before comparing averages. Judge labels
still require human calibration.

Outputs include `logs/`, per-sample source/images/judgments under `artifacts/`, and
`summary.json` with scores, errors, usage, and provenance. Open transcripts with:

```sh
.eval-venv/bin/inspect view --log-dir eval-results/base/logs
```

For paired checkpoint comparisons, run the bridge with
`ARTSY_TINKER_MODEL_PATH` empty so the base alias remains the base model, then use:

```sh
.eval-venv/bin/python -m evaluation.run_benchmark \
  --output eval-results/checkpoint-comparison
```

This selects the base model and available completed adapters. Repeat
`--model NAME=MODEL_ID` to choose at least two specific catalog entries. See
[benchmark scoring, blind review, and calibration](docs/benchmark.md).

## Generate and review training data

The prompt-writing LLM authors requests across scene families, drawing skills,
styles, voices, lengths, and difficulty levels. Labels guide authoring and human
review; they are not automatic acceptance checks. The default 50-prompt mix is
35 drawings, five benign boundary drawings, five harmful-image refusals, and five
malicious-code refusals. With an explicit `--count N`, omitted slice counts are
zero. Refusal and boundary counts are disjoint subsets of the total.

Configure the prompt writer and vision-capable teacher using `.env.example`:

```sh
# Prompt writing only: paid API calls and checkpoint writes, no rendering.
make dataset ARGS="--prompts-only --count 20 --difficulty mixed --output datasets/my-prompts"

# Collect two teacher candidates per ordinary prompt, with isolated Docker rendering.
make dataset ARGS="--harness drawing-tools --count 100 --harmful-count 10 --malicious-code-count 10 --output datasets/my-run"

# Inspect all prompts and candidates, then export approved SFT records.
make ui ARGS="--dataset datasets/my-run/prompts.jsonl --candidates datasets/my-run/candidates.jsonl"
```

`--prompts-only` calls only the author and saves a resumable plan. Its legacy alias
`--dry-run` also makes API calls and writes files. `--difficulty` accepts `easy`,
`medium`, `hard`, `very-hard`, or `mixed` (balanced quotas). `--input prompts.jsonl`
bypasses prompt authoring. Minimal custom records look like:

```json
{"id":"blue-circle","family":"circles","prompt":"Draw one centered blue circle on white.","seed":42,"difficulty":"easy","skills":["primitives","positioning"]}
```

Prompt writing uses one request per prompt, with up to eight in flight
(`--prompt-batch-size 8`) and up to five attempts per prompt. `--workers` separately
controls teacher/render concurrency (1–128), including the HTTP connection and
Docker render thread pools. Workers take the next prompt as soon as they finish,
without waiting for a group of prompts to complete. Model-authored wording is checkpointed; the
seed alone does not reproduce it. Repeat the original command to resume missing
work; `--retry-errors` retries candidate infrastructure failures. Changed settings
or incompatible generator formats require a fresh output directory.

For new experiment-5 trajectories, generation defaults to mini-SWE-agent and
50 drawing prompts, with no refusal pairs. It omits unsolicited image text and
exposes no refusal tool. Use drawing-only input specs; explicit refusal specs
require `--harness drawing-tools`. Mini generation requires Docker, including
when a renderer is supplied programmatically.

Build and verify the editing workspace before collecting trajectories:

```sh
make agent-image
make dataset ARGS="--harness mini-swe --input datasets/my-prompts/prompts.jsonl --output datasets/experiment-5 --candidates 2"
```

`make agent-image` builds `artsy-editing-agent:latest` and smoke-tests an
actual ArtCanvas drawing inside the network-isolated editing container. Dataset
generation performs the same preflight before any teacher/model calls, comparing
the installed ArtCanvas, API reference, and rendering helpers with the repository
sources. If you see a stale-image or `ModuleNotFoundError: artcanvas` diagnostic,
run `make agent-image` and restart generation in a **new output directory** if the
earlier run already saved candidates.

For mini-SWE, the editing workspace and independent final renderer now use the
**same verified, immutable image ID**; a separate `artsy-rl-runtime:latest` build
is not required for mini-SWE collection. `ARTSY_AGENT_IMAGE` can select a custom
editing image, provided its installed runtime matches the checked-out sources.
Its name, image ID, and preflight result are recorded in the dataset manifest.
The editing container has no network, host mounts, or credentials. Its files
persist within one candidate. Interrupted collection saves a bounded, hashed
checkpoint of generated files in `/workspace`, `/tmp`, and `/home/artsy`, including
PNGs and other binary artifacts. Recovery restores those files before continuing
the same conversation through `--retry-errors`, retaining its original turn budget. Empty provider replies consume a turn and receive normal
missing-tool feedback instead of terminating collection.

Each response counts toward the 12-turn SFT limit, including malformed replies.
Saved trajectories retain the actual shell commands, tool results, source hashes,
and PNG observations. SFT export validates the final source and image hashes and
supervises every executed action in mini trajectories: targeted edits accumulate,
so only supervising the last patch would omit earlier work. Images and tool
feedback remain context. Historical records keep their existing loss policy.
A dataset manifest records the pinned harness, schemas, prompts, and source hashes;
changed configuration requires a new output directory.

`--harness drawing-tools` retains the previous complete-source generation flow,
including authored refusal pairs and optional trusted `--renderer local` execution.

Generation performs no automated content, safety, quality, or duplicate filtering.
Parsing, execution budgets, PNG validity, and artifact integrity checks still apply.
The 12-turn limit is an additional structural acceptance requirement. Generation
requires an HTTP endpoint supporting native tool calls and image input. Use a new
output directory for the new format; old JSON drafts cannot be reused with it.
Generated Python runs in disposable containers without host mounts, host credentials,
or network access. `--renderer local` explicitly opts into host execution for
trusted programs. All outcomes remain
in `candidates.jsonl`; no train/validation/test splits are assigned automatically.

For the local PartiPrompts dataset, reproduce the prompt assignments with
`python3 scripts/prepare_parti_prompts.py`.
The canonical SFT dataset is `datasets/parti-prompts/sft/`: **1,142 prompts**,
2,284 generated candidates in `candidates.jsonl`, and 2,219 previously validated
trajectories in `training-ready.jsonl` (1,134 unique prompts; visual review pending).
This includes the 228 prompts restored from the historical extended evaluation.
Evaluation uses the original **163** prompts; the reserved RL pool has **327**: **227 for RLVR** and **100 for RLHF**, with no overlap. Preference is empty.
Review the combined dataset with
`make ui ARGS="--dataset datasets/parti-prompts/sft/prompts.jsonl --candidates datasets/parti-prompts/sft/candidates.jsonl"`.
Preserved ArtCanvas trajectories are historical training sources. Recent native
pycairo trajectories are incompatible with the restored ArtCanvas runtime; generate
new mini-SWE trajectories or explicitly validate and convert older ArtCanvas records.
Original batch directories remain as historical sources. The 391-prompt evaluation
belongs to Experiment 2 and is not held out for models trained on the restored prompts.
Assignments are deterministic and stratified by upstream Category and Challenge.
Use `--preference-count 228` to reproduce the historical four-way split, adding
`--preference-to-evaluation` only to reproduce Experiment 2's extended evaluation.
Dataset files and their manifest are local,
Git-ignored artifacts; the preparation script reproduces them from the pinned TSV.

The output includes prompt plans/checkpoints, `prompts.jsonl`, candidate records,
rendered artifacts, `manifest.json`, and `summary.json`. In **SFT candidates**,
review full trajectories and images, set Approve/Reject/Pending, and save your
reviewer name. **Export approved SFT** writes `accepted.jsonl` containing only
approved, structurally trainable records with valid artifacts. Keep the referenced
source/images available for training. Prompt approval is separate from candidate
approval. See the [pipeline diagrams](docs/pipeline-diagrams.md) and
[review output paths](ui/README.md).

## Supervised fine-tuning

SFT trains a hosted LoRA adapter; artifact validation and tokenization run locally.
No local GPU or model weights are needed. Install training dependencies into the
project environment:

```sh
uv pip install --python .venv/bin/python -r training/requirements.txt

# Validate artifacts and inspect a converted conversation; no API calls.
.venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl --dry-run

# Check all token/image inputs and masks; may download tokenizer metadata.
.venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl --check-batch

# Hosted training; set this key or authenticate with the Tinker CLI.
export TINKER_API_KEY=your-tinker-api-key
.venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --output-dir checkpoints/sft-run

# Resume from local metadata containing hosted weights and optimizer state.
.venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --output-dir checkpoints/sft-run \
  --resume-from-checkpoint checkpoints/sft-run/checkpoint-000050.json
```

The CLI defaults to Qwen3.5 4B; `--model` also supports the configured Qwen3.6 and
Qwen3.8 vision profiles. The Runs UI has its own model default, Qwen3.6 35B-A3B
with the 262,144-context training suffix.

| Setting | CLI default |
| --- | --- |
| Epochs / trajectories per optimizer step | 1 / 8 |
| Learning rate / schedule | `2e-4` / cosine |
| LoRA rank | 32 |
| Maximum trajectory length | 262,144 tokens |
| Checkpoint cadence | Every 50 steps and at completion |
| Loss policy | `final-write-final-preview-submit` |

Each complete trajectory is one example. Earlier attempts, execution feedback,
and actual previews remain context. The default loss policy supervises the final
write, final preview call, and submission; earlier assistant actions are masked.
Refusal writing and submission also receive loss. Use `--loss-policy assistant-only`
to supervise every assistant action. Tool feedback/images never receive loss.
Recorded teacher reasoning summaries remain ordinary assistant text and are
supervised only with their associated action. Judge labels/critiques are excluded.
Native trajectories require recognized separate write and preview commands for
the default mask; ambiguous shell actions require the assistant-only policy.

**Experiment 6 (mini-SWE only):** pass `--loss-policy error-aware` to exclude
code-writing tool calls from the SFT loss when their shell operation failed or
a subsequent source-hashed `artsy-preview` recorded a render failure. Edits
since the last successful source preview are considered unverified and masked
on such a failure; successful intermediate edits, debugging commands, preview
calls, corrective edits, and submission remain supervised. All actions and
feedback remain in the causal context. Detection uses conservative shell-source
write patterns; it is not a general static analyzer and cannot identify visual
bugs or edits with no observed failure. The standard default is unchanged,
so control and treatment can train on the **same accepted mini-SWE JSONL**.
This policy is rejected for legacy drawing-tool/bash records.

Sequences are neither packed nor truncated. Preflight validates complete
trajectories before creating a hosted training client. `--dry-run --export PATH`
exports converted conversations, and `--max-steps 1` limits a hosted smoke run.
Cosine resumes must retain the original schedule length; constant-rate runs can
extend epochs/steps. Changing the loss policy or conversion version requires a
new run.

Training output directories have exclusive process locks. SFT resumes require the
latest checkpoint in its original output directory; completed runs cannot be
resumed in place. Interrupted final sampler export can be recovered by resuming
the latest checkpoint even when all optimizer steps are complete. Manifests,
checkpoint metadata, and final exports are published atomically.

Outputs include `training-manifest.json`, `metrics.jsonl`, checkpoint JSON files
with remote state paths, and `final.json` with training-state and sampler paths.
These are metadata, not local full-model weights. The bridge discovers compatible
samplers for evaluation and Studio.

## RLVR and RLHF training

The recommended policy pipeline is **SFT → RLVR → RLHF**. Both RL stages use
Tinker Cookbook PPO/GRPO with grouped multi-turn Docker episodes, reference-KL
regularization, and hosted LoRA checkpoints. The inference bridge is unnecessary.

**RLVR** uses deterministic **negative-only penalties**, with no HPSv3 service.
The maximum reward is **0**; successful actions and valid images earn no positive
bonus:
- **−0.75** when the final submitted source fails to produce a valid 1920×1200
  PNG in an independent network-isolated Docker scorer, or is not submitted
- **−0.125 per invalid tool call**: malformed replies or calls that violate the
  `bash(command)` schema; an episode with no tool calls incurs one penalty
- **−0.125 per failed execution**: nonzero shell exit status, failed preview
  feedback, or a recorded execution error

Penalties accumulate for each error rather than being divided by the number of
successful actions, so trivial successful commands cannot dilute earlier errors.
Per-rollout `result.json` stores `image_penalty`, `tool_penalty`,
`execution_penalty`, aggregate `reward`, diagnostic success ratios, and raw
error counts.

**RLHF** starts from the RLVR checkpoint and uses the normalized HPSv3 preference
score for valid submitted images. Its KL reference remains the original frozen
SFT policy, propagated through the RLVR checkpoint. Direct SFT → RLHF initialization
remains supported for older experiments but skips the new stage.

```sh
uv pip install --python .venv/bin/python -r training/requirements-grpo.txt

# Check the RLVR plan without Tinker, Docker or HPSv3 calls.
.venv/bin/python -m training.rlvr --dry-run \
  --init-checkpoint checkpoints/sft-run/final.json --max-steps 1

# Train one RLVR update. Docker and Tinker credentials are required.
make rlvr ARGS="--init-checkpoint checkpoints/sft-run/final.json --max-steps 1 --output-dir checkpoints/rlvr-smoke"

# Check the preference stage using the RLVR final checkpoint.
.venv/bin/python -m training.grpo --dry-run \
  --init-checkpoint checkpoints/rlvr-smoke/final.json --max-steps 1

# Train one RLHF update. Configure ARTSY_HPSV3_MODEL and ARTSY_HPSV3_URL.
make rlhf ARGS="--init-checkpoint checkpoints/rlvr-smoke/final.json --max-steps 1 --output-dir checkpoints/rlhf-smoke"
```

The stage entrypoints default to `checkpoints/rlvr-v1` and `checkpoints/rlhf-v1`,
with `make rlhf` taking `checkpoints/rlvr-v1/final.json` as its starting state.
The historical `make grpo` still defaults to the older SFT input for compatibility.

For a full run, omit `--max-steps 1` and choose a new output directory. Repeat that
original command to resume. `--finalize-only --output-dir PATH` recovers the final
export after training completes.

**Both RL stages use `--harness mini-swe` by default.** It uses the same mini-SWE-agent
2.2.3 `bash(command)` schema, prompt, persistent editing workspace, optional
`artsy-preview` images, and `artsy-submit` (or native completion) as new SFT
trajectories. Submissions snapshot all Python modules; a separate network-isolated
container re-renders their bundled source for final-image scoring.

RLVR requires mini-SWE to verify bash-tool and execution evidence. For legacy
pre-mini-SWE SFT checkpoints, direct GRPO/RLHF still accepts
`--harness drawing-tools`; its separate RLVR path is not supported. New SFT
checkpoints declare a `tool_contract` and incompatible stages are rejected.
Continuation must stay within the same reward stage and algorithm version.

The shared default input file, `datasets/parti-prompts/rl/prompts.jsonl`,
contains 327 reserved Parti RL prompts. The trainer **automatically partitions
it into two disjoint stages**: **227 RLVR prompts** and **100 RLHF prompts**.
Assignment is reproducible, stratified by upstream Category/Challenge, independent
of rollout RNG seeds, and applied **before** `--limit`. The selected prompt IDs,
source SHA-256, and split version are pinned in the training manifest and checked
on continuation. With CLI defaults of eight prompts and four rollouts per prompt,
one epoch takes **29 RLVR updates (454 rollouts with group size 2)** and
**13 RLHF updates (400 rollouts with group size 4)**. The maximum Docker
concurrency remains 32 episodes. Both stages support an explicit
`--group-size N` override.

`python3 scripts/prepare_parti_prompts.py` also writes inspectable
`datasets/parti-prompts/rl/rlvr.prompts.jsonl` and
`datasets/parti-prompts/rl/rlhf.prompts.jsonl`, with counts and hashes in
`datasets/parti-prompts/manifest.json`. Explicit **custom** `--dataset`
files are consumed as provided (not automatically repartitioned); the default
canonical source path activates the 227/100 split. The default initialization is
`checkpoints/sft-v2-qwen3.8-27b-final-preview-masked-cosine/final.json`;
pass `--init-checkpoint` for your own SFT run.

| Setting | Default |
| --- | --- |
| Prompt groups / trajectories per group | 8 / 4 |
| Concurrent Docker episodes | 32 |
| Learning rate / sampling temperature | `1e-5` / 1.0 |
| PPO ratio clipping / reference-KL coefficient | 0.8–1.2 / 0.01 |
| Epochs | 1 (41 updates, 1,308 episodes on 327 prompts; final batch has 7 prompts) |
| Model turns per trajectory | 12 for mini-SWE; 20 for legacy drawing-tools |
| Per-response / aggregate generated-token cap | No separate cap |
| Maximum context | 262,144 tokens |
| Checkpoint cadence | Every update |

Responses are clipped to remaining context; `--max-tokens N` adds an optional
per-response cap. RL containers have 2 GiB memory and two CPUs, a 60-second render
limit, and a 65-second drawing-operation timeout. Cookbook mean-centers rewards within groups
without standard-deviation scaling and removes constant-reward groups.
`--normalize-rewards` is a deprecated no-op.

GRPO uses a separately configured HPSv3 final-image preference scorer. A
successfully submitted, valid PNG receives the normalized preference reward
`sigmoid((hpsv3_score - center) / scale)`. Missing submissions, invalid renders,
refusals, and budget-limit terminations earn zero. The raw HPSv3 score and
normalized preference are saved per rollout. Set `ARTSY_HPSV3_MODEL` and
`ARTSY_HPSV3_URL`, with optional `ARTSY_HPSV3_CENTER` and `ARTSY_HPSV3_SCALE`,
before starting training. The judge is separate from the policy sampler.
See the [reward architecture](docs/rlhf-architecture.md).

Outputs include a training manifest, Cookbook checkpoints/metrics, grouped rollout
summaries, per-episode source/images/judgments/rewards, and `final.json`. Resume
checks dataset, source, initialization, and hyperparameter compatibility. To extend
a completed run in a new directory while preserving optimizer state and its SFT
reference, use `--continue-from` with `--additional-steps`; see
[continuation and checkpoint selection](docs/grpo-continuation.md). Evaluate
checkpoints separately with Inspect and human review.

## ArtCanvas v2 and local episodes

New drawings use the original [ArtCanvas v2 API](docs/artcanvas.md):
`from artcanvas import ArtCanvas`. The complete, one-page API reference is
included directly in the agent prompt, so a documentation-search tool call is
unnecessary. ArtCanvas wraps the installed Cairo bindings and creates a white
1920×1200 canvas by default. Use the context manager to write the relative
`output.png` file:

```python
from artcanvas import ArtCanvas

with ArtCanvas(1920, 1200, "output.png") as canvas:
    canvas.set_color(0, 0, 1)
    canvas.draw_circle(960, 600, 200, fill=True)
```

Python previews and final validation rerun the program in a fresh directory.
Absolute output paths and previously copied PNGs do not satisfy that contract.
The runtime provenance is `artcanvas-v2`; recent `pycairo-v1` trajectories
and checkpoints remain historical and must not be mixed with new ArtCanvas SFT
and GRPO runs.

Render an existing program without a model server or Docker:

```sh
make render ARGS='examples/transforms.py --prompt "Draw twelve golden shapes" --seed 42'
```

[`DrawingEnvironment`](drawing/environment.py) exposes async `reset`, `write_code`,
`run_code`, `submit`, and `refuse` operations for local episodes. Its defaults are
20 operations, a ten-minute episode deadline, a 60-second render timeout,
262,144 generated tokens, 64 KiB source, and an 8 MiB PNG. Adapters report token
usage through `charge_tokens`; the environment does not tokenize responses.
The default backend is the isolated Docker renderer. `--renderer local` selects
trusted host execution; `ARTSY_PYTHON` overrides its interpreter. Source/PNG
validation is shared by local execution, Docker execution, and evaluation.

Each reset creates a new episode directory with configuration, ordered trajectory,
and state. Successful submissions add source/image files and hashes; previews are
retained separately. Seeds reproduce Python random/hash ordering within a runtime,
but Cairo and font differences can affect pixels across hosts.

## Development checks

```sh
make check
# Individual suites: make test, make test-ui, make test-eval, make test-js

# Optional local Docker boundary checks; no hosted calls.
ARTSY_RENDERER_DOCKER_TEST=1 .venv/bin/python -m unittest tests.test_docker_renderer
```

`make` uses `.venv/bin/python` by default. Evaluation targets prefer `.eval-venv`
when present; override with `PYTHON` or `EVAL_PYTHON`. Pass CLI options through
`ARGS`. Tests exercise rendering, datasets, trajectory conversion, training plans,
judge parsing, Inspect scoring, and UI behavior using fixtures/mocks; no paid
model calls or running Docker daemon are required. Rendering tests need Cairo and
Python dependencies for the modules under test. The fake teacher CLI test binds
a loopback socket. The Docker boundary suite is skipped unless explicitly enabled.

`pyproject.toml` is the source of dependency pins and role membership. After
editing it, run `make requirements-sync` and `uv lock`; `make check` verifies the
generated requirements, compiles Python and JavaScript, and runs all four suites.
Evaluation and TTS retain their separate Python 3.12 environments.

Further reading: [pipeline diagrams](docs/pipeline-diagrams.md),
[benchmark and review](docs/benchmark.md), [RL continuation](docs/grpo-continuation.md),
and the [original Artsy post](https://blog.rashik.sh/tech/can-ai-learn-to-draw/).
