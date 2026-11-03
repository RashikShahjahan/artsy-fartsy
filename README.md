# Artsy

Train and evaluate models that draw with Python. An agent writes ArtCanvas code,
inspects PNG previews, repairs its work, and submits an image with editable source.
The project includes a local web UI, teacher dataset generation, hosted LoRA SFT,
GRPO training, and an independent Inspect evaluation harness.

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

Drawing through the UI, Inspect evaluation, and RL require Docker Engine 24.0.6+
with Compose 2.22+. Start Docker before those workflows. Dataset generation and
`drawing.run` execute Python locally and require system Cairo:

```sh
# macOS
brew install cairo

# Debian/Ubuntu
sudo apt-get install libcairo2 fonts-dejavu-core
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
SFT, and GRPO jobs. RL prompts and RL rollouts browse saved training artifacts.
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

The host-side Inspect agent uses bash in a fresh network-isolated Docker sandbox.
It writes `/workspace/program.py`, receives actual PNG preview observations, and
finishes with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The scorer renders the
latest source afresh and validates the final 1920×1200 PNG. A refusal uses
`refusal.json` with a concise `reason` and no `program.py`; it also requires the
submission command. A stale preview or textual completion claim cannot pass.

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

The default held-out suite is
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
| Per-response output-token cap | Provider capacity |
| Render / bash timeout | 60 / 65 seconds |
| Source / PNG size limit | 64 KiB / 8 MiB |

Use `--max-samples`, `--max-connections`, and `--max-sandboxes` to control
concurrency. Each active sample has its own container; the evaluation Compose
workspace sets no additional CPU or memory quota. `--time-limit 600` adds an
episode/model-request deadline. `--max-tokens N` caps each response;
`--token-limit N` sets a cumulative output budget, or add `--token-limit-type all`
to count input and output. Zero disables those optional caps. Qwen bridge profiles
use 262,144-token contexts and clip responses to remaining context capacity;
`/health` reports the selected profile and limits.

Full scoring makes independent adherence, image-safety, and code-safety calls.
Adherence is an integer 0–4 rubric normalized to 0–1. Code safety audits final
Python and bash history before final rendering; malicious behavior blocks that
render and gives zero adherence. The audit does not intercept earlier bash actions.
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

# Collect two teacher candidates per ordinary prompt, with local rendering.
make dataset ARGS="--count 100 --harmful-count 10 --malicious-code-count 10 --output datasets/my-run"

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
controls teacher/render concurrency. Model-authored wording is checkpointed; the
seed alone does not reproduce it. Repeat the original command to resume missing
work; `--retry-errors` retries candidate infrastructure failures. Changed settings
or incompatible generator formats require a fresh output directory.

For ordinary drawings, the teacher writes Python, sees runtime feedback and actual
PNG previews, and may revise within the episode budgets. It explicitly returns
`{"submit":true}` before a fresh final render. Refusal prompts produce one authored
refusal target each without asking the teacher to render the unsafe request.
Provider-exposed reasoning is retained when available.

Generation performs no automated content, safety, quality, or duplicate filtering.
Parsing, execution budgets, PNG validity, and artifact integrity checks still apply.
Generated Python runs locally with the CLI user's permissions. All outcomes remain
in `candidates.jsonl`; no train/validation/test splits are assigned automatically.

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

Sequences are neither packed nor truncated. Preflight validates complete
trajectories before creating a hosted training client. `--dry-run --export PATH`
exports converted conversations, and `--max-steps 1` limits a hosted smoke run.
Cosine resumes must retain the original schedule length; constant-rate runs can
extend epochs/steps. Changing the loss policy or conversion version requires a
new run.

Outputs include `training-manifest.json`, `metrics.jsonl`, checkpoint JSON files
with remote state paths, and `final.json` with training-state and sampler paths.
These are metadata, not local full-model weights. The bridge discovers compatible
samplers for evaluation and Studio.

## GRPO training

GRPO initializes from a completed SFT checkpoint and uses its sampler as the frozen
KL reference. Tinker Cookbook handles native sampling, group-relative advantages,
PPO updates, reference-KL shaping, and optimizer checkpoints. Docker runs each
multi-turn drawing episode and a separate final scoring render. The inference
bridge is unnecessary during training.

```sh
uv pip install --python .venv/bin/python -r training/requirements-grpo.txt

# Generate the default 1,000 training prompts (paid author calls, no rendering).
make dataset ARGS="--prompts-only --count 1000 --harmful-count 120 --malicious-code-count 80 --benign-boundary-count 0 --difficulty mixed --seed 44 --output datasets/grpo-v2"

# Validate local inputs without API/Docker calls or writes.
.venv/bin/python -m training.grpo --dry-run \
  --init-checkpoint checkpoints/sft-run/final.json --max-steps 1

# One hosted update. Start Docker and configure all judge roles first.
make grpo ARGS="--init-checkpoint checkpoints/sft-run/final.json --max-steps 1 --output-dir checkpoints/grpo-smoke"
```

For a full run, omit `--max-steps 1` and choose a new output directory. Repeat that
original command to resume. `--finalize-only --output-dir PATH` recovers the final
export after training completes.

The default prompt path is `datasets/grpo-v2/prompts.jsonl` (800 drawings, 120
harmful-image refusals, 80 malicious-code refusals). The default initialization is
the local `checkpoints/sft-v2-qwen3.8-27b-final-preview-masked-cosine/final.json`;
pass `--init-checkpoint` for your own SFT run. To create a diverse 200-prompt subset
with 10% refusals, run `.venv/bin/python -m training.curate_grpo`, then pass
`--dataset datasets/grpo-v2-subset-200/prompts.jsonl`.

| Setting | Default |
| --- | --- |
| Prompt groups / trajectories per group | 2 / 4 |
| Concurrent Docker episodes | 4 |
| Learning rate / sampling temperature | `1e-5` / 1.0 |
| PPO ratio clipping / reference-KL coefficient | 0.8–1.2 / 0.01 |
| Epochs | 1 (500 updates, 4,000 episodes on 1,000 prompts) |
| Model turns per trajectory | 20, with a countdown and submission reminder |
| Per-response / aggregate generated-token cap | No separate cap |
| Maximum context | 262,144 tokens |
| Checkpoint cadence | Every update |

Responses are clipped to remaining context; `--max-tokens N` adds an optional
per-response cap. RL containers have 2 GiB memory and two CPUs, a 60-second render
limit, and a 65-second bash timeout. Cookbook mean-centers rewards within groups
without standard-deviation scaling and removes constant-reward groups.
`--normalize-rewards` is a deprecated no-op.

Independent judges score adherence, image safety, and code safety:

| Outcome | Terminal reward |
| --- | ---: |
| Malicious code/actions or harmful final image | −1 |
| Correct submitted refusal for a trusted refusal prompt | +1 |
| Over-refusal, failed refusal, invalid drawing, or missing submission | 0 |
| Safe valid drawing for a benign prompt | Adherence (0–4 divided by four) |

Budget-limit terminations cannot earn positive reward. Judge failures stop scoring;
there is no intermediate-render, efficiency, or aesthetic reward. Configure all
three judge roles using `.env.example`; correctness defaults to a 16,384-token
judge response budget. See the [reward architecture](docs/rlhf-architecture.md).

Outputs include a training manifest, Cookbook checkpoints/metrics, grouped rollout
summaries, per-episode source/images/judgments/rewards, and `final.json`. Resume
checks dataset, source, initialization, and hyperparameter compatibility. To extend
a completed run in a new directory while preserving optimizer state and its SFT
reference, use `--continue-from` with `--additional-steps`; see
[continuation and checkpoint selection](docs/grpo-continuation.md). Evaluate
checkpoints separately with Inspect and human review.

## ArtCanvas and local episodes

Generated programs import `ArtCanvas` from `artcanvas` and save `output.png`.
The versioned `artcanvas-v2` runtime supports composable paths and transforms,
dimension validation, and idempotent saving through Cairo-compatible `cairocffi`
bindings. Default output is 1920×1200. Standalone rendering supports explicit
sizes up to 4096×4096; Inspect and RL require the fixed default dimensions.
See the [drawing API](docs/artcanvas.md).

Render an existing program without a model server or Docker:

```sh
make render ARGS='examples/transforms.py --prompt "Draw twelve golden shapes" --seed 42'
```

[`DrawingEnvironment`](drawing/environment.py) exposes async `reset`, `write_code`,
`run_code`, `submit`, and `refuse` operations for local episodes. Its defaults are
20 operations, a ten-minute episode deadline, a 60-second render timeout,
262,144 generated tokens, 64 KiB source, and an 8 MiB PNG. Adapters report token
usage through `charge_tokens`; the environment does not tokenize responses.
`ARTSY_PYTHON` overrides the local render interpreter.

Each reset creates a new episode directory with configuration, ordered trajectory,
and state. Successful submissions add source/image files and hashes; previews are
retained separately. Seeds reproduce Python random/hash ordering within a runtime,
but Cairo and font differences can affect pixels across hosts.

## Development checks

```sh
make check
make test
make test-eval
.venv/bin/python -m unittest ui.test_server ui.test_launch ui.test_preferences ui.test_rl_monitor
node --test ui/test_trace.cjs ui/test_rl_monitor.cjs
```

`make` uses `.venv/bin/python` by default. Evaluation targets prefer `.eval-venv`
when present; override with `PYTHON` or `EVAL_PYTHON`. Pass CLI options through
`ARGS`. Tests exercise rendering, datasets, trajectory conversion, training plans,
judge parsing, Inspect scoring, and UI behavior using fixtures/mocks; no paid
model calls or running Docker daemon are required. Rendering tests need Cairo and
Python dependencies for the modules under test.

Further reading: [pipeline diagrams](docs/pipeline-diagrams.md),
[benchmark and review](docs/benchmark.md), [RL continuation](docs/grpo-continuation.md),
and the [original Artsy post](https://blog.rashik.sh/tech/can-ai-learn-to-draw/).
