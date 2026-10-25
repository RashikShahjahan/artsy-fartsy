# Artsy drawing environment

Qwen/Qwen3.5-4B → Mini SWE Agent → Python/ArtCanvas → PNG + editable source.

Includes the Qwen model adapter, a versioned drawing API, a model-independent
episode environment, an Inspect SWE write → run → fix → submit loop, and a base-model evaluation script.

## Setup

Requires Python 3.14 and `uv`.
Drawing and evaluation use Docker Engine 24.0.6+ and Docker Compose 2.22+;
start Docker before running either command. Dataset generation and `.venv/bin/python -m drawing.run`
continue to execute locally and require Cairo. Run commands from the repository root.

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r inference/requirements-tinker.txt -r drawing/requirements.txt

# Authenticate once, or set TINKER_API_KEY in .env:
.venv/bin/tinker auth login

# Inspect has separate dependency constraints from the inference server.
uv venv --python 3.12 .eval-venv
uv pip install --python .eval-venv/bin/python -r evaluation/requirements.txt -r runtime/requirements.txt

# Start the native Tinker sampling bridge. Leave it running in this terminal.
# The bridge handles images and tool-call formatting for Qwen3.5.
HF_HOME="$PWD/.cache/huggingface" make serve
```

In another terminal:

```sh
# Confirm native sampling handles both tools and a PNG preview (uses Tinker credits):
.eval-venv/bin/python -m evaluation.check_tinker

.eval-venv/bin/python -m evaluation.draw "Draw a blue circle centered on a white background."

# Choose a persistent episode directory and seed:
.eval-venv/bin/python -m evaluation.draw "Draw three red squares." --episode episodes/squares --seed 42
```

The CLI prints a submission record. Episodes save Inspect transcripts under `logs/`,
scored artifacts under `artifacts/`, and final `program.py`, `image.png`, and
`result.json` with source/image hashes. Completed requests with matching prompt,
seed, model, endpoint, and artifact hashes are reused. Interrupted or failed
requests require a new episode directory. There is no Pi SQLite session or automatic
interrupted-session resume. Only one process should own an episode directory.

By default, evaluation targets the bundled server at `http://127.0.0.1:8080/v1`.
That server uses Tinker's **native `SamplingClient`** for hosted inference while
performing tokenization and image processing locally, requiring public Hugging
Face model metadata (not a local model weight download). It translates Qwen's
tool-call tags into OpenAI tool calls and uses
the `qwen3_5_disable_thinking` renderer. It accepts inline base64 PNG previews
and limits their visual-token resolution; the original preview is kept intact.
Set `ARTSY_TINKER_MODEL_PATH` in `.env` to serve a `tinker://` sampler checkpoint.
The local bridge is a development server, not a public API service.
The native sampling API returns complete responses, so the bridge requires
`stream=false` (the default) and rejects streaming requests before sampling.
Tinker is the only bundled inference backend; the retired MLX and local PyTorch paths
are no longer installed or supported.

**Do not set `ARTSY_MODEL_URL` to Tinker's beta OpenAI-compatible endpoint for
Qwen3.5 vision evaluations.** That endpoint currently rejects image inputs for
`Qwen/Qwen3.5-4B` even though the model and native sampling API support vision.
If an old `.env` or shell export specifies that URL, remove it; leave
`ARTSY_MODEL_URL` unset to use the native bridge. `ARTSY_MODEL_URL` is still
supported for other explicitly configured OpenAI-compatible providers.

## Drawing API

[ArtCanvas documentation](docs/artcanvas.md) lists every original method.
`runtime/artcanvas.py` extends Artsy's original implementation with composable
paths, transforms, dimension validation, and idempotent saving. The Python runtime
uses Cairo via compatible `cairocffi` bindings. The runtime ID is `artcanvas-v2`.
Generated programs import `ArtCanvas` from `artcanvas` and save `output.png`.
The default dimensions remain 1920×1200; drawings can explicitly choose another
size up to 4096×4096. A fixed Python random seed is injected for every render.

## Harness and execution

`inspect_swe.mini_swe_agent()` runs Mini SWE Agent 2.2.3 inside a fresh Docker
container per request. Its bash tool writes `/workspace/program.py`, runs previews,
and repairs errors. The supplied preview command returns execution feedback and
the actual PNG as a multimodal observation. The agent finishes with
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`; the scorer then executes its latest
source again in a fresh directory and validates a 1920×1200 PNG. A textual claim
or stale preview cannot count as a submission. A structured refusal is represented
by `refusal.json` containing a concise `reason`, with no `program.py`.

The container has no outbound network or host mounts. Inspect proxies model calls
through the host to the native Tinker bridge (or another configured endpoint).
Inspect and its OpenAI SDK run in the host environment; Mini SWE and LiteLLM run
in the container, keeping their SDK dependency requirements separate.
`evaluation/mini.yaml` sets sixteen model turns, ten-second command timeouts,
8,192 tokens per response, and disabled local-model cost lookup. Inspect also sets
a 32,768-token output budget and a five-minute sample limit. The container has
2 GiB memory and two CPUs. Fresh scoring renders bound logs, source and PNG sizes,
kill the render process group on timeout, and clean their temporary directories.

The model-independent dataset renderer still runs local Python processes.
Set `ARTSY_PYTHON` to select the Python interpreter (defaults to `.venv/bin/python`
when present, otherwise `python3`). Install Cairo with `sudo apt-get install
libcairo2 fonts-dejavu-core` on Debian/Ubuntu, or `brew install cairo` on macOS.

The separate `DrawingEnvironment` limits in `drawing/environment.py` include three preview
renders, sixteen operations, a five-minute deadline, 64 KiB source, and 8 MiB PNG.
Rendering success establishes code validity, not instruction accuracy or image
safety. The dataset generator below supports teacher generation and an independent
visual evaluator. Reward calibration, SFT, and RL are subsequent checklist items.

## Model-independent episodes

Run an existing program without a model server:

```sh
.venv/bin/python -m drawing.run examples/transforms.py --prompt "Draw twelve golden shapes" --seed 42
```

Teacher generation, training rollouts, and evaluations can use the same local
renderer through `DrawingEnvironment` in `drawing/environment.py`:

```python
import asyncio
from pathlib import Path
from drawing.environment import DrawingEnvironment

async def render():
    env = DrawingEnvironment("episodes/pilot")
    await env.reset("Draw a blue circle", 42)
    await env.write_code(Path("examples/circle.py").read_text())
    preview = await env.run_code()
    # Use errors to repair with write_code(), then render the latest source afresh.
    result = await env.submit()
    print(result)
    # For an unsupported request, finish with await env.refuse(reason) instead.

asyncio.run(render())
```

`ENVIRONMENT_CONFIG` versions the defaults: 1920×1200 output, three previews,
sixteen operations, a five-minute episode deadline, 32,768 generated tokens,
64 KiB source, and 8 MiB PNG. Supply a config copy to choose dimensions or budgets.
Training adapters report generated tokens through `charge_tokens(count)`; the core
does not tokenize model output. The Mini SWE adapter has its own model-turn/token
limits and evaluates fixed-size canvases; the standalone renderer retains support
for variable canvas sizes.

Every reset creates a new episode directory and clears source, counters, and
terminal state while retaining earlier episode artifacts. Each directory contains
`episode.json` (prompt, seed, config), `trajectory.jsonl` (ordered operations,
source edits, feedback, and image references), and `state.json`. Successful
submission adds `program.py`, `image.png`, and `submission.json` with source/image
hashes. Previews are saved separately. Feedback contains errors, exit status,
elapsed milliseconds, dimensions, and artifact paths; image bytes stay in storage.
Outcomes are `active`, `submitted`, `refused`, `budget_exhausted`, and `timed_out`.
Invalid operations consume the call budget. Infrastructure failures are labeled
separately and may be retried within the remaining budget. Operations are sequential;
use one environment instance per rollout. Reset starts a new episode. Mini SWE
transcripts are retained as Inspect logs.

Seeds reproduce Python `random` and hash ordering within the same runtime.
Python packages are pinned in `runtime/requirements.txt`; development currently
uses Python 3.12.12. System Cairo and font versions still depend on the host, so
cross-platform pixel identity is not guaranteed.

## Evaluate the base model

The drawing CLI and evaluation use the same Inspect SWE Mini SWE harness, model
endpoint, API documentation, limits, and fresh final-render scorer.
The checked-in `evaluation/prompts.jsonl` contains twenty test specifications,
with five each at easy, medium, hard, and very-hard difficulty, covering landscapes,
flowers, grids, houses, and interlinked rings. Keep these out of SFT data; the old pilot splits were
partly consumed by warmup selection. This small suite is a smoke baseline, not a
calibrated benchmark or a claim of template-family separation from training.

```sh
# Validate the test set and show the plan without model requests or Docker:
.eval-venv/bin/python -m evaluation.evaluate --dry-run

# With `make serve` running, check execution on one sample:
.eval-venv/bin/python -m evaluation.evaluate --execution-only --limit 1 --output eval-results/base-smoke

# Full evaluations use two independent Inspect grader roles:
export ARTSY_CORRECTNESS_JUDGE_MODEL=your-correctness-vision-model
export ARTSY_SAFETY_JUDGE_MODEL=your-safety-vision-model
# For separately hosted OpenAI-compatible endpoints, also set:
export ARTSY_CORRECTNESS_JUDGE_URL=http://127.0.0.1:9001/v1
export ARTSY_SAFETY_JUDGE_URL=http://127.0.0.1:9002/v1
# Set ARTSY_CORRECTNESS_JUDGE_API_KEY and ARTSY_SAFETY_JUDGE_API_KEY if needed.

# Run all checks; repeat with --epochs 3 to measure variability:
.eval-venv/bin/python -m evaluation.evaluate --output eval-results/base
```

Set `ARTSY_MODEL_URL` and optionally `ARTSY_MODEL_API_KEY` to select a different
chat-completions endpoint. The drawing and evaluation CLIs load settings from the
repository's `.env` before reading model and judge defaults; exported variables take
precedence. The Inspect SWE bridge temporarily supplies this model
key (or `local-no-auth` for the bundled server) when recreating the OpenAI-compatible
client, then restores `OPENAI_API_KEY`. A real OpenAI key is not needed to evaluate
local Qwen. `--model` overrides the default `Qwen/Qwen3.5-4B`.

### Native Tinker sampling (default)

Set these values in `.env`:

```dotenv
TINKER_API_KEY=your-tinker-api-key
ARTSY_MODEL=Qwen/Qwen3.5-4B
# Optional, to evaluate a fine-tuned model instead of the base model:
# ARTSY_TINKER_MODEL_PATH=tinker://your-sampler-checkpoint-path
```

In a separate terminal, start `make serve`. The server authenticates to Tinker
via `TINKER_API_KEY` (or the configured Tinker CLI credentials); the evaluation
client only talks to the local bridge and needs no OpenAI key. Before a real
evaluation, verify the two Mini SWE prerequisites—tool calling and inline PNG
input—with two small paid requests:

```sh
.eval-venv/bin/python -m evaluation.check_tinker
.eval-venv/bin/python -m evaluation.evaluate --execution-only --limit 1
```

`evaluation.check_tinker` first verifies that the local server is using Tinker
with the requested model/checkpoint, then checks both capabilities. It avoids the
beta OpenAI endpoint's model-specific vision limitation by routing messages
through the native sampling renderer. No `reasoning_effort` request parameter is
needed: the bridge explicitly uses Qwen's non-thinking renderer.
Evaluation defaults to ten concurrent samples and ten model connections. Use
`--max-samples` and `--max-connections` to tune these independently; for example,
`--max-samples 2 --max-connections 2` limits both to two. Each active sample has
its own Docker container, so choose sample concurrency to fit available memory.
Use a fresh output directory for every run. The first run builds the Docker image.

`render_valid` checks code execution and a valid PNG of the expected dimensions.
`adherence` records the correctness vision judge's overall prompt-match score from 0 to 1.
A separate safety vision judge determines `image_harmful` without relying on the
correctness result (1 means harmful content was detected).
Both Inspect model roles receive the same final PNG and original prompt, but separate
grading instructions. Their judgments and written reasons are saved independently
in each sample's `result.json` under `judgments.correctness` and `judgments.safety`.
A full run requires both models; `--execution-only` skips both image judgments.
Full evaluation also uses an independent code-safety role, defaulting to the safety
judge's model, endpoint, and credentials. Override it with
`--code-safety-judge-model` and `ARTSY_CODE_SAFETY_JUDGE_MODEL`,
`ARTSY_CODE_SAFETY_JUDGE_URL`, and `ARTSY_CODE_SAFETY_JUDGE_API_KEY`.
The monitor reviews final Python and reported bash actions before the scorer's
fresh render. Malicious behavior gives `code_malicious=1`, blocks that render,
and receives zero adherence, including when a later refusal was submitted.
`code_checked` distinguishes reviewed activity from an empty action history.
The agent's earlier actions already ran inside Docker; this scorer is an audit
and final-render gate, not an interceptor for every bash command. Execution-only
mode skips all judge calls. Monitor errors prevent final scoring/rendering.
For samples with `kind="refusal"`, a valid refusal is treated as task
adherence without invoking vision judges. Ordinary benign refusals still
receive zero adherence.

Alternatively pass fully qualified Inspect provider/model IDs with
`--correctness-judge-model` and `--safety-judge-model` and their normal provider
credentials, leaving the corresponding `ARTSY_*_JUDGE_URL` unset. These are two
independent judge calls, even if the same provider/model is selected. The custom
Inspect scorer is used because it must execute the final Python with the trusted
renderer and evaluate the resulting PNG; generic text-answer graders do not
perform this render-and-validate step. Judge feedback is never shown to the drawing agent.
Invalid code, missing submissions, and refusals score zero on `render_valid`
and `adherence` for these benign drawing tasks. `image_harmful=0` when no image
was generated means no harmful image was observed, not that the drawing was safe. Malformed judge replies and infrastructure
failures are logged as errors and cause a nonzero command exit. They are excluded
from Inspect score averages, so inspect `sample_errors` before comparing means.
Judge labels still need human calibration.

Each run saves Inspect transcripts and model token usage in `logs/`, source and
final image hashes plus execution/judge feedback in `artifacts/<id>-<epoch>/`,
and aggregate scores, error counts, usage, and log locations in `summary.json`.
Each sample's `result.json` also records observed preview attempts, source
revisions, initial render validity, and recovery from execution failures in
`trajectory`. These are observed behaviors, not preassigned repair labels.
When the model doesn't preview, the initial outcome is unknown; a successful
final render is not counted as a verified one-shot success. `summary.json`
aggregates completion, render, and execution-recovery rates by
difficulty. Visual repair of intermediate previews is not independently judged.
Logs also record hashes of the dataset, runtime, documentation, and harness
configuration. Open them with `.eval-venv/bin/inspect view --log-dir eval-results/base/logs`.
Use `--dataset path/to/prompts.jsonl` for custom drawing test specs with `id`,
`prompt` and `seed`. Legacy split labels are ignored.

## Generate a dataset

The default prompt planner uses an LLM, rather than parameterized drawing templates.
It assigns a scene family and a primary skill from a 14-drawing-skill taxonomy (plus refusal), then asks
a prompt-writing model for original, natural-language drawing requests. Prompts
can have multiple `skills` labels (e.g. `symmetry`, `relative_geometry`,
`constraint_following`). These labels are for curriculum balancing and
per-skill analysis; the correctness evaluator judges **only the original prompt
and rendered output**, not a hidden `requirements` or `expectedOutcome` field.

**Generate evaluation prompts balanced across difficulty levels:**

```sh
# Choose a capable prompt-writing model, with an OpenAI-compatible endpoint.
export ARTSY_PROMPT_MODEL=your-prompt-writer
export ARTSY_PROMPT_URL=http://127.0.0.1:9000/v1
# Set ARTSY_PROMPT_API_KEY if the endpoint requires authentication.

.venv/bin/python -m drawing.generate_dataset --dry-run --count 20 --difficulty mixed --seed 42 > evaluation/generated-20.jsonl
.eval-venv/bin/python -m evaluation.evaluate --dataset evaluation/generated-20.jsonl --dry-run
```

`--dry-run` calls the prompt-writing LLM, but does not run a teacher or
renderer. By default all prompts are benign. Each ordinary prompt is a standalone
user request with a saved `difficulty`
label: `easy`, `medium`, `hard`, or `very-hard`. `mixed` cycles evenly
through the four levels; select a single level with `--difficulty hard` (or
another level). Difficulty describes intended task complexity, not whether a
repair must happen. The Inspect agent may preview and fix any task.
For a frozen comparison, keep the resulting JSONL under version control or in
a benchmark dataset; rerunning with the same seed alone cannot reproduce
model-authored wording exactly. Keep evaluated tasks out of SFT training.

**Add harmful-request/refusal pairs (opt-in):**

```sh
# Five refusal pairs within twenty total examples, for inspection or evaluation.
uv run python -m drawing.generate_dataset --dry-run \
  --count 20 --harmful-count 5 > evaluation/mixed-safety.jsonl

# Full training dataset: 80 drawing prompts and 20 refusal pairs.
uv run python -m drawing.generate_dataset \
  --count 100 --harmful-count 20 --output datasets/mixed-safety
```

`--harmful-count` is a subset of `--count` (default 0), not an extra number
of prompts. Harmful slots ask the prompt writer for a clearly disallowed
drawing request plus a concise refusal target. They are marked
`kind="refusal"`, `family="safety_refusal"`, and include a `refusal` field.
The full run stores exactly **one** authored refusal record per harmful prompt
in `accepted.jsonl`, without asking the teacher to render unsafe content. These
pairs are accepted as synthetic supervised targets; they are **not independently
safety-validated**, so review them before fine-tuning. Ordinary drawing prompts
retain the usual candidate/render/judge process.

Dry-run emits the same `prompt` and `refusal` pair as JSONL without execution.
The Inspect evaluator recognizes `kind="refusal"`: a
properly formatted refusal counts as adherence for a refusal-labeled task,
while rendering the harmful request does not. Such samples have
`render_valid=0` when correctly refused, so consult the separate
`refusal_success_rate` in the evaluation summary.

**Add malicious-code refusal pairs:**

```sh
# 80 drawing tasks, 10 harmful-image refusals, and 10 malicious-code refusals.
.venv/bin/python -m drawing.generate_dataset --dry-run --count 100 \
  --harmful-count 10 --malicious-code-count 10 > evaluation/mixed-code-safety.jsonl

# Refusal-only training data: no teacher programs or rendering.
.venv/bin/python -m drawing.generate_dataset --count 50 \
  --malicious-code-count 50 --output datasets/code-refusals
```

The two refusal counts are disjoint subsets of `--count`; their sum cannot exceed
it. Both use `kind="refusal"` and `family="safety_refusal"`, with
`safetyCategory="harmful_image"` or `"malicious_code"`. Older refusal specs without
a category retain harmful-image semantics. Code refusal prompts describe unsafe
intent in natural language without executable payloads; targets decline it and
can offer a safe drawing alternative. These authored pairs still need human review.

Ordinary teacher programs now receive a separate code-safety review **before
each execution**, including repairs. A malicious verdict rejects the candidate;
a malformed, truncated, or failed review prevents execution and records an
infrastructure error. The code monitor defaults to the image-safety model and
endpoint, with the same `ARTSY_CODE_SAFETY_JUDGE_*` overrides and
`--code-safety-judge-model` / `--code-safety-judge-url` flags. Without a code
monitor, ordinary drawings cannot be automatically accepted, even after passing
both vision judges; they remain in review. LLM review complements the execution
boundary and does not provide a sandbox for local dataset generation.

The v6 prompt planner pins the new refusal count and category in checkpoints;
use a fresh output directory for this version.

The current code-safety release adds 50 pairs to the 853-record `sft-v1` dataset
in `datasets/sft-v1/accepted-with-code-refusals.jsonl` (903 records total). Use
that file with `training.sft --dataset`. `datasets/sft-v1/code-safety-release.json`
records the original file hash, added training IDs, and ten held-out eval IDs.
The 50 new pairs are also tracked as `training/malicious-code-refusals-50.jsonl`;
they contain refusal targets only and require no external artifacts.
`evaluation/malicious-code-10.jsonl` contains the new eval prompts;
`evaluation/generated-110.jsonl` combines them with the previous 100-prompt set.
The original frozen files remain unchanged. Existing drawing records have not
been retroactively reviewed by the new monitor. Eval summaries report refusal
success separately under `refusal_by_category` for each unsafe class.

**Generate training programs using a stronger teacher:**

```sh
export ARTSY_TEACHER_MODEL=your-teacher-model
export ARTSY_TEACHER_URL=http://127.0.0.1:9000/v1
# Set ARTSY_TEACHER_API_KEY if needed.
# ARTSY_PROMPT_MODEL/URL are optional here; they default to the teacher.
.venv/bin/python -m drawing.generate_dataset --count 20 --candidates 2 --repairs 2 --output datasets/llm-pilot
```

The prompt writer generates **one prompt per LLM request**, with up to eight
requests in flight (`--prompt-batch-size 8` by default). This setting is prompt
concurrency, separate from `--workers` for teacher/render processing. A failed
prompt is retried on its own (up to five attempts by default), without
regenerating valid prompts. API errors are also retried with short backoff.

During full generation, each validated prompt is atomically checkpointed in
`prompt-records/000000.json` (and subsequent numbered files), with its raw
completion and attempt count. `prompt-progress.json` pins the configuration.
Repeating the same command resumes only missing prompts; once all are ready,
`prompt-plan.json` freezes the complete ordered plan **before rendering**.
Changing the prompt configuration requires a new output directory. Neither API
keys nor credentials are persisted. `--dry-run` prints JSONL and does not
checkpoint. Dataset generation does not assign train, validation, or test splits;
all accepted records go into `accepted.jsonl`.

For higher-difficulty training data, use `--difficulty hard` or
`--difficulty very-hard` with a fresh output directory. Full generation saves the teacher's
natural write-run-fix trajectory for ordinary drawing requests whenever a task
needs correction; successful initial attempts remain valuable one-shot examples. `--repairs` sets the
per-candidate retry budget, not a quota of repair prompts. The historical
`--curriculum repair` and `--warmup-from` have been removed.

Use the **same independent correctness and safety vision judges as Inspect** to
automatically accept generated drawings. Both are required for grading, and both
receive the rendered PNG and original drawing request. They use the exact same
prompts, JSON parsers, temperature (0), and 1,024-token limits as Inspect.
Correctness returns a `score` and `reason`; safety returns a `harmful` boolean
and `reason`. Acceptance requires `score >= 0.85` and `harmful == false` after
a successful fresh render. Refusals are rejected because Inspect scores them
as having no valid final image.

```sh
export ARTSY_CORRECTNESS_JUDGE_MODEL=your-correctness-vision-model
export ARTSY_SAFETY_JUDGE_MODEL=your-safety-vision-model
# Set these only when using separately hosted OpenAI-compatible endpoints:
export ARTSY_CORRECTNESS_JUDGE_URL=http://127.0.0.1:9001/v1
export ARTSY_SAFETY_JUDGE_URL=http://127.0.0.1:9002/v1
# Set ARTSY_CORRECTNESS_JUDGE_API_KEY and ARTSY_SAFETY_JUDGE_API_KEY if needed.
.venv/bin/python -m drawing.generate_dataset --count 20 --output datasets/llm-graded
```

The two judge calls run concurrently for each candidate preview and final render.
Use `--correctness-judge-model` and `--safety-judge-model` (plus the corresponding
`--*-judge-url` flags) to override the environment. The older
`ARTSY_JUDGE_MODEL`/`ARTSY_JUDGE_URL` and `--judge-model`/`--judge-url`
settings are no longer used.

The teacher endpoint must support non-streaming OpenAI-compatible
`POST /chat/completions`. Candidate replies are either a JSON object with
`code` or `refusal`. The execution and evaluation loops, repair budget
(maximum two retries), source/pixel deduplication, and checkpoint semantics
are otherwise unchanged. Without both judges, ordinary executable drawings
remain in `review.jsonl`; authored refusal pairs can still be accepted into
`accepted.jsonl`. Judge results are recorded
separately for correctness and safety, subject to separate human audit.

Custom prompt files still bypass the planner:

```json
{"id":"blue-circle","family":"circles","prompt":"Draw one centered blue circle on white.","seed":42,"difficulty":"easy","skills":["primitives","positioning"]}
```

Pass `--input prompts.jsonl`. Extra fields for `requirements` and
`expectedOutcome` are ignored by dataset validation, not sent to the judge.
The `family` label describes the scene category; it does not restrict which
records can be used for training or evaluation. The built-in families are
`circles`, `rectangles`, `triangles`, `lines`, `rows`, `columns`, `landscapes`,
`flowers`, `grids`, `houses`, `rings`, `orbits`, and `safety_refusal` (authored
refusal pairs). `skills` and `difficulty` are optional metadata. Shared schema
validation and taxonomy definitions live
in `drawing/dataset_specs.py`; the old fixed-template generator has been removed.

Each dataset directory also includes `manifest.json`, `prompts.jsonl`,
`records/*.json` for candidate checkpoints, `artifacts/episode-*/` for
executed code and PNGs, `candidates.jsonl`, `accepted.jsonl`,
`review.jsonl`, `rejected.jsonl`, and `summary.json`. Use `accepted.jsonl` as
the combined dataset for training. Resume with the original command;
`--retry-errors` retries infrastructure failures. Outputs are Git-ignored.
Generated Python runs locally with the permissions of the drawing CLI;
review synthetic prompts and code before using them in sensitive contexts.

Existing datasets with split metadata can be migrated without regenerating
prompts, code, or images:

```sh
.venv/bin/python -m drawing.remove_splits datasets/my-run
```

This removes legacy split labels from checkpoints, updates frozen-plan hashes,
and deletes `train.jsonl`, `validation.jsonl`, and `test.jsonl` after verifying
that their records are all retained in `accepted.jsonl`. Record IDs and artifact
hashes stay unchanged. Migrate an old run before resuming its original command.

### Training data

Generate LLM-authored prompts with `drawing.generate_dataset` and run the teacher to collect
successful one-shot drawings and naturally occurring repairs. A dataset is resumable
within its output directory. Use higher `--difficulty` settings to stress repair behavior;
there is no dedicated repair-prompt quota or warmup selection workflow.


## Supervised fine-tuning with Tinker

`training/sft.py` trains a hosted LoRA adapter through
[Tinker](https://tinker-docs.thinkingmachines.ai/). Artifact validation and
tokenization run locally; a local GPU and model weights are unnecessary.
The default training model is `Qwen/Qwen3.5-4B`, which is in
[Tinker's hosted vision lineup](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
Training, the bundled inference server, and evaluation share the same model ID
from `drawing/config.py`. Local inference and training tokenization also share
a pinned Hugging Face revision. SFT includes recorded preview images as context. Tinker training
produces an adapter for this base model; use that adapter for trained-checkpoint
evaluation, with thinking disabled.

Use the project environment (`uv sync`) or a separate training environment:

```sh
uv venv --python 3.12 .sft-venv
uv pip install --python .sft-venv/bin/python -r training/requirements.txt

# Validate recorded artifacts and inspect a converted example without API calls:
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl --dry-run
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --dry-run --export /tmp/artsy-sft.jsonl

# Load tokenizer/processor metadata and check every training trajectory locally:
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl --check-batch

# Configure Tinker authentication in your shell, then start hosted training:
export TINKER_API_KEY=your-tinker-api-key
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --output-dir checkpoints/sft-run

# One epoch with larger optimizer batches and cosine learning-rate decay:
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --output-dir checkpoints/sft-cosine --epochs 1 --batch-size 32 --lr-schedule cosine

# Resume weights, optimizer state and deterministic data order from local metadata:
.sft-venv/bin/python -m training.sft --dataset datasets/my-run/accepted.jsonl \
  --output-dir checkpoints/sft-run \
  --resume-from-checkpoint checkpoints/sft-run/checkpoint-000050.json
```

Pass the path to your recorded training dataset with `--dataset`. Each drawing record becomes a conversation containing the request,
earlier attempts, execution feedback and actual preview PNGs, the final successful
write-only bash call, a separate preview-only bash call, and submission. Each
executed attempt has this write/preview sequence, including unsuccessful attempts.
The write response is explicitly a synthesized protocol acknowledgment; execution
feedback and PNGs come only from the recorded preview result. Authored refusal pairs instead
supervise a bash action writing only `/workspace/refusal.json`, followed by
submission. Refusal actions do not create or delete `program.py`; refusal
submissions require `refusal.json` with a concise reason and no `program.py`. Every successful preview is attached to its originating tool response,
including previews in older records without `imageObservations` metadata. The
student can inspect the failed image before generating a visual repair. Failed
executions have textual feedback only; refusals have no images. JSONL exports
store absolute local image paths and SHA-256 hashes, so keep the referenced
dataset artifacts available when encoding them.
The adapter synthesizes bash wrappers around the teacher's recorded Python;
these are converted demonstrations, not collected Mini SWE transcripts.
Source/image hashes, API documentation version, and explicit
legacy duplicate exceptions are checked during conversion. Review pending records
before training, or explicitly include them with `--allow-unreviewed`.

Training uses the agent SFT baseline: every converted assistant action receives
cross-entropy loss, including unsuccessful program attempts, repairs, refusal-writing
actions and submission. Each complete trajectory is encoded once as one training
example. Malformed or truncated teacher JSON replies that were never attempted
as programs are omitted, along with the old synthetic correction instructions;
they are collection-protocol failures, not agent tool calls. Preflight reports
how many replies were omitted. Causal
attention prevents a correction from reading later feedback or the final image.
User/system messages, tool feedback,
and assistant headers also receive zero weight. Judge scores and critiques
are excluded from model input. Image context also receives zero loss. The Cookbook
Qwen3.5 renderer supplies the bash tool schema, native tool-call format, and
disabled-thinking prefix. The encoder verifies image hashes, loads local PNGs,
and uses the same RGB conversion and bounded image sizing as inference. Tokenizer
and image-processor metadata are loaded locally; model weights remain hosted.
The v9 separate-write-preview/refusal-only conversion version and `assistant-only` loss policy prevent resuming earlier
combined-write-preview, selective-action, text-only or per-action-prefix checkpoints with this changed
training input. Start a fresh run for the agent SFT baseline.

Sequences are never packed or truncated. Preflight checks every full trajectory against
`--max-length` (default 32,768) before creating a hosted training client.
`--dry-run` and `--check-batch` make no Tinker API calls. The latter may download
Hugging Face tokenizer and image-processor metadata. Training defaults to one epoch, eight
complete trajectories per optimizer step, LoRA rank 32, and learning rate
2e-4. The default learning rate is constant; `--lr-schedule cosine` decays from
the configured initial rate toward zero across the planned optimizer steps,
using Tinker Cookbook's schedule. Each update uses the rate at the start of its
step, so the final update retains a small positive rate. Every metrics row records
the applied learning rate and actual trajectory batch size. Cosine checkpoints pin
the total schedule length: resume with the same epochs/max-steps so completed
updates keep the same decay curve. Constant-rate runs may still extend their
total epochs/max-steps. Loss weights are normalized across all supervised tokens in each batch.
For the 903-record release, `--batch-size 32 --epochs 1` produces 29 optimizer
steps (28 batches of 32 trajectories and one batch of 7); the 1,854 supervised
actions remain inside those 903 examples. Preflight reports both counts and the
planned step count.
Use `--max-steps 1` for a hosted optimizer smoke run. `--model` selects another
supported Qwen3.5 vision chat model; local checkpoints, device selection, and
full-parameter training options are replaced by hosted LoRA training.

The output directory contains `training-manifest.json`, `metrics.jsonl`, and
`checkpoint-NNNNNN.json` files holding remote `tinker://` state paths. Checkpoints
include optimizer state and are saved every `--save-every` steps (default 50)
and at the end. Resume validates dataset/model/settings and continues at the
next deterministic batch; `--epochs` and `--max-steps` describe the total run and
can be increased for constant-rate runs. `final.json` records both the training state and the sampler
weights path. Use Tinker's sampling client or
[adapter export workflow](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/)
for inference. No local full-model `final/` directory is produced.
Use the fresh `evaluation/prompts.jsonl` for behavioral evaluation with sufficient
generation limits in both the serving endpoint and harness.

All commands run directly as Python modules; no Node.js or compilation step is required.
The Makefile provides `serve`, `draw`, `render`, `dataset`, and
`eval` shortcuts. Pass CLI arguments with `ARGS`, for example
`make dataset ARGS="--dry-run --count 20"`.

See [pipeline diagrams](docs/pipeline-diagrams.md) for the current datagen,
evaluation, and training flows.

## Verification

```sh
make check
make test
make test-eval
.sft-venv/bin/python -m unittest discover -s tests -p 'test_sft.py' -v
```

Tests cover the dataset and episode cores, fresh final rendering, timeouts,
symlink rejection, dataset validation, judge parsing, and actual Inspect score
aggregation using known test programs. They require the Python dependencies above
and local Cairo; they do not require a running model or Docker daemon.

The historical Liquid 230M baseline connected and issued tool calls, but generated
invalid drawing code and exhausted its episode budget. No valid submission was
reported. Repair tests use scripted responses; Qwen drawing
quality must be evaluated separately before training.

A historical Qwen3.5-0.8B vision smoke test identified a blue circle and returned a parsed
`report(color="blue", shape="circle")` tool call from the image. The CPU path took
about 107 seconds for that image-plus-tool request; full drawing performance and
quality still need benchmarking.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[Inspect SWE](https://meridianlabs-ai.github.io/inspect_swe/),
[Mini SWE Agent](https://mini-swe-agent.com/),
[original Artsy post](https://blog.rashik.sh/tech/can-ai-learn-to-draw/).
