# Artsy drawing environment

Qwen/Qwen3.5-0.8B → Mini SWE Agent → Python/ArtCanvas → PNG + editable source.

Includes the Qwen model adapter, a versioned drawing API, a model-independent
episode environment, an Inspect SWE write → run → fix → submit loop, and a base-model evaluation script.

## Setup

Requires Python 3.12 and `uv`.
Drawing and evaluation use Docker Engine 24.0.6+ and Docker Compose 2.22+;
start Docker before running either command. Dataset generation and `.venv/bin/python -m drawing.run`
continue to execute locally and require Cairo. Run commands from the repository root.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r inference/requirements.lock -r drawing/requirements.txt

# Inspect has separate dependency constraints from the inference server.
uv venv --python 3.12 .eval-venv
uv pip install --python .eval-venv/bin/python -r evaluation/requirements.txt -r runtime/requirements.txt

# Starts a loopback-only server; the first start downloads Qwen3.5-0.8B.
HF_HOME="$PWD/.cache/huggingface" make serve
```

In another terminal:

```sh
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

`ARTSY_MODEL_URL` can point to another OpenAI-compatible serving endpoint, using
the same model identifier and tool-call support. The bundled CPU server translates
Qwen's function/parameter tool-call tags into the OpenAI tool protocol. Tool
arguments are decoded as data; the bridge never evaluates generated expressions.
It loads the full vision-language model at a pinned revision, with thinking
disabled. Image inputs use OpenAI `image_url` blocks containing inline base64
image data URLs. Images are decoded with Pillow and resized to at most 1,024
visual tokens (about one megapixel); original preview files retain their resolution.
Requests are serialized and capped at 2,048 generated tokens and a
32,768-token context to bound CPU inference, below Qwen's native context capacity.
This is a small development server, not a public API service.

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
through the host, so `ARTSY_MODEL_URL` can keep using its loopback address.
Inspect and its OpenAI SDK run in the host environment; Mini SWE and LiteLLM run
in the container, keeping their SDK dependency requirements separate.
`evaluation/mini.yaml` sets sixteen model turns, ten-second command timeouts,
2,048 tokens per response, and disabled local-model cost lookup. Inspect also sets
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
The checked-in `evaluation/prompts.jsonl` contains eight new test specifications
covering overlap, repeated layouts, dashes, rounded shapes, gradients, curves,
transforms, and sectors. Keep these out of SFT data; the old pilot splits were
partly consumed by warmup selection. This small suite is a smoke baseline, not a
calibrated benchmark or a claim of template-family separation from training.

```sh
# Validate the test set and show the plan without model requests or Docker:
.eval-venv/bin/python -m evaluation.evaluate --dry-run

# Start the base-model server with make serve, then check execution on one sample:
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
chat-completions endpoint. `--model` overrides the default `Qwen/Qwen3.5-0.8B`.
Runs use one sample at a time to match the bundled serialized CPU server.
Use a fresh output directory for every run. The first run builds the Docker image.

`render_valid` checks code execution and a valid PNG of the expected dimensions.
`image_correct` uses a dedicated correctness vision judge to assess the entire original
prompt, requiring an overall prompt-match score of at least 0.85 (`--threshold` overrides it).
A separate safety vision judge determines `image_harmful` without relying on the
correctness result. `visual_pass` requires valid execution, a correct image, and no
flagged harm. `adherence` records the correctness judge's overall prompt-match score.
Both Inspect model roles receive the same final PNG and original prompt, but separate
grading instructions. Their judgments and written reasons are saved independently
in each sample's `result.json` under `judgments.correctness` and `judgments.safety`.
A full run requires both models; `--execution-only` skips both image judgments.

Alternatively pass fully qualified Inspect provider/model IDs with
`--correctness-judge-model` and `--safety-judge-model` and their normal provider
credentials, leaving the corresponding `ARTSY_*_JUDGE_URL` unset. These are two
independent judge calls, even if the same provider/model is selected. The custom
Inspect scorer is used because it must execute the final Python with the trusted
renderer and evaluate the resulting PNG; generic text-answer graders do not
perform this render-and-validate step. Judge feedback is never shown to the drawing agent.
Invalid code, missing submissions, and refusals
score zero on correctness and overall success for these benign drawing tasks.
`image_harmful=0` for a request with no generated image means no harmful image was
observed; it does not grant an overall pass. Malformed judge replies and infrastructure
failures are logged as errors and cause a nonzero command exit. They are excluded
from Inspect score averages, so inspect `sample_errors` before comparing means.
Judge labels still need human calibration.

Each run saves Inspect transcripts and model token usage in `logs/`, source and
final image hashes plus execution/judge feedback in `artifacts/<id>-<epoch>/`,
and aggregate scores, error counts, usage, and log locations in `summary.json`.
Logs also record hashes of the dataset, runtime, documentation, and harness
configuration. Open them with `.eval-venv/bin/inspect view --log-dir eval-results/base/logs`.
Use `--dataset path/to/prompts.jsonl` for custom drawing test specs with `id`,
`prompt` and `seed`. Explicit splits must be `test`.

## Generate a dataset

The default prompt planner uses an LLM, rather than parameterized drawing templates.
It assigns a scene family and a primary skill from a 14-label taxonomy, then asks
a prompt-writing model for original, natural-language drawing requests. Prompts
can have multiple `skills` labels (e.g. `symmetry`, `relative_geometry`,
`constraint_following`). These labels are for curriculum balancing and
per-skill analysis; the correctness evaluator judges **only the original prompt
and rendered output**, not a hidden `requirements` or `expectedOutcome` field.

**Generate evaluation prompts with explicitly chosen counts (15 standard, 5 repair-oriented):**

```sh
# Choose a capable prompt-writing model, with an OpenAI-compatible endpoint.
export ARTSY_PROMPT_MODEL=your-prompt-writer
export ARTSY_PROMPT_URL=http://127.0.0.1:9000/v1
# Set ARTSY_PROMPT_API_KEY if the endpoint requires authentication.

.venv/bin/python -m drawing.generate_dataset --dry-run --count 15 --repair-count 5 --seed 42 > evaluation/generated-20.jsonl
.eval-venv/bin/python -m evaluation.evaluate --dataset evaluation/generated-20.jsonl --dry-run
```

`--dry-run` now **calls the prompt-writing LLM**, but does not run a teacher or
renderer. Each prompt is a standalone user request; the five `caseType=repair`
cases are complex geometry/composition challenges designed to stress the
write → run → fix loop. They do not contain deliberately broken code. Inspect
checks the actual trajectory and final PNG, and ignores the metadata labels.
For a frozen comparison, keep the resulting JSONL under version control or in
a benchmark dataset; rerunning with the same seed alone cannot reproduce
model-authored wording exactly. Keep evaluated tasks out of SFT training.

**Generate training programs using a stronger teacher:**

```sh
export ARTSY_TEACHER_MODEL=your-teacher-model
export ARTSY_TEACHER_URL=http://127.0.0.1:9000/v1
# Set ARTSY_TEACHER_API_KEY if needed.
# ARTSY_PROMPT_MODEL/URL are optional here; they default to the teacher.
.venv/bin/python -m drawing.generate_dataset --count 20 --candidates 2 --repairs 2 --output datasets/llm-pilot
```

The output directory stores `prompt-plan.json` **before rendering**. This
records the validated prompts, generator model/endpoint, taxonomy-slot settings,
raw prompt-generation completions and token usage. Resuming the same command
reuses the exact plan with no additional prompt-writing calls; changing its
configuration requires a new output directory. Neither API keys nor credentials
are persisted. For prompts only, `--dry-run` prints JSONL without split labels.
For complete dataset runs, existing family-based training splits remain internal.

No fixed standard/repair proportion is imposed. `--count 20` makes only
standard prompts; `--count 0 --repair-count 20` makes only repair-oriented
prompts, and `--count 15 --repair-count 5` makes a specified mix. For a wholly challenging
**natural repair** curriculum, use:

```sh
.venv/bin/python -m drawing.generate_dataset --curriculum repair --pool-size 32 --output datasets/llm-repair
```

This generates advanced, computationally demanding tasks with
`repairTask.skills` metadata. The repair pipeline accepts examples only after
a genuine execution or visual failure followed by successful correction.
The separate `--warmup-from` command still uses its frozen legacy repair
pool to reproduce the existing 80/20 warmup selection.

Set an independent **vision** judge to automatically accept generated drawings.
It must handle PNG `image_url` messages:

```sh
export ARTSY_JUDGE_MODEL=your-vision-evaluator
export ARTSY_JUDGE_URL=http://127.0.0.1:9001/v1
# Set ARTSY_JUDGE_API_KEY if needed.
.venv/bin/python -m drawing.generate_dataset --count 20 --output datasets/llm-graded
```

The teacher endpoint must support non-streaming OpenAI-compatible
`POST /chat/completions`. Candidate replies are either a JSON object with
`code` or `refusal`. The execution and evaluation loops, repair budget
(maximum two retries), source/pixel deduplication, and checkpoint semantics
are unchanged. Without a judge, valid programs remain in `review.jsonl`;
`accepted.jsonl` is empty. The judge scores PNG correctness, content safety,
and refusal appropriateness against the original request; a score of at least
0.85 is needed for automatic acceptance, subject to separate human audit.

Custom prompt files still bypass the planner:

```json
{"id":"blue-circle","family":"circles","prompt":"Draw one centered blue circle on white.","seed":42,"skills":["primitives","positioning"]}
```

Pass `--input prompts.jsonl`. Extra fields for `requirements` and
`expectedOutcome` are ignored by dataset validation, not sent to the judge.
The `family` label is used for group-level splits; `skills` and `caseType`
are optional metadata. Original fixed-template generation is retained as a
legacy Python helper for compatibility tests, not used by the default CLI.

Each dataset directory also includes `manifest.json`, `prompts.jsonl`,
`records/*.json` for candidate checkpoints, `artifacts/episode-*/` for
executed code and PNGs, `candidates.jsonl`, `accepted.jsonl`,
`review.jsonl`, `rejected.jsonl`, `train.jsonl`, `validation.jsonl`,
`test.jsonl`, and `summary.json`. Resume with the original command;
`--retry-errors` retries infrastructure failures. Outputs are Git-ignored.
Generated Python runs locally with the permissions of the drawing CLI;
review synthetic prompts and code before using them in sensitive contexts.

### First SFT dataset: 80 one-shot drawings and 20 repairs

Build the first 100-example selection from the existing `datasets/pilot-100`:

```sh
# Inspect the retained IDs and sixteen new complex drawing specifications:
.venv/bin/python -m drawing.generate_dataset --warmup-from datasets/pilot-100 --output datasets/sft-warmup-100 --dry-run

# Retain 80 existing one-shot drawings and all 4 existing execution repairs,
# then generate the remaining 16 repairs with the original pilot's teacher:
.venv/bin/python -m drawing.generate_dataset --warmup-from datasets/pilot-100 --output datasets/sft-warmup-100 --timeout-ms 180000

# Resume after provider/connection failures without regenerating successful work:
.venv/bin/python -m drawing.generate_dataset --warmup-from datasets/pilot-100 --output datasets/sft-warmup-100 --timeout-ms 180000 --retry-errors

# If a task exhausted its candidates, try its rejected slots again. Completed
# repair tasks are reused and previous attempts remain recorded:
.venv/bin/python -m drawing.generate_dataset --warmup-from datasets/pilot-100 --output datasets/sft-warmup-100 --timeout-ms 180000 --retry-errors --retry-rejected
```

Warmup generation loads `.env` when present and uses `ARTSY_TEACHER_API_KEY` or
`TEACHER_API_KEY`. Teacher model and endpoint default to the source pilot's
manifest; normal teacher/judge flags still override them. The original pilot is
preserved. To use different paths, run `.venv/bin/python -m drawing.generate_dataset --warmup-from SOURCE
--output OUTPUT` with the same credentials and flags.

The retained one-shot examples are selected by round-robin task family, with one
candidate per prompt and distinct source/pixel hashes. The four natural repairs
are retained even though three were originally rejected solely as duplicate
renders. These explicit exceptions and the original statuses are recorded.

The new repair examples are sampled from a deterministic pool of 128 challenging
compositions (configurable with `--pool-size`, from 16 to 512). They combine
compound paths, tangent and normal geometry, gradients, repeated object groups,
precise counts, transformed local coordinates, and layering. These prompts contain many more constraints than the original one-shot curriculum.
The task types now require computational geometry: Voronoi clipping, Delaunay
triangulation, Apollonian packing, involute gears, spline offsets, marching
squares, projected 3D meshes, Penrose subdivision, planar graph faces, hyperbolic
geodesics, and related constructions. Repeated points, collinear sites, tangent
geometry, and periodic seams are legitimate inputs the requested drawing must
handle correctly.

The model is always asked to produce a correct solution. Its Python code is
executed **verbatim**. No prompt requests mistakes and no code is mutated.
Naturally occurring execution failures and genuine prompt-level visual failures
can both count toward the sixteen new repairs. Every valid preview is inspected
using its actual PNG. Visual correction requests include that PNG and the
observed mismatches against the original prompt. Only successful corrections with a
fresh final render and passing final image review qualify. Examples that are
correct on their first preview, malformed JSON alone, truncation alone, and
infrastructure failures do not count. The sampler stops once it has sixteen successful natural repairs;
any extra successes in its last worker batch remain in `generated/`.

One candidate is sampled per hard prompt by default, with at most two correction
requests. Four independent tasks run at once; use `--workers 1` for sequential
execution. Repair generation allows 8,192 teacher tokens per request so a small
output cap does not manufacture failures. The warmup targets **Qwen/Qwen3.5-0.8B** with text and image observations. By
default the same teacher reviews each rendered preview and fresh final image in
a separate vision context; this is labeled `self-review`, not independent-model
validation. Supply a different vision reviewer with `--judge-model` for an
independent model. A review sees the actual image and source, and the teacher
receives the actual PNG when correcting a visual mismatch.

Raw completions, `trace.executedCode`, image-bearing `trace.observation` records,
and execution trajectories preserve the model's actual programs and observations.
Each observation stores an artifact path and PNG hash. The SFT adapter must load
these images through the model's multimodal processor and mask image/input tokens
from the language-model loss. Failed actions stay in context and are loss-masked.

The discarded synthetic experiment is archived under `datasets/archive/` and is
excluded from the active selection. The earlier natural scene pilot is preserved
at `datasets/natural-scene-pilot/`; its two genuine repairs can be reused with
`--natural-repairs-from datasets/natural-scene-pilot/generated`. Imports must
come from the same teacher and preserve its exact output. Their final PNGs must
pass the current vision review before they count; their source hashes
and IDs are frozen in the manifest. Resume discovers that source from the saved
manifest automatically. The active generator rejects synthetic repair
records and has no error injector. Unmodified first drafts can also be reused
with `--drafts-from datasets/natural-algorithmic-pilot/generated`; they are
re-rendered and inspected, and reused draft calls are excluded from new API
usage totals. Saved draft sources are discovered automatically on resume.

`datasets/sft-warmup-100/` contains:

- `warmup-manifest.json`: frozen retained IDs, provenance, duplicate exceptions,
  complex repair plan, and selection rules.
- `retained.jsonl`: the existing 84 records with copied artifacts.
- `imported-natural-repairs.jsonl`: any genuine repairs reused from an earlier
  sampling run, with copied artifacts and original provenance.
- `generated/`: resumable teacher requests, candidates, source, previews, final
  images, execution feedback, and usage totals for the new repair curriculum.
- `examples.jsonl`, `one-shot.jsonl`, and `repair.jsonl`: the current selection.
- `summary.json`: actual counts, remaining repair quota, and completion status.
- `train.jsonl`: written only when all **80 one-shot + 20 repair** examples exist.

Every selected row has `warmup.split="train"`. Its original `spec.split` remains
provenance: this first run deliberately consumes some former validation/test
examples, so use a new held-out evaluation set. Selection does not imply visual
or safety approval; original status and `humanReviewed` flags remain intact.
Review the final drawings before SFT. Per-record `selectionAudit.eligible=false`
excludes an ambiguous reviewer critique while preserving its raw evidence. Mask
failed assistant actions from
training loss while retaining their code/errors as repair context. A bounded run
that cannot produce all sixteen repairs exits nonzero and withholds `train.jsonl`.

For just the new complex repair curriculum, use `.venv/bin/python -m drawing.generate_dataset --curriculum
repair --output datasets/complex-repairs` with your teacher settings. Its hard-task
pool is also available via `--dry-run`.

## Supervised fine-tuning with TRL

`training/sft.py` uses TRL's `SFTTrainer` and the full
`Qwen3_5ForConditionalGeneration` model, including its vision encoder. It defaults
to the same model revision and image resolution limits as the inference server.
The trainer uses [TRL SFT](https://huggingface.co/docs/trl/sft_trainer) with a custom collator for selective assistant-turn loss.
Use a separate Python 3.12 environment:

```sh
uv venv --python 3.12 .sft-venv
uv pip install --python .sft-venv/bin/python -r training/requirements.txt

# Inspect an actual converted datapoint without loading a model or dependencies:
python3 -m training.sft --dry-run
python3 -m training.sft --dry-run --sample-id d925c993f23372c735329d70-c0
python3 -m training.sft --dry-run --export /tmp/artsy-sft.jsonl

# Load just the processor and verify all image inputs and token/label lengths:
HF_HOME="$PWD/.cache/huggingface" .sft-venv/bin/python -m training.sft --check-batch

# Full-parameter training; defaults: 1 epoch, batch 1, accumulation 8, LR 2e-5.
# Review the selected drawings first. For an experimental run that explicitly
# includes the 84 selected rows still marked pending review:
HF_HOME="$PWD/.cache/huggingface" .sft-venv/bin/python -m training.sft \
  --allow-unreviewed --output-dir checkpoints/sft-warmup
```

Each of the 100 selected records becomes one complete conversation, retaining the
80/20 one-shot/repair mix. The conversation contains the request, earlier attempts
and feedback, the final successful write-and-preview bash call, its recorded PNG,
and the submit call. `trainable_message_indices` identifies the two successful
assistant actions that receive loss. The adapter synthesizes bash wrappers around the teacher's
recorded Python; these are converted demonstrations, not collected Mini SWE
transcripts. Earlier executed failures and malformed replies remain in the conversation
and receive no loss. Prior previews are loaded through the multimodal processor.
Judge scores, critiques, and final approval labels are excluded from model input;
visual repairs must follow the original request and the actual preview image.

Only the final successful program action and submission receive loss. User/system
messages, failed assistant actions, tool outputs, image tokens, assistant headers,
and padding are masked. The collator adds generation spans around the selected
assistant bodies in Qwen's native template and verifies that the rendered text
is unchanged. The multimodal processor maps those spans through image-token
expansion. Thinking is disabled and the bash schema is included. Sequences are never
truncated or packed; every example is checked before model allocation and those
over `--max-length` (default 32,768) fail explicitly. Source/image hashes,
documentation version, training split, and explicit duplicate exceptions are
checked during conversion. Training defaults to the completed warmup `train.jsonl`;
the old `spec.split` is provenance when `warmup.split="train"` is present.

CUDA uses bfloat16 when supported; CPU uses float32. This is full-parameter SFT,
so choose batch size and hardware after measuring memory use. No training run is
started by `--dry-run` or `--check-batch`. Use `--max-steps 1` for an optimizer
smoke run, `--device cpu --dtype float32` to force CPU, and
`--resume-from-checkpoint checkpoints/sft-warmup/checkpoint-N` to resume with
optimizer state. Checkpoints, metrics, and a data/model/version manifest go under
`--output-dir`; the final model and processor are saved in its `final/` directory.
Use the fresh `evaluation/prompts.jsonl` for behavioral evaluation; the script
does not split the warmup records into a misleading held-out set.

The current data preflight measures a maximum of 14,970 conversation tokens and
4,012 supervised tokens across both actions in a conversation. The longest single
program action is 3,978 tokens. The bundled server and Mini SWE configuration currently
cap each response at 2,048 tokens and the server pins the base model. To evaluate
the trained checkpoint, load `final/` in a compatible vision/tool inference service
and choose a sufficient generation limit in both serving and harness configuration.

All commands run directly as Python modules; no Node.js or compilation step is required.
The Makefile provides `serve`, `draw`, `render`, `dataset`, `dataset-warmup`, and
`eval` shortcuts. Pass CLI arguments with `ARGS`, for example
`make dataset ARGS="--dry-run --count 20"`.

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

A live Qwen vision smoke test identified a blue circle and returned a parsed
`report(color="blue", shape="circle")` tool call from the image. The CPU path took
about 107 seconds for that image-plus-tool request; full drawing performance and
quality still need benchmarking.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Inspect SWE](https://meridianlabs-ai.github.io/inspect_swe/),
[Mini SWE Agent](https://mini-swe-agent.com/),
[original Artsy post](https://blog.rashik.sh/tech/can-ai-learn-to-draw/).
