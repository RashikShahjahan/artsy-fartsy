# Artsy drawing environment

LiquidAI/LFM2.5-230M → Pi Durable → Python/ArtCanvas → PNG + editable source.

Includes the Liquid model adapter, a versioned drawing API, a model-independent
episode environment, and a persistent Pi Durable write → run → fix → submit loop.

## Setup

Requires Node 22.19+ (tested with Node 26), Python 3.12, `uv`, and Cairo.
Execution is local for this phase: programs run with the current user’s
environment and permissions. No sandbox or Docker setup is required. Run commands from this repository's root.

```sh
npm ci
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r inference/requirements.lock -r runtime/requirements.txt

# Starts a loopback-only server; the first start downloads the 230M model.
HF_HOME="$PWD/.cache/huggingface" npm run serve
```

In another terminal:

```sh
npm run draw -- "Draw a blue circle centered on a white background."

# Choose a persistent episode directory and seed:
npm run draw -- "Draw three red squares." --episode episodes/squares --seed 42
```

The CLI prints a submission record. Each episode saves `session.sqlite`, previews,
`program.py`, `image.png`, and `result.json` with source/image hashes. Opening the
same episode with the same prompt reuses the submission; interrupted Pi tasks
resume from storage. An interrupted render is marked unsafe to replay, so the
model receives interruption feedback rather than silently executing it twice.
Use a new episode directory for a new drawing request. Only one process should
own an episode database at a time.

`ARTSY_MODEL_URL` can point to another OpenAI-compatible serving endpoint, using
the same model identifier and tool-call support. The bundled CPU server translates
Liquid's Python-style function calls into the OpenAI tool protocol. It parses
literal arguments with Python's AST; it never evaluates generated expressions.
Requests are serialized and capped at 2,048 generated tokens. This is a small
development server, not a public API service.

## Drawing API

[ArtCanvas documentation](docs/artcanvas.md) lists every original method.
`runtime/artcanvas.py` extends Artsy's original implementation with composable
paths, transforms, dimension validation, and idempotent saving. The Python runtime
uses Cairo via compatible `cairocffi` bindings. The runtime ID is `artcanvas-v2`.
Generated programs import `ArtCanvas` from `artcanvas` and save `output.png`.
The default dimensions remain 1920×1200; drawings can explicitly choose another
size up to 4096×4096. A fixed Python random seed is injected for every render.

## Harness and execution

Pi Durable 1.4.0 owns the model/tool loop, transcript, tool tasks, and SQLite
checkpoints. A drawing extension exposes `write_code`, `read_code`, `run_code`,
`submit`, and `refuse(reason)`. Programs and tool feedback are persisted by Pi. The model can
replace a broken program and retry; final submission always re-runs the latest
source in a fresh temporary working directory and validates the PNG.

Each render runs a local Python process in a fresh temporary working directory.
There is no application sandbox or Docker requirement. The runner keeps a ten-second timeout, bounded captured logs,
process-group cleanup on Unix, PNG validation, and temporary-directory cleanup.
Set `ARTSY_PYTHON` to select the Python interpreter (defaults to `.venv/bin/python`
when present, otherwise `python3`). Install Cairo with `sudo apt-get install
libcairo2 fonts-dejavu-core` on Debian/Ubuntu, or `brew install cairo` on macOS.

Limits are in `src/config.ts`: three preview renders, sixteen model turns/tool
calls, a five-minute episode deadline, 64 KiB source, 8 MiB PNG, and bounded logs.
The model is text-only and receives errors/dimensions, not visual observations.
Rendering success establishes code validity, not instruction accuracy or image
safety. The dataset generator below supports teacher generation and an independent
visual evaluator. Reward calibration, SFT, and RL are subsequent checklist items.

## Model-independent episodes

Run an existing program without a model server:

```sh
npm run render -- examples/transforms.py --prompt "Draw twelve golden shapes" --seed 42
```

Teacher generation, training rollouts, and evaluations can use the same local
renderer through `DrawingEnvironment` in `src/environment.ts`:

```ts
import { DrawingEnvironment } from "./src/environment.js";

const env = new DrawingEnvironment("episodes/pilot");
await env.reset("Draw a blue circle", 42);
await env.write_code(source);
const preview = await env.run_code();
// Use errors to repair with write_code(), then execute the latest source afresh:
const result = await env.submit();
// For an unsupported request, finish with env.refuse(reason) instead.
```

`ENVIRONMENT_CONFIG` versions the defaults: 1920×1200 output, three previews,
sixteen operations, a five-minute episode deadline, 32,768 generated tokens,
64 KiB source, and 8 MiB PNG. Supply a config copy to choose dimensions or budgets.
Training adapters report generated tokens through `chargeTokens(count)`; the core
does not tokenize model output. The Pi adapter separately bounds model turns and
tool calls using its durable state. It retains support for variable canvas sizes.

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
use one environment instance per rollout. Reset starts a new episode; the Pi harness
provides persistent model-session resume.

Seeds reproduce Python `random` and hash ordering within the same runtime.
Python packages are pinned in `runtime/requirements.txt`; development currently
uses Python 3.12.12. System Cairo and font versions still depend on the host, so
cross-platform pixel identity is not guaranteed.

## Generate a dataset

Preview the seeded curriculum without model requests or code execution:

```sh
npm run dataset -- --dry-run --count 20 --seed 42
```

Generate programs with a stronger teacher served by a chat-completions endpoint.
Set the model identifier and endpoint to the teacher you have available:

```sh
export ARTSY_TEACHER_MODEL=your-teacher-model
export ARTSY_TEACHER_URL=http://127.0.0.1:9000/v1
# Set ARTSY_TEACHER_API_KEY if that endpoint requires authentication.

npm run dataset -- --count 20 --candidates 2 --repairs 2 --output datasets/pilot
```

The endpoint must support non-streaming `POST /chat/completions`, `messages`, and
`max_tokens`, returning a text completion containing the requested JSON object.
The bundled 230M server implements that protocol, but is the student baseline;
select a stronger teacher for training data. Each candidate gets at most three
teacher requests and three previews, with fresh final execution through
`DrawingEnvironment`. Teacher replies are either `{"code":"complete Python"}` or
`{"refusal":"reason"}`. Syntax errors, API errors, and failed visual requirements
can trigger repairs. Generated programs use the existing local executor and its
current permissions and resource limits.

To automatically accept examples, add an independent **vision** evaluator with
support for PNG data URLs in chat `image_url` content:

```sh
export ARTSY_JUDGE_MODEL=your-vision-evaluator
export ARTSY_JUDGE_URL=http://127.0.0.1:9001/v1
# Set ARTSY_JUDGE_API_KEY if needed.

npm run dataset -- --count 20 --candidates 2 --output datasets/graded-pilot
```

The evaluator checks every valid preview and the freshly submitted image for
safety and each requirement. Acceptance requires `safe` and a score of at least
0.85 on **every** requirement; use `--threshold` to change it. Unsafe previews
terminate the candidate. Uncertain safety goes to review. Correct refusals need
an explicitly labeled refusal task and evaluator approval. With no evaluator,
executable examples and expected refusals go to `review.jsonl`; the accepted
training files stay empty. Model approval is provisional: human auditing and
judge calibration remain separate work.

The built-in curriculum covers 12 primitive, composition, scene, and procedural
families. Counts up to 10,000 are supported; start with 20, then run a 1,000-prompt
pilot in a new directory. The seed reproduces prompt specifications and render
randomness; remote teacher replies may vary. Whole families have fixed splits
before wording and parameters are generated. Increasing the count preserves the
existing prompt prefix. Custom JSONL specifications can include drawing and
refusal examples:

```json
{"id":"blue-circle","family":"circles","prompt":"Draw one blue circle on white.","requirements":["One centered blue circle","White background"],"expectedOutcome":"drawing","seed":42}
```

Pass `--input prompts.jsonl` instead of `--count`. The generator derives the split
from `family` and rejects conflicting splits, duplicate prompts, and unsafe IDs.
Unknown families receive a stable hash-based 80/10/10 split; small custom inputs
may not cover all splits. Keep related custom templates in the same family.
The built-in curriculum contains benign drawing tasks; provide your own labeled
refusal tasks and rubric if you need refusal data.

Each output directory contains:

- `manifest.json` and `prompts.jsonl`: configuration, prompt plan, model endpoints,
  API documentation hash, runtime source/dependency hashes, and environment limits.
- `records/*.json`: atomic candidate checkpoints with prompt requirements, code
  or refusal, completion usage/provenance, repair trace, evaluator results, seed,
  hashes, split, and relative artifact references. Failed attempts remain available.
- `artifacts/episode-*/`: the environment's complete operation trajectory,
  previews, final Python source, and final PNG.
- `accepted.jsonl`, `review.jsonl`, `rejected.jsonl`, and `candidates.jsonl`:
  evaluated examples, pending review, rejection/infrastructure reasons, and all records.
- `train.jsonl`, `validation.jsonl`, and `test.jsonl`: accepted records by split.
- `summary.json`: acceptance rate, outcome counts, recorded request/token usage,
  and the count of completions with unavailable usage. These are observed totals,
  not a price estimate; requests interrupted before a response may still be billed.

Exact normalized source and decoded pixel duplicates are excluded across candidates
and splits. Near-duplicate image detection and human audits are not implemented.
Credentials come only from environment variables and are excluded from the manifest.
Outputs are ignored by Git.

Repeat the same command to resume completed checkpoints without model calls.
Use `--retry-errors` to retry infrastructure failures; previous records and their
reported token usage are retained. Changed configuration, documentation, or runtime
sources require a new output directory. Aggregated JSONL files are rebuilt after
all candidates finish. Only one generator may own a directory; if a process is
forcibly stopped, remove its `.generation.lock` only after verifying it has exited.
Use `npm run dataset -- --help` for all options.

## Verification

```sh
npm run check
npm test
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
npm run build
```

Tests use Pi's scripted provider to verify repair, fresh final execution,
persistent reopening, and failure budgets. Local execution tests exercise the
original API, error reporting, timeouts, symlink rejection, and a complete repair
episode through Pi. Install the runtime dependencies above before running them.
The scripted provider is test-only; `npm run draw` uses Liquid.

The initial live 230M baseline connected and issued tool calls, but generated
invalid drawing code and exhausted its episode budget. No valid submission was
reported. The successful repair integration test uses a scripted model; SFT is
still needed to teach the small model this task.

References: [Liquid model card](https://huggingface.co/LiquidAI/LFM2.5-230M),
[Pi Durable](https://github.com/earendil-works/pi/tree/main/packages/durable),
[original Artsy post](https://blog.rashik.sh/tech/can-ai-learn-to-draw/).
