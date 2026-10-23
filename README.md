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
safety. Teacher data generation, visual grading, content rewards/penalties, SFT,
and RL are subsequent checklist items.

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
