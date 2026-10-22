# Artsy drawing environment

LiquidAI/LFM2.5-230M → Pi Durable → Python/ArtCanvas → PNG + editable source.

The first four checklist items are implemented: the Liquid model adapter, the
original drawing API, a persistent Pi Durable coding harness for an isolated VM, and a write → run → fix → submit loop.

## Setup

Requires Node 22.19+ (tested with Node 26), Python 3.12, `uv`, and Cairo.
Run the application in an isolated VM; the runner executes code directly with
the VM user’s environment and permissions. Run commands from this repository's root.

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
`runtime/artcanvas.py` is unchanged from Artsy's original implementation at
commit `1468463`. The Python runtime uses Cairo via compatible `cairocffi` bindings.
Generated programs import `ArtCanvas` from `artcanvas` and save `output.png`.
The default dimensions remain 1920×1200; drawings can explicitly choose another
size up to 4096×4096. A fixed Python random seed is injected for every render.

## Harness and execution

Pi Durable 1.4.0 owns the model/tool loop, transcript, tool tasks, and SQLite
checkpoints. A drawing extension exposes `write_code`, `read_code`, `run_code`,
and `submit`. Programs and tool feedback are persisted by Pi. The model can
replace a broken program and retry; final submission always re-runs the latest
source in a fresh temporary working directory and validates the PNG.

Each render runs a local Python process in a fresh temporary working directory.
The external VM supplies isolation; there is no application sandbox or Docker
requirement. The runner keeps a ten-second timeout, bounded captured logs,
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
