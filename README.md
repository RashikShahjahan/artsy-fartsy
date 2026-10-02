# Artsy standalone Python harness

Drawing prompt → local Ollama → ArtCanvas program → local Python execution
→ validated PNG → local append-only episode. Also accepts hand-written programs.
No backend, UI, database, or hosted-provider fallback.
The original ArtCanvas behavior lives in `src/artsy_harness/artcanvas.py`.

## Run

Requires Python 3.11+ and Cairo. On macOS, install Cairo and its build tools with
`brew install cairo pkg-config`; on Debian/Ubuntu, install `libcairo2-dev pkg-config python3-dev`.
**Drawing programs run locally without a sandbox, with your user’s file and network access.**

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/artsy-harness
.venv/bin/python -m unittest discover -s tests -v
```

Use `--program path/to/drawing.py` for a different hand-written program,
`--episodes path/to/episodes` for storage.
Programs import `ArtCanvas` from `artcanvas` and save `output.png` in their current
directory (a temporary rendering directory). Older programs using `/output/output.png`
must be updated to the relative path.

## Generate with Ollama

Start a local server and install the model (macOS with Homebrew):

```bash
brew install ollama
ollama serve
# In another terminal:
ollama pull qwen3.5:9b-q4_K_M
```

Then generate a drawing:

```bash
.venv/bin/artsy-harness --prompt "Draw a blue circle on a white background"
```

The model defaults to `qwen3.5:9b-q4_K_M` (4-bit Q4_K_M); `--model` overrides it.
Thinking is explicitly enabled by default; `--think` enables it and `--no-think`
disables it. This setting is saved in the episode's generation request along with
the decoding options. The exact HTTP body retains any separate thinking field;
only Ollama's `response` field is used verbatim as the program.
`--ollama-url`
defaults to `http://localhost:11434`. Decoding options are `--seed` (0),
`--temperature` (0.2), `--top-p` (0.9), and `--num-predict` (8192).
`--num-ctx` (16384) sets the loaded context size. Thinking and final code share
the generation budget; context must accommodate both the prompt and generation.
`--model-timeout` (600 seconds) bounds the HTTP call, not local rendering.
These larger limits permit longer thinking, not guaranteed valid code, and may
increase generation time and memory use. All sent settings are recorded.
`--prompt` and `--program` are mutually exclusive. The CLI connects to an existing
server; it does not start one or download models.

Each generated episode records the versioned prompt, endpoint, requested model,
returned model/digest when available, and every sent request parameter. It saves
the exact HTTP body in `response.body`, decoded response text in `generated.txt`,
and the same text verbatim in `program.py`. There is no fence removal, stripping,
or repair. Empty output and Python syntax errors are recorded without rendering;
syntax checking does not execute code on the host. Connection, HTTP/protocol,
syntax, execution, and PNG-validation failures remain separate statuses.

Model tags can move; Ollama may not return a digest. A seed does not guarantee
bit-for-bit reproducibility across models, runtimes, or hardware.

To run the opt-in live generation test (requires Ollama and the model):

```bash
ARTSY_OLLAMA_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

This test keeps its episode under `episodes/`, including failures.

Each invocation creates a new UUID directory with `record.json`, `program.py` when available,
and, on success, `image.png`. Existing episodes are never reused or modified by
the harness. Failures also get records and exit status 1. Records include exact
source, source/image hashes, Python executable/version, timeout, bounded stdout/stderr,
return status, and PNG dimensions. Append-only describes harness behavior, not
filesystem tamper protection. Local dependencies and hardware can affect replay.

## Execution boundary

Rendering uses the current Python interpreter in a subprocess on macOS/Linux.
It is **not sandboxed**: programs can read or modify host files, use the network,
access inherited environment variables, and launch processes. Only run trusted
programs; model-generated code is not inherently trustworthy.

A 30-second timeout kills the rendering process group, but is not a security
boundary (programs can escape that group). CPU, memory, process count, and disk
writes are not restricted. Captured logs are bounded to 64 KiB per stream; PNGs
must be regular non-symlink files, at most 8 MiB and 4096×4096, and decode successfully.
Tests include real local rendering; live Ollama tests remain opt-in. Image harm
evaluation is not implemented in this slice.
