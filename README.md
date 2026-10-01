# Artsy standalone Python harness

Drawing prompt → local Ollama → ArtCanvas program → Docker-restricted execution
→ validated PNG → local append-only episode. Also accepts hand-written programs.
No backend, UI, database, or hosted-provider fallback.
The original ArtCanvas behavior lives in `src/artsy_harness/artcanvas.py`.

## Run

Requires Python 3.11+ and a running Docker engine (Docker Desktop on macOS).

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
docker build -t artsy-renderer:0.1.0 .
.venv/bin/artsy-harness
.venv/bin/python -m unittest discover -s tests -v
ARTSY_DOCKER_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

Use `--program path/to/drawing.py` for a different hand-written program,
`--episodes path/to/episodes` for storage, or `--image` for the renderer image.
Programs import `ArtCanvas` from `artcanvas` and save `/output/output.png`.

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
`--model-timeout` (600 seconds) bounds the HTTP call, not Docker rendering.
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

To run the opt-in live generation test (requires Ollama, model, and Docker):

```bash
ARTSY_OLLAMA_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

This test keeps its episode under `episodes/`, including failures.

Each invocation creates a new UUID directory with `record.json`, `program.py` when available,
and, on success, `image.png`. Existing episodes are never reused or modified by
the harness. Failures also get records and exit status 1. Records include exact
source, source/image hashes, Docker image ID, runner limits, bounded stdout/stderr,
return status, and PNG dimensions. Append-only describes harness behavior, not
filesystem tamper protection. Preserve the renderer image by its recorded ID for
replay; rebuilding a tag may change OS dependencies.

## Execution boundary

There is **no host Python execution or unsandboxed fallback**. Docker runs as an
unprivileged UID with no network, dropped capabilities, no new privileges, a
read-only root, Docker's default seccomp profile, and CPU/memory/process/file-size
limits. A 30-second host timeout forcibly removes the container. Only the program
file (read-only) and an isolated temporary output directory are mounted. No keys,
Docker socket, or episode directory are exposed. Logs are bounded; PNGs must be
regular non-symlink files, at most 8 MiB and 4096×4096, and decode successfully.

Docker and its host kernel are trusted; this is not a VM-grade security boundary
for hostile multi-tenant execution. Resource/isolation tests are deliberately
opt-in and require the built image. Ordinary unit tests use mocked execution and
do not establish that real sandboxed rendering works. Image harm evaluation is
separate from generated-code security and is not implemented in this slice.
