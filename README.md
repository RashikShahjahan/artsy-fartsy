# Artsy standalone Python harness

First vertical slice: hand-written ArtCanvas program → Docker-restricted execution
→ validated PNG → local append-only episode. No backend, UI, database, or model API.
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

Each invocation creates a new UUID directory with `program.py`, `record.json`,
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
