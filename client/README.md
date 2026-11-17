# Artsy client

Artsy is one Next.js application. Its same-origin routes start drawings, report
progress, cancel jobs, restore history, and serve private artifacts. Vercel
Workflow coordinates jobs across requests; no separate API server or domain is
required.

The existing Python mini-SWE agent and native Tinker backend run in a trusted
Vercel Sandbox. Every generated shell command runs in a second Sandbox with
network access denied and no credentials. The agent exchanges command requests
and results through files relayed by Workflow steps. Final submissions are
rendered again in a fresh directory using the existing Python rendering logic.

Job metadata, progress, cancellation, and session history live in Upstash Redis.
Images, editable Python source, previews, and agent traces live in a **private**
Vercel Blob store and are served through the app after checking session ownership.
History survives reloads and application restarts. A random HttpOnly cookie owns
the sketchbook; clearing cookies creates a new sketchbook. This is anonymous
session ownership, with no account or cross-device history synchronization.

## Setup

Use Node.js 22.18 or newer. Connect Upstash Redis and a private Vercel Blob store
to the Vercel `artsy` project, then configure the variables in
[.env.example](.env.example). Keep Tinker credentials server-only. Set
`ARTSY_TINKER_MODEL_PATH` to the existing completed RLHF sampler checkpoint URI
(`tinker://…`); the hosted runner does not depend on local checkpoint discovery.
The established model is
`artsy/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery`.

Production can use a prepared Sandbox snapshot through
`ARTSY_SANDBOX_SNAPSHOT`; it takes precedence over `ARTSY_SANDBOX_IMAGE`. The
`client/sandbox/prepare.sh` script installs the runtime from a source-only archive
in a Python 3.14 Sandbox. Snapshot that environment after its smoke test and use
the snapshot ID for both job VMs. Snapshots have bounded retention and must be
refreshed before they expire.

Alternatively, build and publish the Python runtime image to Vercel Container Registry,
using the repository root as the build context and `client/sandbox/Dockerfile`
as the Dockerfile. For example, with the Vercel CLI authenticated (see the
[Container Registry CLI reference](https://vercel.com/docs/container-registry/cli-reference)):

```sh
vercel vcr login docker --project artsy
vercel vcr build docker . artsy-runtime:latest --project artsy --push -- -f client/sandbox/Dockerfile
```

Set `ARTSY_SANDBOX_IMAGE` to the resulting VCR image reference, preferably pinned
to a digest. The Dockerfile-specific ignore file limits the build context to
runtime inputs. The image contains the trusted runner, existing Python drawing
logic, native Tinker dependencies, ArtCanvas, and Cairo. Sandbox starts commands
explicitly; the image needs no API listener or Docker daemon.

For local development, link `client/` to the Vercel project and pull its environment:

```sh
cd client
vercel link
vercel env pull .env.local
npm ci
npm run dev
```

Sandbox authenticates automatically with deployment OIDC on Vercel. Local OIDC
credentials expire; pull the environment again when needed. External callers
can instead configure `VERCEL_TOKEN`, `VERCEL_TEAM_ID`, and `VERCEL_PROJECT_ID`.
Workflow uses its local runtime during development and Vercel's durable runtime
when deployed. Deploy the `client/` directory as the Next.js application.

From the repository root, `make client` starts Next.js only. `make client-check`
runs TypeScript checks, frontend/workflow tests, and the legacy Python backend
tests. `make client-build` builds the production application. The Python Sandbox
adapter smoke test runs with `.venv/bin/python -m unittest client.sandbox.test_agent`.
For environments that block Turbopack worker ports, use
`npm --prefix client run build -- --webpack`.

## Job behavior

- `POST /api/artsy/runs`: accepts a prompt and optional owned `parent_id`, returns
  a job ID immediately. Each session can have one active drawing.
- `GET /api/runs`: restores the session's latest 100 jobs and active drawing.
- `GET /api/runs/:id`: returns saved state and activity for polling.
- `POST /api/runs/:id/cancel`: records cancellation before stopping both VMs.
  The workflow remains responsible for cleanup if cancellation arrives during startup.
- `GET /api/runs/:id/image.png`, `program.py`, `trace.json`, and
  `preview-N-N.png`: serve owned artifacts from private Blob storage.

Jobs have ten model turns and a twenty-minute VM lifetime. Partial work is saved
with editable source when the turn limit is reached. Failed jobs remain in job
history; the visual sketchbook displays completed and paused drawings. Workflow
steps save progress before acknowledging actions. Side-effectful agent startup
and shell execution do not automatically retry, avoiding duplicate inference or
shell actions after ambiguous failures. Infrastructure failures mark the job
failed, trigger VM cleanup, and let the user retry as a new drawing.

Voice input uses browser speech recognition. Optional speech output uses the
browser's speech synthesis, so it needs no Python TTS server.

The legacy local Python API and launcher remain in `backend/` for existing tests
and tools; the Next.js app no longer calls or starts them. Existing local artifacts
are not automatically uploaded or adopted into a new hosted session.

The application is deployed to [artsy.rashik.sh](https://artsy.rashik.sh/) on the
Vercel `artsy` project. Production uses the free `artsy-jobs` Upstash Redis store,
the private `artsy-drawings` Blob store, and the prepared runtime snapshot
`snap_xeqX5HATHWMEXFjsGfXuiRkbDwhu`. The snapshot expires after 30 days without use;
refresh it with the process in [sandbox/README.md](sandbox/README.md) if it expires.

Deployment `dpl_GBihCTAb6Uak1uLoxwq8nia53sWz` was promoted after a real Tinker
job generated a 1920×1200 image, saved source and trace files, restored session
history, denied another session access, and successfully cancelled a second
running job. No job VMs remained running after verification.

The completed SFT, RLVR, and RLHF adapters are also stored in public Hugging Face
repositories. Repository IDs and verified commit revisions are recorded in
[`inference/huggingface_models.json`](../inference/huggingface_models.json).
To export completed checkpoints again, run
`.venv/bin/python -m scripts.publish_models` with Tinker credentials and a
Hugging Face login.
