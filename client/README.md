# Artsy client

The consumer drawing app owns its Next.js frontend, Python API, drawing worker,
process lifecycle, speech integration, and run storage. It does not import or
start the internal UI or its server.

From the repository root, with the project Python environment installed and
Docker Desktop running:

```sh
make client
```

Open http://127.0.0.1:3000. This command installs frontend dependencies if missing,
starts the dedicated consumer API on port 3001, starts or reuses the core native
inference bridge on port 8080, and starts Next.js. Ctrl-C stops the processes it
started, including active drawing workers. An already-running inference bridge
is left running. Node.js 22.18+ and npm are required. Drawing generation requires
the project's Tinker credentials, completed model checkpoint, and renderer image
(see the repository setup instructions).

To browse and inspect saved drawings without starting inference:

```sh
make client ARGS="--no-inference"
```

Voice input uses browser speech recognition and microphone permission. Optional
Kokoro speech output can be installed with `make client-tts-setup`; typing and
drawing work without it.

```sh
make client-check       # TypeScript, frontend tests, and consumer API/lifecycle tests
make client-build       # production frontend build
make client ARGS="--production"
```

The browser calls same-origin Next.js routes. These forward only drawing,
status, PNG, cancellation, and speech requests to the consumer API. The consumer
API exposes no training, review, or internal workspace routes. This is a local,
single-user app; the API binds to loopback.

The default model is the completed RLHF checkpoint
`artsy/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery`.
The consumer backend shares the core `evaluation.draw` execution harness and
`inference.server` bridge; neither requires the internal UI.

Configuration is supplied to `make client` through environment variables:

- `ARTSY_CLIENT_PORT`: frontend port, default `3000`.
- `ARTSY_CLIENT_API_PORT`: consumer API port, default `3001`.
- `ARTSY_CLIENT_JOBS`: run directory, default `episodes/client/`.
- `ARTSY_CLIENT_MODEL`: drawing model, default RLHF checkpoint above.
- `ARTSY_EVAL_PYTHON`: optional worker interpreter override.

The launcher sets the server-only `ARTSY_CLIENT_API_URL` for Next.js. If starting
Next.js and the consumer API manually, this defaults to `http://127.0.0.1:3001`.
The old `ARTSY_BACKEND_URL` and `ARTSY_UI_*` settings are not used by this app.

History is saved in browser storage; drawing artifacts live in `episodes/client/`.
Existing browser history entries referring to internal-server artifacts will show
as unavailable; they can be restarted from their original idea. Client and
internal runs have separate storage and process ownership.

Structure:

- `app/`: page shell, Tailwind theme and base styles, and consumer API proxy.
- `components/`: drawing workspace, versions, activity, and artwork recovery.
- `hooks/`: run polling and browser voice lifecycle.
- `lib/`: shared types, browser history, and frontend policy tests.
- `backend/`: dedicated API, worker, launcher, speech, and backend tests.

UI styling uses Tailwind CSS v4 through `@tailwindcss/postcss`. Theme tokens live
in `app/globals.css`; layout and state utilities live alongside the components.
Shared button styles in `components/ui.ts` keep controls consistent.
