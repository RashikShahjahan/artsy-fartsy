# Artsy client

The consumer drawing experience is a standalone Next.js App Router application
with React and strict TypeScript. The Python API and internal workspaces remain
in `../ui/`. This app imports no scripts, styles, or templates from the internal UI.

Use Node.js 22.18+ and npm. From the repository root:

```sh
npm ci --prefix client
make ui                 # terminal 1: Python service and inference bridge
make client             # terminal 2: Next.js at http://127.0.0.1:3000
```

Start Docker before generating drawings. For browsing without the inference
bridge, run `.venv/bin/python -m ui.server` instead of `make ui`. Voice output
uses the existing Kokoro service; install its assets with `make tts-setup`.
Voice input requires browser speech recognition and microphone permission.
Typing remains available in browsers without recognition support.

The server-only `ARTSY_BACKEND_URL` defaults to `http://127.0.0.1:4173`.
Copy `.env.example` to `.env.local` to override it. The backend must be reachable
from the Next.js server. The browser uses same-origin Next.js API routes.
The proxy allows consumer generation, speech, individual run
status, PNG artifacts, and cancellation. Internal training and review routes
are not proxied. This remains a local application: authentication and
multi-user isolation are not provided by the Python backend.

```sh
make client-check       # strict TypeScript and local tests, no model calls
make client-build       # production build
npm --prefix client start
```

The backend selects the default drawing model; the client has no model settings.

For isolated testing, set `ARTSY_UI_PORT=4174` and
`ARTSY_UI_JOBS="$PWD/episodes/ui-client-test"` when starting the Python backend,
and `ARTSY_BACKEND_URL=http://127.0.0.1:4174` when starting Next.js.

The interface supports new drawings, edits from any selected version, image
downloads, browser-local history, recorded activity, voice, cancellation, and
reload recovery with retry/backoff. Browser storage is optional. It keeps the
legacy sketchbook schema, but browsers isolate storage by origin: history from
port 4173 does not automatically appear on port 3000. Run files stay in
`episodes/ui/` on the backend. The Python home page remains a compatibility
entry point; new consumer UI development belongs here. Internal Studio remains
at http://127.0.0.1:4173/studio.

Structure:

- `app/`: page shell, metadata, global styles, and the restricted API proxy.
- `components/`: drawing workspace, activity, and icons.
- `hooks/`: abortable run polling and browser voice lifecycle.
- `lib/`: shared types, sketchbook validation, and proxy policy tests.
