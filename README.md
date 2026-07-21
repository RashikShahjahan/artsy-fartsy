# AI Drawing Application

An interactive React application that generates ArtCanvas Python drawings through OpenRouter and renders them in an isolated Linux sandbox.

## Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS, and DaisyUI
- Backend: Bun and Express
- AI: OpenRouter's `openrouter/free` model router
- Database: PostgreSQL with optional pgvector or Upstash Vector search
- Drawing: Python, Cairo, and the local `ArtCanvas` library
- Sandbox: Bubblewrap namespaces plus `prlimit` resource limits

## Prerequisites

- Bun 1.0 or newer
- Python 3.11 or newer
- PostgreSQL with pgvector, or the included development database
- Linux with `bubblewrap` and `util-linux` for native code execution
- Docker for sandboxed development on macOS

Python execution fails closed when Bubblewrap is unavailable. There is no unsandboxed fallback.

## Local Development

1. Install JavaScript dependencies from each package directory:

   ```bash
   cd shared && bun install
   cd ../frontend && bun install
   cd ../backend && bun install
   cd ..
   ```

2. Create the Python environment at the path used by the sandbox:

   ```bash
   python3 -m venv backend/art_libraries/venv
   backend/art_libraries/venv/bin/pip install pycairo
   ```

3. Start the development database:

   ```bash
   docker compose -f backend/docker-compose.yml up -d
   ```

4. Create `backend/.env`:

   ```dotenv
   OPENROUTER_API_KEY=your_openrouter_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   DATABASE_URL=postgres://postgres:postgres@localhost:15432/codeart
   DATABASE_SSL=false
   EXECUTION_TOKEN_SECRET=replace_with_at_least_32_random_characters
   APP_URL=http://localhost:5173
   ```

5. On Linux, install and build the sandbox tools:

   ```bash
   sudo apt-get install bubblewrap util-linux libseccomp-dev build-essential
   cc -O2 -Wall -Wextra -Werror backend/sandbox/sandbox-init.c -lseccomp -o /tmp/artsy-sandbox-init
   sudo install -m 0755 /tmp/artsy-sandbox-init /usr/local/bin/artsy-sandbox-init
   ```

6. Start the services in separate terminals:

   ```bash
   cd frontend
   bun run dev
   ```

   ```bash
   cd backend
   bun run src/index.ts
   ```

## Vector Search

PostgreSQL pgvector is used when the database supports it. Upstash Vector is optional and uses its standard REST URL directly:

```dotenv
VECTOR_ENDPOINT=https://your-index.upstash.io
VECTOR_TOKEN=your_read_write_token
VECTOR_READONLY_TOKEN=your_optional_read_only_token
```

`VOYAGE_API_KEY` is required to generate 512-dimensional embeddings. When no embedding provider or vector backend is configured, search explicitly falls back to random saved drawings.

Document rows, pgvector embeddings, and pending Upstash writes are committed in one PostgreSQL transaction. Upstash replication is idempotent and retried from the `vector_outbox` table after failures or restarts.

## Database TLS

TLS is enabled with certificate verification whenever `DATABASE_URL` is set. Use `DATABASE_SSL=false` only for a trusted local database. For a private certificate authority, provide its PEM certificate:

```dotenv
DATABASE_SSL=true
DATABASE_CA_CERT="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
```

## Docker Deployment

The production image installs Bubblewrap, runs the application as an unprivileged user, and applies CPU, memory, process, file-size, and wall-clock limits to every drawing execution.

```bash
docker build -t artsy-fartsy .
docker run --rm -p 8000:8000 --security-opt seccomp=unconfined \
  -e OPENROUTER_API_KEY=your_key \
  -e VOYAGE_API_KEY=your_key \
  -e DATABASE_URL=your_database_url \
  -e DATABASE_SSL=true \
  -e EXECUTION_TOKEN_SECRET=your_random_secret_of_at_least_32_characters \
  artsy-fartsy
```

The container host must permit unprivileged user namespaces. Docker's default outer seccomp profile commonly blocks Bubblewrap setup, so the example disables that outer profile; untrusted Python still runs under the inner seccomp filter and unprivileged Bubblewrap namespaces. Never run this image with `--privileged`. The backend performs a real sandboxed drawing during startup and fails closed if the host cannot provide the isolation boundary.

Set `TRUST_PROXY_HOPS=1` only when exactly one trusted reverse proxy sits in front of the backend, as in the included Fly configuration. Direct Docker deployments leave proxy trust disabled so clients cannot spoof rate-limit identities with `X-Forwarded-For`.

For Fly.io custom domains, provision the certificate with `fly certs add <hostname>` and configure the DNS records Fly reports; custom domains are not declared in `fly.toml`.

## Sandbox Configuration

The defaults permit two concurrent executions, 30 seconds of wall time, 10 seconds of CPU time, 256 MiB of address space, and 8 MiB output files. Seccomp blocks networking, subprocesses, additional threads, namespace changes, and other privileged kernel interfaces. Deployments can lower concurrency or wall time with:

```dotenv
SANDBOX_MAX_CONCURRENT=2
SANDBOX_TIMEOUT_MS=30000
```
