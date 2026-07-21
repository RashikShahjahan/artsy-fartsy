# Backend

Install dependencies:

```bash
bun install
```

Create the drawing environment from the repository root:

```bash
python3 -m venv backend/art_libraries/venv
backend/art_libraries/venv/bin/pip install pycairo
```

Create `backend/.env` as described in the root README, then run:

```bash
bun run src/index.ts
```

Python execution requires Bubblewrap, `prlimit`, and the seccomp launcher on Linux. Build the launcher from the repository root:

```bash
cc -O2 -Wall -Wextra -Werror backend/sandbox/sandbox-init.c -lseccomp -o /tmp/artsy-sandbox-init
sudo install -m 0755 /tmp/artsy-sandbox-init /usr/local/bin/artsy-sandbox-init
```

Execution fails closed if any sandbox component is unavailable.
