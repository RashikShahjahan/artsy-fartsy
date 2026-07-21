FROM oven/bun:1.3.14 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libcairo2-dev \
    libseccomp-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/package.json shared/bun.lock /app/shared/
COPY frontend/package.json frontend/bun.lockb /app/frontend/
COPY backend/package.json backend/bun.lockb /app/backend/

WORKDIR /app/shared
RUN bun install --frozen-lockfile

WORKDIR /app/frontend
RUN bun install --frozen-lockfile

WORKDIR /app/backend
RUN bun install --frozen-lockfile

COPY shared/ /app/shared/
COPY frontend/ /app/frontend/
COPY backend/ /app/backend/

RUN cc -O2 -Wall -Wextra -Werror /app/backend/sandbox/sandbox-init.c -lseccomp -o /app/artsy-sandbox-init

WORKDIR /app/frontend
RUN bun run build

WORKDIR /app/backend/art_libraries
RUN python3 -m venv venv \
    && . venv/bin/activate \
    && pip install --no-cache-dir pycairo

FROM oven/bun:1.3.14

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libcairo2 \
    libseccomp2 \
    bubblewrap \
    util-linux \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/bin/bwrap --version \
    && /usr/bin/prlimit --version

WORKDIR /app

COPY --from=builder /app/frontend/dist ./frontend/dist
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/shared ./shared
COPY --from=builder /app/artsy-sandbox-init /usr/local/bin/artsy-sandbox-init

RUN groupadd --system --gid 10001 artsy \
    && useradd --system --uid 10001 --gid artsy --home-dir /app --shell /usr/sbin/nologin artsy \
    && mkdir -p /app/backend/art_libraries/output \
    && chown -R artsy:artsy /app/backend/art_libraries/output

ENV NODE_ENV=production
ENV PORT=8000

EXPOSE 8000

USER artsy

CMD ["bun", "run", "backend/src/index.ts"]
