# Use multi-stage build
FROM oven/bun:latest as builder

# Install Python and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files first for better caching
COPY shared/package*.json /app/shared/
COPY frontend/package*.json /app/frontend/
COPY backend/package*.json /app/backend/

# Install dependencies
WORKDIR /app/shared
RUN bun install

WORKDIR /app/frontend
RUN bun install

WORKDIR /app/backend
RUN bun install

# Copy rest of the application
COPY shared/ /app/shared/
COPY frontend/ /app/frontend/
COPY backend/ /app/backend/

# Build frontend
WORKDIR /app/frontend
RUN bun run build

# Setup Python environment
WORKDIR /app/backend/art_libraries
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install pycairo

# Start production image
FROM oven/bun:latest

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built assets and backend
COPY --from=builder /app/frontend/dist ./frontend/dist
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/backend/src/venv ./backend/src/venv
COPY --from=builder /app/shared ./shared

# Set environment variables
ENV NODE_ENV=production
ENV PATH="/app/backend/src/venv/bin:$PATH"
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Start the application
CMD cd backend && bun run src/index.ts